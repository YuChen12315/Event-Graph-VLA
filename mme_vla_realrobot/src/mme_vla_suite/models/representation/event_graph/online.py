"""在线推理路径：逐帧喂，和离线落盘走**同一个** EventGraphController。

v1.3 §10 Phase 6 的验收是「graphmem 不得回退 frame_sampling，同一个 Controller
在 policy server 真实增量运行」。这个类就是那个增量运行的地方。

为什么不是「离线算好的搬过来」
---------------------------------------------------------------------------
Controller 是逐帧推进的状态机：第 t 帧看见哪些节点取决于第 0..t 帧。在线做评测时
没有预计算文件可读，也没有未来帧可看。所以这里把同一份输入按到达顺序喂进去，
逐步问它「现在图上是什么」。

和离线落盘的关系
---------------------------------------------------------------------------
离线路径： precompute_eventgraph_memory 逐帧 step → nodes.npz / steps.npz
          → dataset._load_eventgraph_mem 查表 → readout.pack_eg_tensors
在线路径： 本类逐帧 step → 活节点表 → readout.pack_eg_tensors

两边**共用的**是 Controller（切分、medoid、晋升、合并）和 readout（张量拼装）。
各自独有的只有「状态存在哪」：一个在磁盘、一个在内存。所以
`scripts/diag/eventgraph_p6_parity.py` 能把两条路逐字节比出结果 —— 那正是 §11
「逐位一致性测试未通过对不上不许报 eval 数字」要的那道门。

代价与边界
---------------------------------------------------------------------------
本类不复制 2048-d payload：代表帧和最近帧都直接读基类已经存下的 image_emb_4x4
（§11 禁止拿 512-d cache 顶替 2048-d payload）。自己额外留的只有每帧 16×512 的
冻结 PCA token（约 16 KB/帧）—— Phase 3 的合并要按帧号回头查它。
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from mme_vla_suite.shared.mem_buffer import MemoryBuffer

from .config import GraphMemoryConfig
from .controller import PHASE_EXEC, PHASE_VIDEO, EventGraphController
from .frame_summary import FrozenFrameSummary
from .readout import pack_eg_tensors

__all__ = ["EventGraphMemoryBuffer"]


class EventGraphMemoryBuffer(MemoryBuffer):
    """在线增量版。继承 MemoryBuffer 只为复用「图像 → SigLIP → 4x4 token」那一段。

    用法（policy server 侧）::

        buf = EventGraphMemoryBuffer(cfg, pca, dims=..., normalize_state=fn,
                                     num_views=1, img_emb_dim=2048, pos_emb_dim=768,
                                     state_emb_dim=8, vision_enc_fn=enc)
        buf.add_buffer(images, states, step_idx_list, exec_start_idx=...)
        mem = buf.memory()          # 15 个 eg_* 张量，与训练侧同一个 readout 产出
    """

    def __init__(
        self,
        cfg: GraphMemoryConfig,
        pca: FrozenFrameSummary,
        *,
        dims: tuple[int, int, int],
        normalize_state: Callable[[np.ndarray], np.ndarray],
        exec_start_idx: int = 0,
        **mb_kwargs,
    ):
        """
        dims            : (d1, d2, d3) 视觉 token / 位置编码 / 关节角三个宽度，与
                          dataset 侧取自同一处（yaml 的 memory_feature）。
        normalize_state : 模型输入用的关节角归一化；**不是** Control Plane 内部
                          按 cfg.state_mean/std 算「动静」的那份。
        exec_start_idx  : §5.1 的阶段分界（第几帧起进入执行阶段）。counting 数据集
                          全是 0，但 video 分支必须留着 —— 其它 RoboMME 任务有。
        """
        mb_kwargs.setdefault("prepare_buffer", True)
        mb_kwargs.setdefault("num_views", 1)
        super().__init__(**mb_kwargs)

        self.cfg = cfg
        self.pca = pca
        self.dims = dims
        self._normalize_state = normalize_state
        self.exec_start_idx = int(exec_start_idx)

        self.ctl = EventGraphController(cfg, record_trace=False,
                                        pca_tokens_fn=self._pca_tokens)
        self.ctl.reset(phase=PHASE_EXEC if self.exec_start_idx <= 0 else PHASE_VIDEO)

        # 逐帧的冻结 PCA token / 帧摘要 / 原始 state。前两个只给 Controller 用
        # （合并要按帧号回查 token），state 还要供 readout 填 eg_node_state。
        self._tok: dict[int, np.ndarray] = {}
        self._states: list[np.ndarray] = []
        self._arrival = 0

    # ── 喂帧 ───────────────────────────────────────────────────────────
    def add_buffer(self, images, states, step_idx_list, exec_start_idx=None, precomputed_emb=None):
        """与基类同名同形，只是多一步「顺手把图往前推一格」。

        exec_start_idx 在**这一批帧入图之前**更新：评测协议是第一批就把整段
        conditioned video 发过来并带上 exec_start_idx，所以要在这批里立刻生效。
        """
        if exec_start_idx is not None and int(exec_start_idx) > 0:
            self.exec_start_idx = int(exec_start_idx)
        super().add_buffer(images, states, step_idx_list, precomputed_emb=precomputed_emb)
        for s in step_idx_list:
            self._ingest(int(s))

    def _ingest(self, s: int) -> None:
        """一帧入图：算 f_t → 推进 Controller。"""
        if s != self._arrival:
            # eg_node_state / frame_states 是按帧号直接索引的（与 store.py 的
            # 「frame_id == 行号」同一个约定）。跳号会静默读错行，所以在这里炸。
            raise ValueError(
                f"在线帧号必须从 0 起连续：第 {self._arrival} 帧给的 step_idx={s}。"
                f"eg_node_state 按帧号索引，跳号会静默读错行。")

        img, pos, state = self._frame_views(s)
        # 冻结 PCA：**先落回 cache 精度再池化**，与离线读 event_sum 走同一条路。
        # 顺序反过来（先在 fp32 上池化再压）会得到与离线不同的 f_t —— 而且差得很小，
        # 小到看不出是错，只是在线线下的切分慢慢对不上。
        tok = self.pca.project(img, cache_dtype=self.cfg.event_cache_dtype)   # (P, E) fp16
        ft = self.pca.summarize(tok)                                          # (E,) fp32

        self._tok[s] = tok
        self._states.append(np.asarray(state, np.float32).reshape(-1))

        # §5.1：exec_start_idx 之前是 conditioned video，也必须进同一张图
        phase = PHASE_EXEC if self._arrival >= self.exec_start_idx else PHASE_VIDEO
        self.ctl.step(ft, state=state, timestamp=s, phase=phase)
        self._arrival += 1

    # ── 读出 ───────────────────────────────────────────────────────────
    def memory(self) -> dict[str, np.ndarray]:
        """当前这一帧的 15 个 eg_* 张量（不含 batch 维，与训练侧同形同 dtype）。

        每次都现算：Controller 的状态是活的，上一次读出的结果这一帧就不成立了。
        """
        gs = self.ctl.get_graph_state()
        assert self._states, "还没喂过任何帧"
        return pack_eg_tensors(
            cfg=self.cfg,
            node_rows={
                "frame_ids": gs.frame_ids,
                "frame_mask": gs.frame_mask,
                "rep_frame": gs.rep_frame,
                "start_frame": gs.start_frame,
                "end_frame": gs.end_frame,
                "event_mass": gs.event_mass,
                "boundary_strength": gs.boundary_strength,
                "phase_code": gs.phase_code,
                "grasp_count": gs.grasp_count,
            },
            # GraphState 已经把槽位排好了（左 padding、最新在最高槽位），所以可见的
            # 就是 node_mask 为真的那些槽 —— 行号即槽号，无需再映射一次。
            vis_ids=np.flatnonzero(gs.node_mask),
            recent_ids=gs.recent_frame_ids,
            recent_mask=gs.recent_mask,
            frame_states=np.stack(self._states, axis=0),
            dims=self.dims,
            read_payload=self._payload,
            read_payload_many=self._payload_many,
            read_pca_tokens=self._pca_tokens,
            normalize_state=self._normalize_state,
            exec_start_idx=int(self.exec_start_idx),
        )

    # ── 取帧 ───────────────────────────────────────────────────────────
    def _frame_views(self, s: int):
        """基类已经存好的那一帧：(img [P,d1], pos [P,d2], state [d3])。

        与训练侧 dataset._eg_raw_frame 读的是同一份东西 —— 都是 4x4 的原始视觉
        token，不是投影后的 cache。
        """
        f = self._history_feats[s]
        img = np.asarray(f["image_emb_4x4"], np.float32)
        pos = np.asarray(f["pos_emb_4x4"], np.float32)
        if img.ndim == 3:          # (v, P, d) → 只取主视角
            img = img[0]
        if pos.ndim == 3:
            pos = pos[0]
        state = np.asarray(f["state_emb"], np.float32).reshape(-1)
        return img, pos, state

    def _payload(self, frame_id: int):
        return self._frame_views(int(frame_id))

    def _payload_many(self, frame_ids):
        """一次取多帧 payload —— 与离线紧凑缓存同一个 readout 接口。

        在线这边数据已经在 `_history_feats` 里，没有磁盘 I/O，逐帧 stack 即可；
        但走同一条 `read_payload_many` 路径，离线/在线的拼装代码才真正是同一份。
        """
        idx = np.asarray(frame_ids, np.int64).reshape(-1)
        views = [self._payload(int(f)) for f in idx]
        return (np.stack([v[0] for v in views]),
                np.stack([v[1] for v in views]),
                np.stack([v[2] for v in views]))

    def _pca_tokens(self, frame_ids) -> np.ndarray:
        """帧号 → (n, P, E) fp16 冻结 token。Controller 合并时按帧号回查。"""
        idx = np.asarray(frame_ids, np.int64).reshape(-1)
        missing = [int(i) for i in idx if int(i) not in self._tok]
        if missing:
            # 只可能是「合并要用的帧还没喂进来」。真出现说明 Controller 看到的帧号
            # 超出了喂进去的范围 —— 那种情况下它算出来的 v̄ 是错的，不能糊弄过去。
            raise KeyError(
                f"这些帧还没喂进来，拿不到冻结 PCA token：{missing[:5]}"
                f"（共 {len(missing)} 个）。")
        return np.stack([self._tok[int(i)] for i in idx], axis=0)
