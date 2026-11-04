"""EventGraphController —— Control Plane 的唯一实现（v1.3 §3 / §5 / §6）。

**全仓库只有这一个类做切分**（§6 硬性要求）：离线预处理、训练 replay、在线
mem_buffer/policy server 全走它 + 同一个 GraphMemoryConfig，不许把 boundary /
scheduling 逻辑复制到别处。理由很实际：两边切出来的边界只要不一样，Data Plane 拿到的
就是两套对不上的输入 —— 这种不一致不报错，只让指标悄悄变差。

**无参数、确定、不吃梯度**（§1-1）：没有可训练模块也没有随机数，同样输入跑两遍逐位相同。

═══════════════════════════════════════════════════════════════════════════
边界怎么切：安静点切分（替代 v1.3 §2.2/§2.3 的「三项证据 + 滑动门槛」）
═══════════════════════════════════════════════════════════════════════════

⚠️ 这是**有意偏离规范**，别按规范改回去。原方案算异常分
B_t = 画面预测误差 + 关节变化量 + 画面偏离度，超过滑动门槛就切。2026-09-24 实测
推翻了它的物理前提（4 个 Counting 任务 × 25 条 episode）：

  · 真值 is_subgoal_boundary 标在**机械臂最安静的那一帧**：边界处关节变化量 0.022，
    全 episode 均值 0.095，落在最安静的 10~19% 分位，前后是个 V 形。394 个边界点
    全是这个形状，四个任务一致。
  · 而 B 的三项全是「变化越大分越高」，方向正好相反 —— 命中率 9.1%，比随机放 7 个点
    （22.1%）还差，单段最多跨 8 个子目标。
  · 「与先前轨迹差异最大」也排除了：边界处的分位是 0.36~0.53（0.5 = 没信号）。谷底
    那帧恰恰最"听话"，完全符合「减速然后停住」的预期，意外检测器不报警。

现方案：**在最安静的地方下刀**。停在哪儿说明正处在两段动作之间，在那儿切不可能把
一个动作劈成两半。每帧的「动静」信号 = 相邻两帧关节角变化幅度，没给 state（机械臂
冻住、真在看视频）就用画面摘要变化幅度。以 cut_block 帧为一格，格内挑最安静的一帧
下刀；两刀至少隔 min_event_len 帧，单段超 cut_max_len 帧强制补刀兜底。

判据只看**本条 episode 内部的相对大小**，不跟全局比，所以机械臂整体快慢都不影响。
实测 StopCube 有 69% 的帧是静止的，用「低于全局中位数」判据只切得出 2.6 段、单段
最多跨 8 个子目标；格子制压到最多 2 个。四个任务合计单段最多跨 1~2 个子目标。

**没有门槛常数、不需要标定** —— 格子大小 / 封顶 / 间隔都是结构性参数，不是从数据里
拟合的。所以 scripts/diag/eventgraph_calibrate.py 已废弃。

---------------------------------------------------------------------------
step() 的处理顺序（顺序本身就是规范的一部分）
---------------------------------------------------------------------------

    ① phase 变了（video→exec）：finalize 当前 Event、清空信号缓冲，但**不清空**
       Event Graph，也**不清空** Recent 窗口（§5.1）。
    ② 算本帧「动静」信号 —— 用**更新前**的上一帧状态，只看过去。
    ③ 因果滑动平均（只看过去 cut_smooth 帧）。
    ④ 本帧进 Active Event，也进当前格子。
    ⑤ 格子满 ⇒ 格内挑最安静的一帧下刀。
    ⑥ Active Event 超 cut_max_len 帧 ⇒ 强制补刀（格子可能一个坑都没有）。
    ⑦ Pending 里「终点已完全滑出 Recent 窗口」的 Event 晋升 Long-Term。

Episode 结束调 `flush()` 关掉最后一个 Event。

---------------------------------------------------------------------------
三处容易写错、且写错不会报错的地方
---------------------------------------------------------------------------
**(a) 下刀点可能落在 Active Event 的中间。**
   格子攒满才知道哪一帧最安静，而那一帧可能已经在 29 帧之前了。所以切分是回填的：
   把 Active Event 从那一帧劈成两段 —— 直接劈开就是对的，但 (b) 的检查不能省。

**(b) 一格只能出一刀。**
   回填之后，同一格的**后半段**又攒够 min_event_len 的话会再出一刀 —— 那一格就出了
   两刀，切分密度不再是配置说的那个数。用 `_active[0].arrival >= 格子起点` 拦掉：
   成立就说明这一格已经出过刀了。

**(c) 窗口 / 晋升一律按「到达顺序」而不是 timestamp 算术。**
   timestamp 只是 label（pkl 的 step_idx），允许跳号；用 timestamp 做减法算窗口
   一遇到跳号就会漏帧或多帧。

---------------------------------------------------------------------------
Phase 3：图满了怎么办（v1.3 §4）
---------------------------------------------------------------------------

节点数超过 max_nodes 时把**相邻**的一对并成一个，直到回到容量内。挑 C 最大的一对：

    C_i = cos(v̄_i, v̄_i+1) − λ_b · b_i

v̄_i 是节点自己 L×16 个冻结 PCA token 的均值再归一（§4.1）。最大 C = 「最像 + 分隔
最不明确」，相似度高的两个相邻段本来就是一个动作被切碎了。

三个容易被"优化"掉、但必须保持的约束：

  · **只用相邻对**。不做全局语义最近邻 —— Counting 里「又放了一个方块」会反复出现
    几乎一样的画面，按全局相似度合并等于把重复证据抹平（§4.1 / §11）。
  · **不依赖 h_event**。Control Plane 读不到 Data Plane 当前 checkpoint 的编码，
    v1.0 那个 cos(h_i, h_i+1) 在这里根本算不出来（§11 禁止）。
  · **合并只改 metadata**。新节点的 h_event 由 Data Plane 在 forward 里拿合并后的
    新 L=8 真实帧重新编码（§4.2），所以这里只要换掉 `frame_ids`，没有「两个旧编码
    怎么取平均」的问题。

合并要 `pca_tokens_fn`（算 v̄ 和重采样后的 medoid）。不传也能跑完，只是图满时退化成
丢弃最旧的节点并告警 —— 那条路径只给诊断用，正式跑必须带读出器。

---------------------------------------------------------------------------
夹爪计数（「已经抓了几次」这个量）
---------------------------------------------------------------------------

Control Plane 顺手从关节角里的夹爪宽度**数**出一个累计成功抓取数，挂在节点上当标量
特征。不是新机制、不改切分 —— 只是又一路「无参数、确定、不吃梯度」的读数。

为什么由 Control Plane 数而不是让 Data Plane 学：夹爪的三态（张开 0.0398 / 夹住
0.0183 / 夹空 0.0000）是物理事实，用两个常数就能**精确**分开，实测 4 任务 × 100 条
episode 逐条对得上真值。让网络从画面里学「到这个画面时已经放了三个方块」，是拿几万条
梯度去逼近一个查表就能得到的整数。

⚠️ 它**不参与切分**：改判据不会移动任何一条 event 边界，影响面只有新加的那一列输入。
   用不上的任务（StopCube 真值恒为 0）由 Data Plane 自己学成忽略。

---------------------------------------------------------------------------
数值确定性
---------------------------------------------------------------------------
余弦与向量长度一律用显式乘加（`np.sum(a*b)`）而不是 `np.dot`/`np.linalg.norm`：后者
走 BLAS，线程数 / 后端不同可能改变归约顺序。Event 内均值用显式顺序累加，不用
`mean(axis=0)` 的成对求和。滑动平均用 Python 的 sum 而不是 `np.mean`。这些量最终
决定 argmin（下刀点与代表帧），差 1 ulp 就可能翻转结果。
"""
from __future__ import annotations

import math
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .config import GraphMemoryConfig
from .frame_summary import _l2_normalize as l2_normalize

__all__ = [
    "EventGraphController", "EventNode", "GraphState", "StepInfo",
    "sample_event_frames", "PHASE_VIDEO", "PHASE_EXEC",
]

PHASE_VIDEO = "video"
PHASE_EXEC = "exec"
_PHASES = (PHASE_VIDEO, PHASE_EXEC)

# Event 是怎么被切下来的。
CUT_EPISODE_START = "episode_start"
CUT_BOUNDARY = "boundary"        # 格子里的安静点
CUT_MAX_LEN = "max_len"          # 长度封顶，强制补刀
CUT_PHASE_SWITCH = "phase_switch"
CUT_EPISODE_END = "episode_end"
CUT_MERGE = "merge"              # 这一段的起点是某次相邻合并造出来的（假的边界）

# 「动静」信号从哪来
SIGNAL_STATE = "state"      # 关节角变化幅度（首选）
SIGNAL_VISUAL = "visual"    # 画面摘要变化幅度（没给 state 时的退路）

# node.state
PENDING = "pending"
LONG_TERM = "long_term"


# ══════════════════════════════════════════════════════════════════════
# 无 BLAS 的小工具
# ══════════════════════════════════════════════════════════════════════
def _cos(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """余弦相似度。显式乘加 + float64 累加，不走 BLAS。"""
    af = a.astype(np.float64, copy=False)
    bf = b.astype(np.float64, copy=False)
    na = math.sqrt(float(np.sum(af * af)))
    nb = math.sqrt(float(np.sum(bf * bf)))
    return float(np.sum(af * bf)) / max(na * nb, eps)


def _norm2(a: np.ndarray) -> float:
    """L2 长度。同样避开 np.linalg.norm（可能走 BLAS 的 dnrm2）。"""
    af = a.astype(np.float64, copy=False)
    return math.sqrt(float(np.sum(af * af)))


def _ordered_mean(rows: Sequence[np.ndarray]) -> np.ndarray:
    """按顺序累加求均值。

    `np.stack(rows).mean(axis=0)` 用的是成对求和，行数不同归约树就不同。这里干脆
    不依赖「两条路径上行数一定相同」这个前提。
    """
    acc = np.zeros_like(rows[0], dtype=np.float32)
    for r in rows:
        acc += r
    return acc / np.float32(len(rows))


def sample_event_frames(n_frames: int, rep_pos: int, L: int) -> list[int]:
    """从 Event 的真实帧里挑 L 帧（v1.3 §1-3 / §2.3）。

    必选 {首帧, 代表帧, 末帧}，其余在内部帧里等距取；不足 L 帧时全取
    （调用方靠 frame_mask 知道哪些有效）。

    返回的是 **位置下标**（0..n_frames-1），已升序去重，长度 ≤ L。
    """
    if n_frames <= 0:
        return []
    if n_frames <= L:
        return list(range(n_frames))

    mandatory = {0, n_frames - 1, int(rep_pos)}
    need = L - len(mandatory)
    interior = [i for i in range(1, n_frames - 1) if i not in mandatory]

    chosen: list[int] = []
    if need > 0 and interior:
        if len(interior) <= need:
            chosen = list(interior)
        else:
            idx = np.linspace(0, len(interior) - 1, need).round().astype(np.int64)
            chosen = [interior[int(i)] for i in sorted(set(idx.tolist()))]

    return sorted(mandatory | set(chosen))


# ══════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════
@dataclass
class _Frame:
    """Active Event 里的一帧。

    `sig` 是本帧的**因果**滑动平均信号（只用本帧及之前 cut_smooth 帧算出来）。
    格子挑最安静的一帧时读它。

    `grasp_cum` 是截至本帧的累计成功抓取数（见 `_update_grasp_count`）。关节点拿
    **末帧**的这个值当自己的计数。
    """
    frame: int
    arrival: int
    f: np.ndarray
    sig: float
    grasp_cum: int


@dataclass
class EventNode:
    """一个已关闭的 Event（v1.3 §3）。

    `frame_ids` / `frame_mask` 是交给 Data Plane 的 L=8 采样帧（真实 frame id）。
    `boundary_strength` ∈ [0,1]：这一刀切在多安静的地方 —— 格内均值与刀口值的相对
    落差，供 Phase 3 的 merge 当惩罚项用；它不是原方案那个 margin。

    Phase 3（§4）额外挂四个字段。**合并只改 metadata** —— 理由见 `_merge_pair`。
    """
    event_id: int
    phase: str
    cut_reason: str
    start_frame: int
    end_frame: int
    start_arrival: int
    end_arrival: int
    n_frames: int
    rep_frame: int
    frame_ids: np.ndarray          # (≤L,) int64，升序
    frame_mask: np.ndarray         # (≤L,) bool
    boundary_strength: float
    state: str = PENDING
    promote_step: int | None = None      # 晋升 Long-Term 时的 step（timestamp）
    promote_arrival: int | None = None
    # ── Phase 3（§4） ──
    event_mass: int = 0                  # 证据量。合并时相加（§4.2）
    # 截至本 Event 末帧的**累计成功抓取数**（Control Plane 从夹爪读数直接数出来的，
    # 无参数、不学）。给「数循环次数」这类任务用：BinFill 需要知道「已经放进去几个」。
    #
    # ⚠️ 合并时**取 b 的值、不能相加** —— 它是累计读数（截至某一时刻的计数），
    #    不是证据体积。把两个相邻段的读数相加会凭空造出抓取次数。
    grasp_count: int = 0
    provenance: tuple[int, ...] = ()     # 由哪些原始 event_id 合成。合并时拼接
    # v̄_i（§4.1）：节点自己 L×16 个 PCA token 的均值再归一。**无参数、不学**。
    # 拿不到 token 读出器时为 None —— 那种情况下不允许合并（见 _merge_once）。
    fixed_summary: np.ndarray | None = None
    merged_from: tuple[int, int] | None = None   # 被合并掉的两个 event_id
    merge_arrival: int | None = None     # 被并掉的到达序号：这之后的帧看不见它了

    def __post_init__(self) -> None:
        # 新建节点没显式给 event_mass 时，默认就是它的帧数（未合并过的新事件）。
        if self.event_mass == 0:
            self.event_mass = int(self.n_frames)
        if not self.provenance:
            self.provenance = (int(self.event_id),)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "phase": self.phase,
            "cut_reason": self.cut_reason,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "n_frames": self.n_frames,
            "rep_frame": self.rep_frame,
            "frame_ids": self.frame_ids.tolist(),
            "boundary_strength": self.boundary_strength,
            "state": self.state,
            "promote_step": self.promote_step,
            "event_mass": self.event_mass,
            "grasp_count": self.grasp_count,
            "provenance": list(self.provenance),
            "merged_from": list(self.merged_from) if self.merged_from else None,
        }


@dataclass
class StepInfo:
    """step() 的一次返回值，同时进 self.trace。

    `signal` 是本帧的动静原始值，`smoothed` 是因果滑动平均后的值（格子挑最小值时
    用的是后者）。`signal_source` 说明这一帧用的是关节角还是画面。
    """
    arrival: int
    timestamp: int
    phase: str
    signal: float
    smoothed: float
    signal_source: str
    confirmed_boundary: bool = False
    boundary_frame: int | None = None
    boundary_strength: float | None = None
    cut_reason: str | None = None
    rejected_short: bool = False         # 格子里的候选点离上一刀太近，作废
    n_long_term: int = 0
    n_pending: int = 0
    n_grasps: int = 0                    # 截至本帧的累计成功抓取数（诊断用）


@dataclass
class GraphState:
    """交给 Data Plane / 诊断的图状态快照（v1.3 §5.2 的槽位布局）。

    slots 按时间升序；**最新节点在最高有效槽位**，不足 max_nodes 时在左边补零，
    padding 全程靠 mask 排除。
    """
    node_mask: np.ndarray          # (max_nodes,) bool
    frame_ids: np.ndarray          # (max_nodes, event_len) int64，-1 = pad
    frame_mask: np.ndarray         # (max_nodes, event_len) bool
    rep_frame: np.ndarray          # (max_nodes,) int64，-1 = pad
    start_frame: np.ndarray        # (max_nodes,) int64，-1 = pad
    end_frame: np.ndarray          # (max_nodes,) int64，-1 = pad
    n_frames: np.ndarray           # (max_nodes,) int64
    event_mass: np.ndarray         # (max_nodes,) int64  证据量（§4.2，合并时相加）
    grasp_count: np.ndarray        # (max_nodes,) int64  截至该节点末帧的累计成功抓取数
    boundary_strength: np.ndarray  # (max_nodes,) float32
    phase_code: np.ndarray         # (max_nodes,) int8  0=video 1=exec
    event_id: np.ndarray           # (max_nodes,) int64，-1 = pad
    recent_frame_ids: np.ndarray   # (recent_window,) int64，-1 = pad
    recent_mask: np.ndarray        # (recent_window,) bool
    n_nodes: int
    n_pending: int
    n_dropped: int                 # 超出 max_nodes 被挤掉的节点数（Phase 3 merge 前应为 0）

    def to_dict(self) -> dict[str, Any]:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in self.__dict__.items()}

    def diff(self, other: "GraphState") -> dict[str, str]:
        """逐字段逐位比较，返回「不一致的字段 → 说明」。

        v1.2 §3.3 / v1.3 §10 的一致性验收直接用这个：不是「差不多」，是 tobytes 相等。
        """
        out: dict[str, str] = {}
        for k, v in self.__dict__.items():
            w = getattr(other, k, None)
            if isinstance(v, np.ndarray):
                if not isinstance(w, np.ndarray) or v.shape != w.shape or v.dtype != w.dtype:
                    out[k] = (f"形状/dtype 不同：{v.shape}/{v.dtype} vs "
                              f"{getattr(w, 'shape', None)}/{getattr(w, 'dtype', None)}")
                elif v.tobytes() != w.tobytes():
                    out[k] = f"{int(np.count_nonzero(v != w))} 个元素不同（共 {v.size}）"
            elif v != w:
                out[k] = f"{v!r} vs {w!r}"
        return out

    def summary(self) -> str:
        rows = []
        for i in np.flatnonzero(self.node_mask):
            ph = PHASE_EXEC if self.phase_code[i] else PHASE_VIDEO
            rows.append(
                f"    slot{i:>3}  ev{self.event_id[i]:>3}  "
                f"[{self.start_frame[i]:>4},{self.end_frame[i]:>4}]  "
                f"n={self.n_frames[i]:>3}  rep={self.rep_frame[i]:>4}  {ph:>5}  "
                f"b={self.boundary_strength[i]:.3f}  g={self.grasp_count[i]:>2}  "
                f"frames={self.frame_ids[i][self.frame_mask[i]].tolist()}")
        head = (f"  节点 {self.n_nodes}（Long-Term），Pending {self.n_pending}，"
                f"丢弃 {self.n_dropped}；"
                f"Recent={self.recent_frame_ids[self.recent_mask].tolist()}")
        return head + ("\n" + "\n".join(rows) if rows else "")


# ══════════════════════════════════════════════════════════════════════
# 控制器
# ══════════════════════════════════════════════════════════════════════
class EventGraphController:
    """Control Plane 状态机（v1.3 §6 的唯一 API）。"""

    def __init__(
        self,
        cfg: GraphMemoryConfig,
        *,
        record_trace: bool = True,
        pca_tokens_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """
        pca_tokens_fn : frame_ids (l,) int → (l,16,512) fp16。传
            `EventFeatureStore.pca_tokens` 绑定方法即可。**只有 Phase 3 的合并要用
            它**（算 v̄_i 和合并后重采样的 medoid），不传也能跑，只是图满时退化成
            「丢弃最旧的节点」并告警。

            为什么用回调而不是直接收一个 store：Control Plane 不许依赖 Data Plane
            的 checkpoint（§11），但它**可以**读那个冻结的 PCA 资产 —— PCA 是离线
            拟合死的，与训练无关。收回调把这条边界摆在签名上：控制器拿到的只是一张
            「帧号 → 冻结 token」的查表，别的它碰不到。
        """
        self.cfg = cfg
        self.record_trace = record_trace
        self._pca_tokens = pca_tokens_fn
        self.reset(phase=PHASE_VIDEO)

    # ── 生命周期 ───────────────────────────────────────────────────
    def reset(self, *, phase: str = PHASE_VIDEO) -> None:
        """开始新 episode。phase 只决定初始阶段，不预先建 Event。"""
        if phase not in _PHASES:
            raise ValueError(f"phase 必须是 {_PHASES} 之一，得到 {phase!r}")
        cfg = self.cfg
        self._phase: str = phase

        self._n_frames = 0          # 已见帧数 = 下一帧的 arrival 序号
        self._n_events = 0
        self._capacity_warned = False
        self._src_warned = False

        # 当前 Active Event
        self._active: list[_Frame] | None = None

        # 夹爪计数：当前「闭合段」内见过的最小夹爪宽度（None = 当前不在闭合段），
        # 以及已确认的成功抓取总数。见 _update_grasp_count。
        self._grasp_min: float | None = None
        self._n_grasps = 0
        # hand_closure 模式的滞回状态：当前是否处在「手闭合」段。见 _update_hand_grasp。
        self._hand_closed = False

        # 信号：上一帧（算差分用）+ 因果滑动平均的历史 + 当前格子
        self._prev_f: np.ndarray | None = None
        self._prev_state: np.ndarray | None = None
        self._raw_hist: deque[float] = deque(maxlen=cfg.cut_smooth)
        self._signal_source: str | None = None
        self._block: list[tuple[int, float]] = []   # (arrival, 平滑后信号)

        # 最近 recent_window 帧的 (arrival, frame)
        self._recent: deque[tuple[int, int]] = deque(maxlen=cfg.recent_window)

        # 图
        self._pending: deque[EventNode] = deque()
        self._long_term: list[EventNode] = []

        # Phase 3 合并流水：(到达序号, 被并掉的 event_id_a, _b, 新 event_id)。
        # 必须单独记一份：被并掉的两个节点就从 _long_term 里消失了，光看最终节点表
        # 只能知道「谁合成了谁」，不知道**什么时候**合的 —— 而离线落盘要知道每个节点
        # 在哪一帧"死掉"（这之后的帧不该再看见它）。
        self.merge_log: list[tuple[int, int, int, int]] = []

        self.trace: list[StepInfo] = []

    def step(
        self,
        frozen_frame_summary: np.ndarray,
        state: np.ndarray | None = None,
        timestamp: int | None = None,
        phase: str | None = None,
    ) -> StepInfo:
        """喂一帧。

        frozen_frame_summary : (event_dim,) 的 f_t（已 PCA + L2 归一化）
        state                : **原始** state（本类内部按 state_mean/std 归一化）。
                               给了就用它算「动静」信号；不给则退回用画面摘要。
                               实测 video 阶段多数任务机械臂照样在动，所以 video
                               阶段给了 state 也是用的（原方案在这里强制忽略）。
        timestamp            : 该帧的 label（用 pkl 的 step_idx）。缺省则按到达顺序编号。
                               窗口/晋升一律按**到达顺序**算，所以 timestamp 允许有跳号。
        phase                : 'video' / 'exec'；与上一帧不同即触发阶段切换（§5.1）
        """
        cfg = self.cfg
        ts = self._n_frames if timestamp is None else int(timestamp)
        ph = self._phase if phase is None else phase
        if ph not in _PHASES:
            raise ValueError(f"phase 必须是 {_PHASES} 之一，得到 {ph!r}")

        # ① 阶段切换：强制关掉当前 Event + 清信号缓冲，但不清图、不清 Recent（§5.1）
        #    切出来的那一刀要记在**本帧**的 StepInfo 上 —— 它发生在 info 创建之前，
        #    不显式带过去的话，诊断脚本按 trace 数边界就会漏掉这一刀，
        #    而它恰恰是最不该漏的一刀。
        sw_node = self._switch_phase(ph) if ph != self._phase else None

        arrival = self._n_frames
        self._n_frames += 1

        f = np.asarray(frozen_frame_summary, np.float32).reshape(-1)
        if f.shape != (cfg.event_dim,):
            raise ValueError(
                f"frozen_frame_summary 形状应为 ({cfg.event_dim},)，得到 {f.shape}")
        f = l2_normalize(f)

        # ② 本帧的「动静」信号 —— 用更新前的上一帧，所以只看过去
        s_tilde, src = self._state_signal(state)
        self._note_signal_source(src)
        raw = self._raw_signal(f, s_tilde, src)

        # ③ 因果滑动平均（含本帧）
        self._raw_hist.append(raw)
        sig = sum(self._raw_hist) / len(self._raw_hist)

        # ④ 本帧进 Active Event，也进当前格子
        #    夹爪计数必须在 append **之前**更新，本帧才带得上本帧的读数。
        self._update_grasp_count(state)
        if self._active is None:
            self._active = []
        self._active.append(_Frame(frame=ts, arrival=arrival, f=f, sig=sig,
                                   grasp_cum=self._n_grasps))
        self._recent.append((arrival, ts))
        self._block.append((arrival, sig))

        self._prev_f = f
        self._prev_state = s_tilde

        info = StepInfo(arrival=arrival, timestamp=ts, phase=ph,
                        signal=raw, smoothed=sig, signal_source=src,
                        n_grasps=self._n_grasps)
        if sw_node is not None:
            info.confirmed_boundary = True
            info.boundary_frame = arrival           # 新阶段从本帧开始
            info.boundary_strength = sw_node.boundary_strength
            info.cut_reason = CUT_PHASE_SWITCH

        # ⑤ 格子攒满 ⇒ 挑格子内最安静的一帧下刀（封顶也在这一步里补，见 _resolve_block）
        #
        #    封顶**必须挂在格子解析上，不能逐帧做**。逐帧做会出这种事故：格子 30 帧、
        #    封顶 50 帧，某条 episode 的安静点在第 44 帧，而格子要到第 59 帧才攒满、
        #    才知道该切 44 —— 可逐帧的封顶在第 50 帧就已经抢先把 0~49 划成一段了，
        #    44 那个安静点被划进上一段，这一格白攒。实测这一条会把「安静点」的意义
        #    整个抹掉，退化成每 30 帧均匀切一刀。
        if len(self._block) >= cfg.cut_block:
            self._resolve_block(info)

        # ⑦ 晋升
        self._promote(arrival, ts)

        info.n_long_term = len(self._long_term)
        info.n_pending = len(self._pending)
        if self.record_trace:
            self.trace.append(info)
        return info

    def flush(self) -> list[EventNode]:
        """Episode 结束：关掉剩下的 Active Event。

        返回本次关掉的**全部**节点（通常是 1 个；只有收尾那段超过 cut_max_len 帧时
        才会先补几刀封顶，返回多个）。没有 Active Event 时返回空列表。

        末尾那一个不会立刻晋升 —— 它的帧就是 Recent 窗口里的帧，按 §7「Recent 与
        Long-Term 时间互斥」，它应当由 Recent 分支代表。要等本 episode 之后再来
        recent_window 帧才会晋升，而那已经是下一个 episode 了（reset 会清图）。
        所以 episode 末尾的这最后一个 Event 在本次 episode 内始终是 Pending，
        这是设计如此，不是漏了。
        """
        before = self._n_events
        self._apply_cap(None)                    # 收尾这段超过封顶也要切开
        self._finalize(self._active, cut_reason=CUT_EPISODE_END,
                       boundary_strength=0.0)
        self._active = None
        self._block = []
        return self._pending_or_long(before)

    def _pending_or_long(self, since: int) -> list[EventNode]:
        """取出 event_id >= since 的节点（刚被 flush 关掉的那些）；按 event_id 升序。"""
        out = [nd for nd in self._long_term if nd.event_id >= since]
        out += [nd for nd in self._pending if nd.event_id >= since]
        return sorted(out, key=lambda nd: nd.event_id)

    # ── 读状态 ─────────────────────────────────────────────────────
    def get_graph_state(self) -> GraphState:
        """当前图快照（§5.2 槽位对齐）。"""
        cfg = self.cfg
        M, L, R = cfg.max_nodes, cfg.event_len, cfg.recent_window

        n_total = len(self._long_term)
        n_dropped = max(0, n_total - M)
        if n_dropped and not self._capacity_warned:
            warnings.warn(
                f"Long-Term 节点数 {n_total} 超过 max_nodes={M}，最旧的 {n_dropped} 个"
                f"被挤出。说明 Phase 3 的相邻合并没能把图压回容量内 —— 要么没给 "
                f"pca_tokens_fn（合并不了），要么所有候选对的 C_i 都是 -inf。"
                f"被丢弃是**静默的证据损失**，出现这条告警就不该出正式评测数字。",
                RuntimeWarning, stacklevel=2)
            self._capacity_warned = True
        used = self._long_term[n_dropped:]

        node_mask = np.zeros(M, bool)
        frame_ids = np.full((M, L), -1, np.int64)
        frame_mask = np.zeros((M, L), bool)
        rep_frame = np.full(M, -1, np.int64)
        start_frame = np.full(M, -1, np.int64)
        end_frame = np.full(M, -1, np.int64)
        n_frames = np.zeros(M, np.int64)
        event_mass = np.zeros(M, np.int64)
        grasp_count = np.zeros(M, np.int64)
        b_strength = np.zeros(M, np.float32)
        phase_code = np.zeros(M, np.int8)
        event_id = np.full(M, -1, np.int64)

        V = len(used)
        base = M - V                                    # 左 padding，最新在最高槽位
        for j, nd in enumerate(used):
            i = base + j
            node_mask[i] = True
            k = len(nd.frame_ids)
            frame_ids[i, :k] = nd.frame_ids
            frame_mask[i, :k] = nd.frame_mask
            rep_frame[i] = nd.rep_frame
            start_frame[i] = nd.start_frame
            end_frame[i] = nd.end_frame
            n_frames[i] = nd.n_frames
            event_mass[i] = nd.event_mass
            grasp_count[i] = nd.grasp_count
            b_strength[i] = nd.boundary_strength
            phase_code[i] = 0 if nd.phase == PHASE_VIDEO else 1
            event_id[i] = nd.event_id

        recent = np.full(R, -1, np.int64)
        recent_mask = np.zeros(R, bool)
        seq = [fr for _, fr in self._recent]
        k = len(seq)
        if k:
            recent[R - k:] = seq
            recent_mask[R - k:] = True

        return GraphState(
            node_mask=node_mask, frame_ids=frame_ids, frame_mask=frame_mask,
            rep_frame=rep_frame, start_frame=start_frame, end_frame=end_frame,
            n_frames=n_frames, event_mass=event_mass, grasp_count=grasp_count,
            boundary_strength=b_strength, phase_code=phase_code,
            event_id=event_id, recent_frame_ids=recent, recent_mask=recent_mask,
            n_nodes=V, n_pending=len(self._pending), n_dropped=n_dropped)

    @property
    def nodes(self) -> list[EventNode]:
        """全部 Long-Term 节点（时间升序），含超出 max_nodes 的部分。"""
        return list(self._long_term)

    @property
    def pending(self) -> list[EventNode]:
        return list(self._pending)

    @property
    def phase(self) -> str:
        return self._phase

    def get_trace(self) -> list[StepInfo]:
        return list(self.trace)

    def __repr__(self) -> str:
        return (f"EventGraphController(phase={self._phase}, frames={self._n_frames}, "
                f"long_term={len(self._long_term)}, pending={len(self._pending)})")

    # ══════════════════════════════════════════════════════════════
    # 内部：信号
    # ══════════════════════════════════════════════════════════════
    def _state_signal(self, state: np.ndarray | None) -> tuple[np.ndarray | None, str]:
        """按 state_mean/std 归一化。没给 state 就退回画面信号。"""
        cfg = self.cfg
        if state is None:
            return None, SIGNAL_VISUAL
        if cfg.state_mean is None or cfg.state_std is None:
            raise ValueError(
                "给了 state 但配置里没有 state_mean/state_std，无法归一化。它们应来自 "
                "runs/assets/pi05_realrobot/realrobot/norm_stats.json 的 norm_stats.state，"
                "填在 yaml 的 perceptual_memory 下。")
        s = np.asarray(state, np.float32).reshape(-1)
        mean = np.asarray(cfg.state_mean, np.float32)
        if s.shape != mean.shape:
            raise ValueError(f"state 长度 {s.shape} 与 state_mean {mean.shape} 不符")
        std = np.maximum(np.asarray(cfg.state_std, np.float32), 1e-6)
        return (s - mean) / std, SIGNAL_STATE

    def _note_signal_source(self, src: str) -> None:
        """信号源在**同一个阶段内**中途变化 = 调用方的 bug，不是设计允许的。

        跨阶段的切换是合法的（_switch_phase 会先清空），所以这里只管阶段内。
        """
        if self._signal_source is None:
            self._signal_source = src
        elif src != self._signal_source and not self._src_warned:
            warnings.warn(
                f"同一 episode 内「动静」信号源从 {self._signal_source} 变成了 {src}。"
                f"这通常说明 state 时有时无 —— 会让切分位置不可预期。",
                RuntimeWarning, stacklevel=3)
            self._src_warned = True

    def _raw_signal(self, f: np.ndarray, s_tilde: np.ndarray | None, src: str) -> float:
        """本帧的「动静」原始值。首帧没有上一帧，记 0。"""
        if src == SIGNAL_STATE:
            if self._prev_state is None:
                return 0.0
            return _norm2(s_tilde - self._prev_state)
        if self._prev_f is None:
            return 0.0
        return _norm2(f - self._prev_f)

    # ══════════════════════════════════════════════════════════════
    # 内部：夹爪计数
    # ══════════════════════════════════════════════════════════════
    def _gripper_width(self, state: np.ndarray | None) -> float | None:
        """从原始 state 里取夹爪宽度。取不到返回 None。

        夹爪宽度是 state 的**最后一维** —— 约定来自仿真数据集的组帧方式
        （state = concat(joint_state(7), gripper_state[:1])）。
        用 len(state_mean)-1 而不是硬编码 7：换机器人只要 state 布局遵循同一约定就不用改。
        真机灵巧手不走这条路径（`grasp_mode: hand_closure`），见 `_hand_closure`。
        """
        if state is None or self.cfg.state_mean is None:
            return None
        s = np.asarray(state, np.float32).reshape(-1)
        if s.shape[0] < len(self.cfg.state_mean):
            return None                      # 给了个不含夹爪的 state（诊断脚本常见）
        return float(s[len(self.cfg.state_mean) - 1])

    def _hand_closure(self, state: np.ndarray | None) -> float | None:
        """左手手指关节的闭合度 = state[hand_slice] 的均值，越大越闭合。取不到返回 None。

        用整只手的均值而不是单根手指：手指之间有联动但不同步，单关节可能半开，
        均值在真机数据上区分度稳定（张开 ~0.14–0.18，闭合 ~0.55–0.83）。
        """
        sl = self.cfg.hand_slice
        if state is None or sl is None:
            return None
        s = np.asarray(state, np.float32).reshape(-1)
        lo, hi = sl
        if s.shape[0] < hi:
            return None
        return float(np.mean(s[lo:hi]))

    def _update_hand_grasp(self, state: np.ndarray | None) -> None:
        """真机灵巧手：闭合度滞回计数。

        闭合度 ≥ hand_close_threshold 进入闭合段，≤ hand_open_threshold 离开并结算一次。
        滞回的两道门槛避免在单条阈值附近抖动时把一次抓取算成多次。

        ⚠️ 只用左手：所有任务都是左手抓取，右手不闭合（见 yaml 的 hand_slice）。
        """
        c = self._hand_closure(state)
        if c is None:
            return
        cfg = self.cfg
        if not self._hand_closed:
            if c >= cfg.hand_close_threshold:
                self._hand_closed = True
            return
        if c <= cfg.hand_open_threshold:
            self._hand_closed = False
            self._n_grasps += 1

    def _update_grasp_count(self, state: np.ndarray | None) -> None:
        """本帧的夹爪读数 → 更新累计成功抓取数。

        判据：把「闭合」（宽度 < grasp_open）的连续帧切成一段，段内宽度的**最小值**
        > grasp_gap 才算这一下真的夹住了东西。取值与「为什么不能用瞬时判据」见
        GraphMemoryConfig.grasp_open / grasp_gap 的注释。

        `grasp_mode="hand_closure"` 时改走 `_update_hand_grasp`（真机灵巧手）。

        **因果**：一次抓取要到夹爪**重新张开**的那一帧才确认 —— 闭合段没结束之前，
        无从知道夹住的是方块还是空气。所以计数只会回溯地增加，任何一帧读到的都是过去时。

        ⚠️ 用的是**原始** state，不是 _state_signal 归一化过的那份：归一化会把门槛
           绑到 norm_stats 上，而 grasp_open/grasp_gap 是绝对宽度。
        """
        cfg = self.cfg
        if cfg.grasp_mode == "hand_closure":
            self._update_hand_grasp(state)
            return
        w = self._gripper_width(state)
        if w is None:
            return
        if w < cfg.grasp_open:
            self._grasp_min = w if self._grasp_min is None else min(self._grasp_min, w)
            return
        # 张开 ⇒ 闭合段到此结束，结算这一次。夹空了就不算数。
        if self._grasp_min is not None and self._grasp_min > cfg.grasp_gap:
            self._n_grasps += 1
        self._grasp_min = None

    # ══════════════════════════════════════════════════════════════
    # 内部：切分
    # ══════════════════════════════════════════════════════════════
    def _resolve_block(self, info: StepInfo) -> None:
        """格子攒满：在格子里挑动静最小的一帧下刀，并把封顶的刀补上。"""
        cfg = self.cfg
        block = self._block
        self._block = []
        if not block or self._active is None:
            return

        vals = np.asarray([s for _, s in block], np.float64)
        i = int(np.argmin(vals))                 # argmin 取首个最小值，确定
        arr = block[i][0]

        # 刀口到上一刀的距离 = 它在 Active Event 里的下标。
        # k ≤ 0 说明这一格最安静的那一帧已经被上一刀（或封顶）划走了，本格出不了刀。
        k = arr - self._active[0].arrival
        if k <= 0:
            self._apply_cap(info)
            return
        if k < cfg.min_event_len:
            info.rejected_short = True
            self._apply_cap(info)
            return

        # ⚠️ 顺序不能反：**先补封顶，再下格子刀**。
        # 反过来的话，最后会被 _cut_at 一次性封成一个 Event，补进去的刀就没地方放了。
        self._apply_cap(info, before=arr)

        # ⚠️ 补完封顶**必须重新量一次间距**。
        # 封顶刀把 Active Event 的起点推到了 P'，格子刀的位置 arr 还是老的 —— 如果
        # P' 离 arr 不到 min_event_len，硬切就会切出 1~9 帧的碎片段（实测 100 条里
        # 有 164 段是这样）。yaml 和 validate() 都写着「两刀最小间隔 = 最短 Event
        # 长度」，这里必须兑现。这一格让位给封顶刀，不出刀。
        if arr - self._active[0].arrival < cfg.min_event_len:
            info.rejected_short = True
            self._apply_cap(info)
            return

        # b_i：这一格的平均动静 vs 刀口处的动静。格子越平、刀口越静，这一刀越"实"。
        m = float(vals.sum()) / len(block)
        strength = 0.0 if m <= 0.0 else min(max(1.0 - block[i][1] / m, 0.0), 1.0)
        self._cut_at(arr - self._active[0].arrival, info,
                     strength=strength, reason=CUT_BOUNDARY)

    def _apply_cap(self, info: StepInfo | None, *, before: int | None = None) -> None:
        """封顶补刀：从上一刀起，每满 cut_max_len 帧补一刀，直到剩下的不够长。

        `before` 给定 ⇒ 只补到那一帧之前（后面紧接着就是格子刀）；
        不给 ⇒ 补到当前末尾（格子出不了刀、或者 episode 收尾时用）。

        规则本身和「安静点」无关 —— 它只是兜底：一个坑都没有的平坦区间（实测
        StopCube 有 69% 的帧完全静止）不能无限长下去，否则一个 Event 就吃掉几百帧，
        96 个节点几下就满了。
        """
        cfg = self.cfg
        if self._active is None:
            return
        limit = (self._active[-1].arrival + 1) if before is None else int(before)
        while limit - self._active[0].arrival > cfg.cut_max_len:
            if not self._cut_at(cfg.cut_max_len, info, strength=0.0,
                                reason=CUT_MAX_LEN):
                break

    def _cut_at(self, k: int, info: StepInfo | None, *,
                strength: float, reason: str) -> bool:
        """把当前 Active Event 从第 k 帧劈开：_active[:k] 成一段，_active[k:] 归下一段。

        k 是**位置下标**，`_active[k]` 是新 Event 的第一帧，也就是边界帧。
        """
        act = self._active
        if not act or k <= 0 or k >= len(act):
            return False
        old, new = act[:k], act[k:]
        self._finalize(old, cut_reason=reason, boundary_strength=strength)
        self._active = list(new)
        if info is not None:
            info.confirmed_boundary = True
            info.boundary_frame = new[0].frame
            info.boundary_strength = float(strength)
            info.cut_reason = reason
        return True

    def _switch_phase(self, new_phase: str) -> EventNode | None:
        """§5.1：exec_start_idx 处强制 finalize；图与 Recent 不动。

        强制切分不是「检测到的」边界，但语义上是更强的边界（视频→执行），b_i 取
        1.0，让 Phase 3 的 merge 强烈避免跨阶段合并。

        返回被关掉的那个节点（没有 Active Event 时返回 None），供调用方记进 trace。
        """
        node = self._finalize(self._active, cut_reason=CUT_PHASE_SWITCH,
                              boundary_strength=1.0)
        self._active = None

        # 两个阶段的信号不同源（执行有 state、视频可能没有），差分跨阶段没有意义
        self._prev_f = None
        self._prev_state = None
        self._raw_hist.clear()
        self._block = []
        self._signal_source = None      # 新阶段重新判定，不该触发阶段内切换告警
        self._phase = new_phase
        # 新 Event 在下一帧到来时建立（step 里做）
        return node

    def _finalize(
        self,
        act: list[_Frame] | None,
        *,
        cut_reason: str,
        boundary_strength: float,
    ) -> EventNode | None:
        """关闭给定的 Active Event，算 medoid + L=8 采样帧，挂到 Pending。"""
        cfg = self.cfg
        if not act:
            return None

        fs = [fr.f for fr in act]
        # medoid：离事件视觉中心最近的**真实**帧（§3）。不是打分选出来的。
        mu = l2_normalize(_ordered_mean(fs))
        scores = [_cos(x, mu) for x in fs]
        rep_pos = int(np.argmax(scores))             # argmax 取首个最大值，确定

        sel = sample_event_frames(len(act), rep_pos, cfg.event_len)
        frame_ids = np.asarray([act[p].frame for p in sel], np.int64)
        node = EventNode(
            event_id=self._n_events,
            phase=self._phase,
            cut_reason=cut_reason,
            start_frame=act[0].frame,
            end_frame=act[-1].frame,
            start_arrival=act[0].arrival,
            end_arrival=act[-1].arrival,
            n_frames=len(act),
            rep_frame=act[rep_pos].frame,
            frame_ids=frame_ids,
            frame_mask=np.ones(len(sel), bool),
            boundary_strength=float(boundary_strength),
            state=PENDING,
            event_mass=len(act),
            # 累计读取数：取本段**末帧**的读数 —— 即「这个 Event 结束时已经抓了几次」。
            grasp_count=act[-1].grasp_cum,
            fixed_summary=self._vbar(frame_ids),
        )
        self._n_events += 1
        self._pending.append(node)
        return node

    def _promote(self, arrival: int, ts: int) -> None:
        """终点完全滑出 Recent 窗口 ⇒ Pending → Long-Term。

        「完全滑出」= end_arrival + recent_window <= 当前 arrival，这样 Recent 窗口里
        的 8 帧与任何 Long-Term 节点在时间上不重叠（§7：两者互斥）。

        Pending 是按 end_arrival 升序入队的，所以从队首 while 即可。
        """
        rw = self.cfg.recent_window
        while self._pending and self._pending[0].end_arrival + rw <= arrival:
            nd = self._pending.popleft()
            nd.state = LONG_TERM
            nd.promote_arrival = arrival
            nd.promote_step = ts
            self._long_term.append(nd)
            # §4：图满就合并，合并完还满就再合，直到回到容量以内。
            # 放在 append 之后而不是之前：刚晋升的这个节点也该参与候选。
            while len(self._long_term) > self.cfg.max_nodes:
                if not self._merge_once(arrival):
                    break        # 合不动了（没有 v̄ 或只剩一个节点）→ 交给丢弃路径告警

    # ══════════════════════════════════════════════════════════════
    # Phase 3：相邻节点合并（v1.3 §4）
    # ══════════════════════════════════════════════════════════════
    def _frame_summaries(self, frame_ids: np.ndarray) -> np.ndarray | None:
        """frame_ids (l,) → (l,512) 帧摘要。算法与 FrozenFrameSummary.summarize 一致。

        **必须复刻那个算法而不是另写一个**：合并后要挑 medoid，而 medoid 的定义
        （§3）用的是 store 里的帧摘要。两边只要差 1 ulp，argmax 就可能选到另一帧，
        于是合并出来的 rep_frame 与「离线预处理时算的」对不上。
        """
        if self._pca_tokens is None:
            return None
        tok = np.asarray(self._pca_tokens(frame_ids), np.float32)   # (l,16,D)
        acc = np.zeros((tok.shape[0], tok.shape[-1]), np.float32)
        for p in range(tok.shape[1]):          # 显式顺序累加，不走 BLAS 归约
            acc += tok[:, p, :]
        acc /= np.float32(tok.shape[1])
        return l2_normalize(acc)

    def _vbar(self, frame_ids: np.ndarray) -> np.ndarray | None:
        """v̄_i = Normalize(Mean_(l,p) EventCache(i,l,p,:))（§4.1）。拿不到就 None。"""
        if self._pca_tokens is None:
            return None
        tok = np.asarray(self._pca_tokens(frame_ids), np.float32)   # (l,16,D)
        acc = np.zeros(tok.shape[-1], np.float32)
        for i in range(tok.shape[0]):
            for p in range(tok.shape[1]):
                acc += tok[i, p, :]
        return l2_normalize(acc / np.float32(tok.shape[0] * tok.shape[1]))

    def _merge_once(self, arrival: int) -> bool:
        """把 C_i 最大的那一对相邻节点并掉。返回是否真的并了。

        C_i = cos(v̄_i, v̄_{i+1}) − λ_b·b_i（§4.1），b_i 取 node_i 的
        boundary_strength —— 按 §3 的约定，节点存的是它**右边界**那一刀，而
        node_i 的右边界正是 i 与 i+1 之间的切分。为什么只用相邻对见模块头。
        并列取最左（严格大于才更新），保证确定性。
        """
        lt = self._long_term
        if len(lt) < 2:
            return False
        lambda_b = float(self.cfg.lambda_b)
        best_i, best_c = -1, -np.inf
        for i in range(len(lt) - 1):
            a, b = lt[i], lt[i + 1]
            if a.fixed_summary is None or b.fixed_summary is None:
                return False                  # 没有 token 读出器 → 一律不合并
            # 跨阶段那一刀的 b 是 1.0（§5.1），λ_b=1.0 时 C ≤ cos − 1 < 0，只要
            # 阶段内还有一对非负的，就轮不到跨阶段合并。是「强烈避开」不是硬禁止。
            c = _cos(a.fixed_summary, b.fixed_summary) \
                - lambda_b * float(a.boundary_strength)
            if c > best_c:
                best_i, best_c = i, c
        if best_i < 0:
            return False
        a, b = lt[best_i], lt[best_i + 1]
        merged = self._merge_pair(a, b, arrival)
        self._long_term[best_i:best_i + 2] = [merged]
        self.merge_log.append((arrival, a.event_id, b.event_id, merged.event_id))
        return True

    def _merge_pair(self, a: EventNode, b: EventNode, arrival: int) -> EventNode:
        """合并两个相邻节点：时间范围取并，在新区间上重新采 L=8 **真实**帧（§4.2）。

        合并只改 metadata —— Data Plane 那边的 h_event 不是从旧节点继承的，而是在
        forward 里拿新采的 L=8 真实帧重新编码，所以这里换掉 `frame_ids` 就够了，
        没有「两个旧编码怎么取平均」的问题。

        rep_frame 也不能沿用 a 或 b 的旧代表帧：新节点覆盖的时间范围变了，旧代表帧
        不再是「离新质心最近的那一帧」。§4.2 明确用「重采样的 L=8 帧里按冻结摘要选
        medoid」取代了早先「两个旧 rep 二选一」的中间方案。
        """
        cfg = self.cfg
        start, end = int(a.start_frame), int(b.end_frame)
        span = end - start + 1
        if span <= cfg.event_len:
            frame_ids = np.arange(start, end + 1, dtype=np.int64)
        else:
            # 等距取 L 帧，含首末。round 后去重（span 大时不会撞，但别依赖这一点）。
            frame_ids = np.unique(
                np.linspace(start, end, cfg.event_len).round().astype(np.int64))

        summ = self._frame_summaries(frame_ids)
        if summ is None:                       # 上面拦过，这里只是不变量
            raise RuntimeError("没有 pca_tokens_fn，不该走到合并")
        mu = l2_normalize(_ordered_mean(list(summ)))
        rep_pos = int(np.argmax([_cos(x, mu) for x in summ]))

        # 发号规则与 _finalize 保持一致：领号后必须自增。少了这一句，下一个新建的
        # 节点会再领到同一个号 —— 两个节点共用一个 event_id，steps.npz 里同一帧的
        # vis_ids 就会出现重复，而训练侧把 id 当行号用（readout.pack_eg_tensors），
        # 于是其中一个节点的真实内容被静默替换成另一个的。
        nid = self._n_events
        self._n_events += 1
        return EventNode(
            event_id=nid,
            phase=a.phase,                     # 两个都是 Long-Term，阶段基本一致
            cut_reason=CUT_MERGE,
            start_frame=start,
            end_frame=end,
            start_arrival=a.start_arrival,
            end_arrival=b.end_arrival,
            n_frames=a.n_frames + b.n_frames,
            rep_frame=int(frame_ids[rep_pos]),
            frame_ids=frame_ids,
            frame_mask=np.ones(len(frame_ids), bool),
            # 合并节点的**右**边界是 b 的右边界，所以沿用 b 的那一刀强度。
            # （节点存的是右边界强度：a 的强度是 a 与 b 之间的切分，合并后已变成内部接缝。）
            boundary_strength=float(b.boundary_strength),
            state=LONG_TERM,
            event_mass=a.event_mass + b.event_mass,          # §4.2（证据体积，相加）
            # 累计抓取数**不是证据体积**：新区间一直延伸到 b 的末尾，所以读数就是 b 的。
            # 做成 a+b 会凭空造出抓取次数（把「已经抓了 2 次」读成 4 次）。
            grasp_count=b.grasp_count,
            provenance=a.provenance + b.provenance,          # §4.2
            fixed_summary=self._vbar(frame_ids),             # 新区间上重算
            merged_from=(a.event_id, b.event_id),
            merge_arrival=arrival,
            promote_arrival=a.promote_arrival,
            promote_step=a.promote_step,
        )
