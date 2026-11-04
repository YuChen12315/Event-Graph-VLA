"""把「这一帧看见了哪些 Event」拼成模型要的那 15 个张量。

离线训练（dataset._load_eventgraph_mem，读 nodes.npz / steps.npz）和在线推理
（EventGraphMemoryBuffer，读内存里的活节点表）走的是**这一个函数**。v1.3 §11
禁止「预处理 / 训练 / online server 各维护一份实现」—— 拼装本身不含 boundary
公式，但一样是「两边悄悄走偏、不报错、只是慢慢学坏」的高危区，所以一并收在这里。
来回切两套实现的代价不是崩溃，是评测数字无法解释。

调用方提供四样东西 —— 这才是离线与在线真正不同的地方：

    node_rows        节点表：每个 Event 一行的 metadata
    vis_ids          这一帧能看见哪几行，**时间升序**
    recent_ids/mask  最近 recent_window 帧的帧号（右对齐，见下）
    read_*           帧号 → 那帧的原始 payload / 冻结 PCA token

⚠️ 槽位对齐有两套，别混（都是 §5.2「最新在最高有效槽位」的同一个约定）：
    · 长期段在 max_nodes 里左补零 → 可见的 m 个落在 [N-m, N)，
      m = 可见事件数 k + 空槽填充的真实帧数（v1.4；k=0 时就是纯 FrameSample）
    · 最近帧在 recent_window 里同样左补零 → 可见的 k 个落在 [R-k, R)
    所以**两边的 payload 都必须往右放**。早期版本把 recent 的 payload 从 0 开始
    往左放、mask 却留在右边，于是前 7 帧（凑不满 8 帧窗口时）真实 token 被 mask
    挡掉、补零槽位反被标成有效 —— 症状是前几步的记忆分支拿到一堆零。
"""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from .config import GraphMemoryConfig

__all__ = ["pack_eg_tensors", "EG_TENSOR_SHAPES"]

# 输出张量的 (形状, dtype) —— 不含 batch 维。模型那边统一 [np.newaxis] 加 b。
# 这里列全是为了让「少接了一根线」在拼装完就能炸，而不是等 shape 广播报错。
#
# dtype 写 None 表示「只要是浮点就行」：三个 state 张量要过调用方的 _normalize_state，
# 而 norm_stats 是 float64，所以原路径出来就是 float64（模型侧 jnp.asarray 仍落到
# float32）。这里**不去改成 float32** —— 那会让在线/离线对不上的同时，还和已经训起来
# 的那条路径不是同一份数据。
EG_TENSOR_SHAPES: dict[str, tuple[tuple[str, ...], type | None]] = {
    "eg_node_tokens":          (("N", "L", "P", "E"), np.float16),
    "eg_node_frame_mask":      (("N", "L"), np.bool_),
    "eg_node_state":           (("N", "L", "S"), None),
    "eg_node_mask":            (("N",), np.bool_),
    "eg_node_mass":            (("N",), np.float32),
    "eg_node_bstr":            (("N",), np.float32),
    "eg_node_phase":           (("N",), np.float32),
    "eg_node_grasp":           (("N",), np.float32),
    "eg_node_payload":         (("N", "P", "D1"), np.float32),
    "eg_node_payload_pos":     (("N", "P", "D2"), np.float32),
    "eg_node_payload_state":   (("N", "S"), None),
    "eg_recent_payload":       (("R", "P", "D1"), np.float32),
    "eg_recent_payload_pos":   (("R", "P", "D2"), np.float32),
    "eg_recent_payload_state": (("R", "S"), None),
    "eg_recent_mask":          (("R",), np.bool_),
}


def pack_eg_tensors(
    *,
    cfg: GraphMemoryConfig,
    node_rows: Mapping[str, np.ndarray],
    vis_ids: np.ndarray,
    recent_ids: np.ndarray,
    recent_mask: np.ndarray,
    frame_states: np.ndarray,
    dims: tuple[int, int, int],
    read_payload: Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    read_pca_tokens: Callable[[np.ndarray], np.ndarray],
    normalize_state: Callable[[np.ndarray], np.ndarray],
    exec_start_idx: int,
    read_payload_many: (
        Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]] | None
    ) = None,
) -> dict[str, np.ndarray]:
    """拼出 15 个 eg_* 张量（不含 batch 维）。

    node_rows    : 每个 Event 一行，至少要有 frame_ids (·,L) / frame_mask (·,L) /
                   rep_frame (·) / event_mass (·) / boundary_strength (·) /
                   phase_code (·) / grasp_count (·)；**强烈建议**再给
                   start_frame (·) / end_frame (·)，填充帧靠它找父事件
    vis_ids      : 能看见的行号，升序；长度 k ≤ max_nodes
    recent_*     : (R,) 帧号与有效位，右对齐（最新的在最高槽位）
    frame_states : (T, d3) 逐帧原始关节角，**按帧号索引**
    dims         : (d1, d2, d3) —— 视觉 token / 位置编码 / 关节角三个宽度
    read_payload     : 帧号 → (img [P,d1], pos [P,d2], state [d3])，原始精度
    read_payload_many: 帧号数组 (n,) → (img [n,P,d1], pos [n,P,d2], state [n,d3])。
                       给了就**整个样本一次取完**（训练侧走每-episode 紧凑缓存）；
                       不给则逐帧调 read_payload（在线路径 / 旧行为）。
    read_pca_tokens  : 帧号数组 (n,) → (n, P, event_dim) 冻结 PCA token
    normalize_state  : 关节角归一化。**模型输入用的那份**，与 Control Plane
                       内部按 cfg.state_mean/std 算「动静」的那份不是同一件事。
    exec_start_idx   : 执行阶段起始帧号（`<0` 表示未知）。只用于给**无父事件**的
                       填充帧判 phase；有父事件的填充帧直接继承该事件。

    k=0（图还空着）是合法状态：长期段不再整段屏蔽，而是退化成纯 FrameSample
    （32 个真实帧）。

    ── 空槽填充（v1.4）────────────────────────────────────────────────
    长期段容量是 max_nodes 个槽，事件只占 k 个。填充目标是 fill_target 个槽
    （事件 + 填充帧；cfg.fill_target=None ⇒ 填到 max_nodes，旧行为；0 ⇒ 不填）：
    剩下的 fill_target−k 个槽用**真实帧**补。
    候选是最近 recent_window 帧之外的全部历史帧，按「离已占帧（事件代表帧 ∪
    最近帧 ∪ 已选填充帧）时间上最远」贪心挑选，平局取帧号最小的。1 维最远点的
    极限就是等间隔采样 —— 所以这既是「均匀采样」，也是 k 很小时退化成 FrameSample
    的保底。最近帧永远不参与（硬过滤），历史不够时能填多少填多少。
    """
    N, L, R, P = cfg.max_nodes, cfg.event_len, cfg.recent_window, cfg.token_per_image
    d1, d2, d3 = dims
    E = cfg.event_dim
    vis_ids = np.asarray(vis_ids, dtype=np.int64)
    k = int(vis_ids.size)
    if k > N:
        raise ValueError(
            f"可见节点 {k} 个 > max_nodes={N}。Control Plane 的合并应当保证不可能发生，"
            f"先查调用方的节点表是不是和配置同源。")

    out: dict[str, np.ndarray] = {
        "eg_node_tokens": np.zeros((N, L, P, E), np.float16),
        "eg_node_frame_mask": np.zeros((N, L), bool),
        "eg_node_state": np.zeros((N, L, d3), np.float32),
        "eg_node_mask": np.zeros(N, bool),
        "eg_node_mass": np.zeros(N, np.float32),
        "eg_node_bstr": np.zeros(N, np.float32),
        "eg_node_phase": np.zeros(N, np.float32),
        "eg_node_grasp": np.zeros(N, np.float32),
        "eg_node_payload": np.zeros((N, P, d1), np.float32),
        "eg_node_payload_pos": np.zeros((N, P, d2), np.float32),
        "eg_node_payload_state": np.zeros((N, d3), np.float32),
        "eg_recent_payload": np.zeros((R, P, d1), np.float32),
        "eg_recent_payload_pos": np.zeros((R, P, d2), np.float32),
        "eg_recent_payload_state": np.zeros((R, d3), np.float32),
        "eg_recent_mask": np.zeros(R, bool),
    }

    rm = np.asarray(recent_mask, bool)
    recent_valid = np.asarray(recent_ids, np.int64)[rm]

    # ── ① 可见事件 → 记录（按时间排序；跨度互不重叠）──
    has_span = ("start_frame" in node_rows) and ("end_frame" in node_rows)
    events: list[dict] = []
    for r in vis_ids:
        rep = int(node_rows["rep_frame"][r])
        fid = np.asarray(node_rows["frame_ids"][r], np.int64)
        fm = np.asarray(node_rows["frame_mask"][r], bool)
        events.append({
            "rep": rep,
            # 老缓存没有 start/end 时退化成「跨度 = 代表帧」，填充帧就全走 orphan 路径。
            "start": int(node_rows["start_frame"][r]) if has_span else rep,
            "end": int(node_rows["end_frame"][r]) if has_span else rep,
            # frame_ids 里没效的位置是 -1，直接整条传下去会被当成负索引（静默读到
            # 最后一帧）—— 必须先按 mask 挑出来。
            "frames": fid[fm],
            "mass": float(node_rows["event_mass"][r]),
            "bstr": float(node_rows["boundary_strength"][r]),
            "phase": float(node_rows["phase_code"][r]),
            "grasp": float(node_rows["grasp_count"][r]),
        })
    events.sort(key=lambda e: (e["start"], e["end"], e["rep"]))

    # ── ② 空槽填充：全历史时间最远点 ──
    occupied = {e["rep"] for e in events} | {int(x) for x in recent_valid}
    # 最近窗口里最早的那帧之前，才是「长期历史」。窗口还没满时 recent 从 0 开始，
    # horizon=0，自然填不了 —— 这与旧行为一致。
    horizon = int(recent_valid.min()) if recent_valid.size else 0
    horizon = min(horizon, int(frame_states.shape[0]))
    cand = np.setdiff1d(np.arange(horizon, dtype=np.int64),
                        np.asarray(sorted(occupied), np.int64))
    fill_target = N if cfg.fill_target is None else int(cfg.fill_target)
    n_fill = max(0, fill_target - k)
    picked: list[int] = []
    if n_fill > 0 and cand.size:
        occ = np.asarray(sorted(occupied), np.int64)
        dmin = (np.min(np.abs(cand[:, None] - occ[None, :]), axis=1)
                if occ.size else np.full(cand.shape, np.inf, np.float64))
        for _ in range(n_fill):
            i = int(np.argmax(dmin))         # 并列时 argmax 取首个 ⇒ 最小帧号，确定
            f = int(cand[i])
            picked.append(f)
            cand = np.delete(cand, i)
            dmin = np.delete(dmin, i)
            if not cand.size:
                break
            dmin = np.minimum(dmin, np.abs(cand - f))

    # ── ③ 事件 + 填充帧 → 统一记录，按时间排序 ──
    records: list[dict] = [{
        "group": e["start"], "order": e["rep"], "kind": "event",
        "frames": e["frames"], "payload": e["rep"],
        "mass": e["mass"], "bstr": e["bstr"],
        "phase": e["phase"], "grasp": e["grasp"],
    } for e in events]

    def _parent(f: int) -> dict | None:
        # 事件跨度互不重叠 ⇒ 归属唯一；k ≤ max_nodes，线性扫足够。
        for e in events:
            if e["start"] <= f <= e["end"]:
                return e
        return None

    for f in picked:
        p = _parent(f)
        if p is not None:
            # 事件内的帧：摆在父事件组里，标量跟父事件一致。
            group, bstr, phase, grasp = p["start"], p["bstr"], p["phase"], p["grasp"]
        else:
            # 不属于任何可见事件（Pending 段 / k=0 等）：phase 按 exec_start 判，
            # bstr 没有边界可继承取 0，累计抓取数取此前最近一个事件的末帧读数。
            group, bstr = f, 0.0
            phase = 0.0 if exec_start_idx < 0 else float(f >= exec_start_idx)
            prev = [e for e in events if e["end"] < f]
            grasp = prev[-1]["grasp"] if prev else 0.0
        records.append({
            "group": group, "order": f, "kind": "frame",
            "frames": np.asarray([f], np.int64), "payload": f,
            "mass": 1.0, "bstr": bstr, "phase": phase, "grasp": grasp,
        })
    records.sort(key=lambda x: (x["group"], x["order"]))

    total = len(records)
    if total > N:                            # 构造上不可能；炸出来比静默截断好
        raise AssertionError(f"长期段记录 {total} 条 > max_nodes={N}")

    # ── ④ payload：整个样本一次取完 ──
    # 旧写法在写槽位的循环里逐帧调 read_payload —— 32 个节点 + 8 个近期帧就是 40 次
    # open + pickle。训练侧现在可以一次 fancy-index 取回整集数组，所以先把所有帧号
    # 汇总，再统一取。顺序：records 的顺序 + recent_valid 的顺序（下面按同一顺序回填）。
    payload_ids = np.asarray(
        [rec["payload"] for rec in records] + [int(f) for f in recent_valid],
        dtype=np.int64)
    imgs = poss = states = None
    if payload_ids.size:
        if read_payload_many is not None:
            imgs, poss, states = read_payload_many(payload_ids)
        else:
            views = [read_payload(int(f)) for f in payload_ids]
            imgs = np.stack([v[0] for v in views])
            poss = np.stack([v[1] for v in views])
            states = np.stack([v[2] for v in views])
        imgs = np.asarray(imgs, np.float32)
        poss = np.asarray(poss, np.float32)
        states = np.asarray(states, np.float32)
        if imgs.shape[0] != payload_ids.size:
            raise ValueError(
                f"read_payload_many 返回 {imgs.shape[0]} 帧，期望 {payload_ids.size} 帧")

    if total:
        for j, rec in enumerate(records):
            slot = N - total + j             # 左 padding：最新的落在最高槽位（§5.2）
            out["eg_node_mask"][slot] = True
            out["eg_node_mass"][slot] = rec["mass"]
            out["eg_node_bstr"][slot] = rec["bstr"]
            out["eg_node_phase"][slot] = rec["phase"]
            out["eg_node_grasp"][slot] = rec["grasp"]
            fid = rec["frames"]
            n_valid = int(fid.size)
            if n_valid == 0:
                continue
            # 掩码必须和 fid 一样压紧：跨度不足 8 帧的短事件（以及 1 帧的填充节点）
            # 只能占前 n_valid 个槽，否则会广播失败。
            out["eg_node_frame_mask"][slot, :n_valid] = True
            out["eg_node_tokens"][slot, :n_valid] = read_pca_tokens(fid)
            out["eg_node_state"][slot, :n_valid] = frame_states[fid]
            out["eg_node_payload"][slot] = imgs[j]
            out["eg_node_payload_pos"][slot] = poss[j]
            out["eg_node_payload_state"][slot] = states[j]

    if rm.any():
        out["eg_recent_mask"] = rm
        # 往 mask 说"有效"的那些槽位里放，而不是从 0 开始 —— 见模块 docstring 的警告。
        for i, slot in enumerate(np.flatnonzero(rm)):
            j = total + i                    # payload_ids 里近期帧排在所有 record 之后
            out["eg_recent_payload"][slot] = imgs[j]
            out["eg_recent_payload_pos"][slot] = poss[j]
            out["eg_recent_payload_state"][slot] = states[j]

    # 关节角统一走同一条归一化（与 static_state_emb 等分支一致）
    for key in ("eg_node_state", "eg_node_payload_state", "eg_recent_payload_state"):
        out[key] = normalize_state(out[key])

    _check(out, {"N": N, "L": L, "R": R, "P": P, "E": E, "D1": d1, "D2": d2, "S": d3})
    return out


def _check(out: dict[str, np.ndarray], sym: dict[str, int]) -> None:
    """拼装完立刻对名单和形状，不要留到模型里广播报错。

    ⚠️ 名单以 history_observation.EG_FIELDS 为准（模型真正会读的那 15 个），
       这里只是把同一份名单再翻成形状。两张表分开维护的话，加的字段会静默地
       只出现在一边 —— 那正是「图是空的、但不报错」的来源。
    """
    from mme_vla_suite.models.integration.history_observation import EG_FIELDS

    if set(out) != set(EG_FIELDS):
        raise AssertionError(
            f"readout 产出的字段与 EG_FIELDS 对不上：多 {sorted(set(out) - set(EG_FIELDS))}，"
            f"缺 {sorted(set(EG_FIELDS) - set(out))}")
    for name, (spec, dt) in EG_TENSOR_SHAPES.items():
        got = out[name]
        want = tuple(sym[a] for a in spec)
        if got.shape != want:
            raise AssertionError(f"{name} 形状 {got.shape}，期望 {want}")
        if dt is None:
            if not np.issubdtype(got.dtype, np.floating):
                raise AssertionError(f"{name} dtype {got.dtype} 不是浮点")
        elif got.dtype != dt:
            raise AssertionError(f"{name} dtype {got.dtype}，期望 {np.dtype(dt)}")
