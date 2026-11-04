"""离线落盘 EventGraph 的逐帧因果状态，供训练时 O(1) 读取。

为什么必须离线算
---------------------------------------------------------------------------
Controller 是**逐帧推进**的状态机：第 t 帧看到哪些 Long-Term 节点，取决于第 0..t
帧。如果放到 dataloader 里现算，每个样本都要从头重放 t 帧 —— 一条 episode 就是
O(T²)，和 graph_streaming / graphmem 踩的是同一个坑。所以整条 episode 跑一遍，
每帧存一份「当时看见了什么」。

为什么每帧只需要存一个「节点 id 列表」
---------------------------------------------------------------------------
一个节点进 Long-Term 之后，它的 metadata 就再也不会变了 —— 合并不会原地改老节点，
而是造一个**新 id**、把两个老的挪走。所以「帧 t 看见哪些节点」这句话里，节点的
内容可以事后去节点表里查，每帧只要记 id 就够了。

于是每条 episode 只有两个数组：
    nodes.npz   每个曾出现在图里的节点一行（帧号、代表帧、证据量、属于哪个阶段…）
    steps.npz   每帧一行：看见的节点 id + 最近 8 帧 + 阶段

**这些 id 是"存活过"的节点，含之后被合并掉的那些。** 合并掉的节点在它被并掉之前
是可见的，所以必须留着 —— 只存最终节点表会让「被合并之前的那几百帧」全部查不到
节点，而且不会报错，只会静默地少一段记忆。

省的是什么（§11 明令禁止的那条）
---------------------------------------------------------------------------
这里**不**复制任何 2048-d 的视觉 token。节点载荷（代表帧的 16×2048）和最近 8 帧
的真实 token，训练时直接从原始 token_emb_{s}.npy 读 —— 那是磁盘上本来就有的那份。
本脚本只产出「选哪些帧」这张表（几十 KB），既不复制大数据，也不会出现「缓存里的
token 和原始 token 不是同一份」这种对不上的情况。

用法：
    python scripts/precompute_eventgraph_memory.py                    # 全量
    python scripts/precompute_eventgraph_memory.py --episodes 0 1 2    # 指定几条
    python scripts/precompute_eventgraph_memory.py --force            # 重算
    python scripts/precompute_eventgraph_memory.py --workers 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.models.representation.event_graph.config import GraphMemoryConfig
from mme_vla_suite.models.representation.event_graph.frame_summary import FrozenFrameSummary
from mme_vla_suite.models.representation.event_graph.controller import (
    PHASE_EXEC, PHASE_VIDEO, EventGraphController,
)
from mme_vla_suite.models.representation.event_graph.store import (
    EventFeatureStore, episode_step_ids, list_episodes,
)

# 节点表每行的字段（缺的会被 -1/0 补上）。顺序即 npz 里的 key 顺序，改动要同步
# 升 VERSION，否则旧文件会被当新文件读。
NODE_INT_FIELDS = ("event_id", "start_frame", "end_frame", "rep_frame", "n_frames",
                   "event_mass", "phase_code", "born_arrival", "grasp_count")
NODE_FLOAT_FIELDS = ("boundary_strength",)
# 2：节点表新增 grasp_count（夹爪累计抓取数）。旧 npz 缺这一列，读出来会是 KeyError
#    —— 升版本让 meta 校验把它们判成"过期"，逼一次重算而不是训到一半炸。
VERSION = 2

_CTX: dict = {}


def _init_worker(cfg, pca, feature_dir):
    _CTX.update(cfg=cfg, pca=pca, feature_dir=feature_dir)


def _paths(feature_dir: str, epis_idx: int) -> tuple[str, str, str]:
    d = os.path.join(feature_dir, f"episode_{epis_idx}", "eventgraph")
    return (os.path.join(d, "nodes.npz"), os.path.join(d, "steps.npz"),
            os.path.join(d, "meta.json"))


def _atomic_npz(path: str, **arrays) -> None:
    """原子写。断点续跑脚本最怕留半个文件被当成"已完成"跳过。"""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    os.replace(tmp, path)


def _atomic_write(path: str, text: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def build_episode(epis_idx: int, *, force: bool = False) -> tuple[str, bool]:
    """跑一条 episode 并落盘。返回 (消息, 是否跳过)。"""
    cfg: GraphMemoryConfig = _CTX["cfg"]
    feature_dir: str = _CTX["feature_dir"]
    pca: FrozenFrameSummary = _CTX["pca"]
    nodes_path, steps_path, meta_path = _paths(feature_dir, epis_idx)
    st = EventFeatureStore(feature_dir, epis_idx, cfg, pca=pca)

    if os.path.exists(meta_path) and not force:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                old = json.load(f)
            if (old.get("version") == VERSION
                    and old.get("graph_fingerprint") == _graph_fingerprint(cfg)
                    and old.get("pca_asset_version") == pca.asset_version
                    and old.get("n_steps") == st.n_steps):
                return f"episode_{epis_idx}: 已存在，跳过", True
        except (json.JSONDecodeError, OSError):
            pass        # 残缺的 meta → 重算

    T = st.n_steps
    steps = episode_step_ids(feature_dir, epis_idx)
    if T != len(steps):
        raise ValueError(f"episode_{epis_idx}: store 报 {T} 帧，step_ids 有 {len(steps)} 个")

    # 只读两样东西进内存：冻结帧摘要（切分用）和关节角（动静信号用）。
    # PCA token **不预读** —— 合并要用时再按帧号查 memmap，几百帧用不满全部。
    summ = st.all_summaries()
    # state 维度从数据里取，不写死：RoboMME 是 8 维，真机数据是 42 维。
    # 用 state_mean 的长度也可以，但它在部分配置里是 None；直接读第一帧最稳。
    states: np.ndarray | None = None
    d = os.path.join(feature_dir, f"episode_{epis_idx}")
    for i, s in enumerate(steps):
        with open(os.path.join(d, f"token_emb_{s}.npy"), "rb") as fh:
            state_emb = np.asarray(np.load(fh, allow_pickle=True).item()["state_emb"],
                                   np.float32)
        if states is None:
            states = np.empty((T, state_emb.size), np.float32)
        elif state_emb.size != states.shape[1]:
            raise ValueError(
                f"episode_{epis_idx}: state_emb 维度不一致 "
                f"（第 {i} 帧 {state_emb.size}，期望 {states.shape[1]}）")
        states[i] = state_emb

    exec_start = int(st.exec_start_idx)
    ctl = EventGraphController(cfg, record_trace=False, pca_tokens_fn=st.pca_tokens)
    # §5.1：exec_start_idx 之前是 conditioned video，也必须进同一张图（不许排除）
    ctl.reset(phase=PHASE_EXEC if exec_start <= 0 else PHASE_VIDEO)

    # 节点表用 dict 累积（id → 一行），因为合并会造新 id，事先不知道总数
    rows: dict[int, dict] = {}
    vis_ptr: list[int] = [0]
    vis_ids: list[int] = []
    recent = np.full((T, cfg.recent_window), -1, np.int64)
    recent_mask = np.zeros((T, cfg.recent_window), bool)
    phase_code = np.zeros(T, np.int8)

    for t in range(T):
        ph = PHASE_EXEC if t >= exec_start else PHASE_VIDEO
        ctl.step(summ[t], state=states[t], timestamp=int(steps[t]), phase=ph)

        for nd in ctl.nodes:                      # 当前可见（时间升序）
            if nd.event_id not in rows:
                rows[nd.event_id] = _node_row(nd, t)
            vis_ids.append(nd.event_id)
        vis_ptr.append(len(vis_ids))

        seq = [fr for _, fr in ctl._recent]
        k = len(seq)
        if k:
            recent[t, cfg.recent_window - k:] = seq
            recent_mask[t, cfg.recent_window - k:] = True
        phase_code[t] = 0 if ph == PHASE_VIDEO else 1

    ctl.flush()      # 收尾段这时才关闭；它仍是 Pending，本 episode 内不晋升 → 不进图

    # ── 落盘 ──
    ids = sorted(rows)
    if not ids:
        raise RuntimeError(f"episode_{epis_idx}: 一个 Long-Term 节点都没有，"
                           f"检查 exec_start_idx={exec_start} 与 cut 参数")
    remap = {eid: i for i, eid in enumerate(ids)}
    nn_ = len(ids)
    node_arrays = {
        f: np.asarray([rows[e][f] for e in ids], dtype) for f, dtype in (
            [(f, np.int64) for f in NODE_INT_FIELDS]
            + [(f, np.float32) for f in NODE_FLOAT_FIELDS])}
    # 补齐到固定 L 再 stack：Event 短于 L 时 sample_event_frames 会返回更少的帧
    # （实测不会发生 —— min_event_len=10 > L=8 保证了中段够长，收尾段又始终是
    #  Pending 进不了图 —— 但这不是代码强制的不变量，直接 np.stack 会撞成长度不等
    #  的报错，且在别处改参数后才炸。补 -1 + frame_mask=False，模型按 mask 排除。）
    L = cfg.event_len
    fid = np.full((nn_, L), -1, np.int64)
    fmask = np.zeros((nn_, L), bool)
    for i, e in enumerate(ids):
        k = len(rows[e]["frame_ids"])
        if k > L:
            raise RuntimeError(f"episode_{epis_idx} 节点 {e} 有 {k} 帧，超过 event_len={L}")
        fid[i, :k] = rows[e]["frame_ids"]
        fmask[i, :k] = rows[e]["frame_mask"]
    node_arrays["frame_ids"] = fid
    node_arrays["frame_mask"] = fmask
    # 合并来源：-1 = 原始切分出来的节点（不是合并产物）
    node_arrays["parent_a"] = np.asarray(
        [remap.get(rows[e]["merged_from"][0], -1) if rows[e]["merged_from"] else -1
         for e in ids], np.int64)
    node_arrays["parent_b"] = np.asarray(
        [remap.get(rows[e]["merged_from"][1], -1) if rows[e]["merged_from"] else -1
         for e in ids], np.int64)

    vis = np.asarray([remap[e] for e in vis_ids], np.int32)     # 重映射成行号
    os.makedirs(os.path.dirname(nodes_path), exist_ok=True)
    _atomic_npz(nodes_path, **node_arrays)
    _atomic_npz(steps_path,
                vis_ptr=np.asarray(vis_ptr, np.int32), vis_ids=vis,
                recent_ids=recent, recent_mask=recent_mask,
                phase_code=phase_code,
                step_ids=np.asarray(steps, np.int64),
                states=states)
    # 「被并掉」= 出现在别人的 merged_from 里。反过来 n_nodes_final 就是活下来的那些。
    merged_away = {p for e in ids for p in (rows[e]["merged_from"] or ())}
    _atomic_write(meta_path, json.dumps({
        "version": VERSION,
        "epis_idx": epis_idx,
        "n_steps": T,
        "exec_start_idx": exec_start,
        "n_nodes_ever": nn_,
        "n_nodes_final": nn_ - len(merged_away),
        "n_nodes_merged_away": len(merged_away),
        "n_merges": sum(1 for e in ids if rows[e]["merged_from"] is not None),
        "max_visible": int(np.diff(np.asarray(vis_ptr)).max()),
        "graph_fingerprint": _graph_fingerprint(cfg),
        "pca_asset_version": pca.asset_version,
    }, indent=2, ensure_ascii=False) + "\n")
    return (f"episode_{epis_idx}: {T} 帧 → {nn_} 个节点"
            f"（其中 {sum(1 for e in ids if rows[e]['merged_from'] is not None)} 个是合并产物），"
            f"单帧最多看见 {int(np.diff(np.asarray(vis_ptr)).max())} 个"), False


def _node_row(nd, t: int) -> dict:
    """从 EventNode 抠出要落盘的一行。

    **只存不可变量**：节点一旦可见就不再变化（合并造新 id 而不是原地改），所以这里
    记的值在它的整个生命周期里都成立，逐帧快照只需要记 id。
    """
    return {
        "event_id": int(nd.event_id),
        "start_frame": int(nd.start_frame),
        "end_frame": int(nd.end_frame),
        "rep_frame": int(nd.rep_frame),
        "n_frames": int(nd.n_frames),
        "event_mass": int(nd.event_mass),
        "grasp_count": int(nd.grasp_count),
        "phase_code": 0 if nd.phase == PHASE_VIDEO else 1,
        "born_arrival": int(t),          # 第一次被看见的到达序号
        "boundary_strength": float(nd.boundary_strength),
        "frame_ids": np.asarray(nd.frame_ids, np.int64),
        "frame_mask": np.asarray(nd.frame_mask, bool),
        "merged_from": nd.merged_from,
    }


def _graph_fingerprint(cfg: GraphMemoryConfig) -> str:
    """影响图结构的那些旋钮。变了就得重算。"""
    import hashlib
    keys = ("max_nodes", "event_len", "recent_window", "cut_block", "cut_max_len",
            "cut_smooth", "min_event_len", "lambda_b",
            # 夹爪门槛也进指纹：它们不移动切分，但决定节点表里 grasp_count 那一列的
            # 数值。改了不重算就会新旧两套读数混在同一个 cache 里。
            "grasp_mode", "grasp_open", "grasp_gap",
            "hand_slice", "hand_close_threshold", "hand_open_threshold")
    s = "|".join(f"{k}={getattr(cfg, k)}" for k in keys)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description="EventGraph 逐帧图状态离线落盘")
    ap.add_argument("--dataset-path", default="data/realrobot_preprocessed")
    ap.add_argument("--history-config", default="perceptual-eventgraph-expert_v2_realrobot.yaml")
    ap.add_argument("--episodes", type=int, nargs="*", default=None,
                    help="只算这几条；不给就是全部")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="已存在的也重算")
    args = ap.parse_args()

    feature_dir = os.path.join(args.dataset_path, "features")
    cfg = GraphMemoryConfig.from_history_config(get_history_config(args.history_config))
    pca = FrozenFrameSummary.load(cfg.resolve_pca_path())

    all_eps = list_episodes(feature_dir)
    eps = all_eps if args.episodes is None else args.episodes
    missing = [e for e in eps if e not in set(all_eps)]
    if missing:
        sys.exit(f"features 下没有这些 episode：{missing[:5]}")
    if cfg.state_mean is None or cfg.state_std is None:
        sys.exit("缺 state_mean/state_std：切分信号要按 norm_stats 归一化，"
                 "从 runs/assets/pi05_realrobot/realrobot/norm_stats.json 抄进 yaml")

    print(f"配置 {args.history_config}   容量 max_nodes={cfg.max_nodes}   "
          f"λ_b={cfg.lambda_b}   PCA 资产 {pca.asset_version[:12]}…")
    print(f"共 {len(eps)} 条 episode，workers={args.workers}"
          f"{'，强制重算' if args.force else ''}\n")

    n_done = n_skip = 0
    errs = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(cfg, pca, feature_dir)) as ex:
        futures = [(e, ex.submit(build_episode, e, force=args.force)) for e in eps]
        for i, (e, fu) in enumerate(futures, 1):
            try:
                msg, skipped = fu.result()
                n_skip += skipped
                n_done += not skipped
                if i % 25 == 0 or i == len(futures):
                    print(f"[{i}/{len(futures)}] {msg}")
            except Exception as ex_:
                errs.append((e, f"{type(ex_).__name__}: {ex_}"))
    print(f"\n完成 {n_done} 条，跳过 {n_skip} 条，失败 {len(errs)} 条")
    for e, m in errs[:10]:
        print(f"   episode_{e}: {m}")
    if errs:
        sys.exit(1)
    print(f"落盘在 {os.path.join(feature_dir, 'episode_<n>', 'eventgraph')}/")


if __name__ == "__main__":
    main()
