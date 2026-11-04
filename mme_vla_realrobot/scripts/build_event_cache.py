"""构建 Event-Graph 的 episode 级 event cache（v1.3 §10 Phase 1 交付物）。

产出两样东西：

  ① 冻结 PCA 资产   runs/assets/event_graph/pca_{event_dim}.npz
     2048→512 的投影尺子。**只拟合一次，之后只读**（v1.2 §1.2：禁止现场拟合）。

  ② 每条 episode 的 event cache（原地写进 features/episode_{ep}/）：
       event_tok_{ep}.npy   [T, 16, 512] fp16   全部 T 帧的 PCA token
       event_sum_{ep}.npy   [T, 512]     fp32   全部 T 帧的 f_t
       event_meta_{ep}.json                      指纹 / exec_start_idx / 步号

     ⚠️ 存的是**全部 T 帧**，不是每个 Event 8 帧。「哪 8 帧」要等 Phase 2 的边界检测
        跑完才知道，所以缓存只存原料。详见 store.py 顶部说明。

前置：先跑 scripts/scan_step_index.py 生成 meta/exec_start_map.json。

用法：
    # 全量构建（400 条 episode，约 3.5 GB，预计十几分钟）
    python scripts/build_event_cache.py --num-workers 16

    # 先拿 5 条试跑
    python scripts/build_event_cache.py --episodes 0 1 2 3 4 --num-workers 2

    # 只重建 cache，复用已有的 PCA 资产
    python scripts/build_event_cache.py --skip-pca-fit --force
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

# 纯 CPU/numpy，强制 CPU 后端避免 import 时抢 GPU（训练可能已占满显存）
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.models.representation.event_graph.config import GraphMemoryConfig
from mme_vla_suite.models.representation.event_graph.frame_summary import (
    FrozenFrameSummary, sample_training_tokens,
)
from mme_vla_suite.models.representation.event_graph.store import (
    episode_step_ids, list_episodes, write_episode_cache,
)

# ── 工作进程的全局上下文（initializer 设一次，避免每个任务重复 pickle）──
_CTX: dict = {}


def _init_worker(cfg, pca, feature_dir, exec_map, force):
    _CTX.update(cfg=cfg, pca=pca, feature_dir=feature_dir,
                exec_map=exec_map, force=force)


def _do_episode(epis_idx: int):
    t0 = time.time()
    steps = episode_step_ids(_CTX["feature_dir"], epis_idx)
    if not steps:
        return epis_idx, 0, True, 0.0, "无 token_emb_* 文件"
    if epis_idx not in _CTX["exec_map"]:
        return epis_idx, 0, True, 0.0, "exec_start_map 里没有（先跑 scan_step_index.py）"
    n, skipped = write_episode_cache(
        _CTX["feature_dir"], epis_idx, _CTX["cfg"], _CTX["pca"],
        steps, _CTX["exec_map"][epis_idx], force=_CTX["force"])
    return epis_idx, n, skipped, time.time() - t0, ""


def _chunks(seq: list, n: int):
    k, m = divmod(len(seq), n)
    start = 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        yield seq[start:start + size]
        start += size


def main() -> None:
    ap = argparse.ArgumentParser(description="构建 Event-Graph event cache")
    ap.add_argument("--dataset-path", default="data/realrobot_preprocessed")
    ap.add_argument("--history-config", default="perceptual-eventgraph-expert_v2_realrobot.yaml")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--episodes", type=int, nargs="*", default=None,
                    help="只处理这些 episode（默认全部）")
    ap.add_argument("--force", action="store_true", help="已存在的也重建")
    # PCA 拟合
    ap.add_argument("--pca-path", default=None, help="默认 runs/assets/event_graph/pca_{dim}.npz")
    ap.add_argument("--skip-pca-fit", action="store_true",
                    help="只加载已有 PCA 资产；不存在则报错（不现场拟合）")
    ap.add_argument("--fit-episodes", type=int, default=400,
                    help="参与拟合的 episode 数（从全部 episode 里均匀取）")
    ap.add_argument("--fit-steps", type=int, default=20, help="每条参与拟合的 episode 抽几步")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    feature_dir = os.path.join(args.dataset_path, "features")
    cfg = GraphMemoryConfig.from_history_config(get_history_config(args.history_config))
    pca_path = args.pca_path or cfg.resolve_pca_path()

    print(f"配置      : {args.history_config}  (cache 指纹 {cfg.cache_fingerprint()})")
    print(f"event_dim : {cfg.event_dim}   dtype {cfg.event_cache_dtype}")
    print(f"features  : {feature_dir}")
    print(f"PCA 资产  : {pca_path}")
    print()

    # ── ① 冻结 PCA ─────────────────────────────────────────────────────
    if os.path.exists(pca_path) and not args.force:
        fs = FrozenFrameSummary.load(pca_path)
        print(f"复用已有 PCA 资产：asset_version={fs.asset_version}")
        meta = fs.fit_meta(pca_path)
        if meta:
            print(f"  拟合样本 {meta.get('n_fit')}  seed {meta.get('seed')}  "
                  f"解释方差比例 {meta.get('explained_variance_ratio')}")
    else:
        if args.skip_pca_fit:
            sys.exit(f"--skip-pca-fit 但资产不存在：{pca_path}")
        all_eps = list_episodes(feature_dir)
        if not all_eps:
            sys.exit(f"{feature_dir} 下没有 episode_* 目录")
        # 均匀取 episode 而不是取前 N 个：数据是按任务排序的，取前 400 个等于只拟合
        # 前几个任务，PCA 会偏。
        k = min(args.fit_episodes, len(all_eps))
        idx = np.linspace(0, len(all_eps) - 1, k).round().astype(int)
        fit_eps = [all_eps[int(i)] for i in sorted(set(idx))]

        print(f"拟合 PCA：{len(fit_eps)} 条 episode × {args.fit_steps} 步 "
              f"= 约 {len(fit_eps) * args.fit_steps * 16} 个 token")
        t0 = time.time()
        toks = sample_training_tokens(feature_dir, fit_eps, args.fit_steps, seed=args.seed)
        print(f"  采样 {toks.shape} 用时 {time.time() - t0:.0f}s")
        t0 = time.time()
        fs = FrozenFrameSummary.fit(toks, dim_out=cfg.event_dim, seed=args.seed)
        print(f"  拟合用时 {time.time() - t0:.0f}s")
        fs.save(pca_path)
        print(f"  已保存 {pca_path}")
        print(f"  asset_version = {fs.asset_version}")
        print(f"  解释方差比例 = {fs._explained:.4f}   "
              f"（512 维保留了全部 {fs.dim_in} 维方差的这个比例）")
        if fs._explained < 0.95:
            print(f"  ⚠️ 低于 0.95：{cfg.event_dim} 维可能不够，建议让技术人员确认 "
                  f"event_dim 是否要上调。")

    # ── ② exec_start_idx ───────────────────────────────────────────────
    map_path = os.path.join(args.dataset_path, "meta", "exec_start_map.json")
    if not os.path.exists(map_path):
        sys.exit(f"找不到 {map_path}\n先跑：python scripts/scan_step_index.py "
                 f"--dataset-path {args.dataset_path}")
    with open(map_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    exec_map = {int(k): int(v) for k, v in raw["exec_start_idx"].items()}
    print(f"\nexec_start_map：{len(exec_map)} 条 episode")

    n_zero = sum(1 for v in exec_map.values() if v == 0)
    if n_zero == len(exec_map):
        print(f"  ⚠️ 全部 {n_zero} 条 episode 的 exec_start_idx 都是 0")
        print(f"     ⇒ 这个数据集没有执行前的 video 阶段，v1.3 §5.1 的 video 分支"
              f"在此数据上不会触发。")
        print(f"       代码仍需实现该分支（其它 RoboMME 任务可能有 video），"
              f"但本数据集上只走得通执行阶段。")
    elif n_zero:
        print(f"  其中 {n_zero}/{len(exec_map)} 条为 0（无 video 阶段）")

    # ── ③ 逐 episode 构建 ──────────────────────────────────────────────
    todo = args.episodes if args.episodes else list_episodes(feature_dir)
    todo = [e for e in todo if e in exec_map]
    skipped_eps = [e for e in (args.episodes or list_episodes(feature_dir))
                   if e not in exec_map]
    if skipped_eps:
        print(f"\n⚠️ 跳过 {len(skipped_eps)} 条没有 exec_start_idx 的 episode："
              f"{skipped_eps[:10]}{' ...' if len(skipped_eps) > 10 else ''}")
    if not todo:
        sys.exit("没有可处理的 episode")

    print(f"\n开始构建 {len(todo)} 条 episode，{args.num_workers} 进程"
          f"{'，--force 全量重建' if args.force else '，断点续跑'}")
    t0 = time.time()
    n_done = n_steps = n_skip = 0
    errs: list[tuple[int, str]] = []
    with ProcessPoolExecutor(max_workers=args.num_workers, initializer=_init_worker,
                             initargs=(cfg, fs, feature_dir, exec_map, args.force)) as ex:
        for ep, n, skipped, dt, msg in ex.map(_do_episode, todo, chunksize=1):
            if msg:
                errs.append((ep, msg))
                continue
            n_done += 1
            if skipped:
                n_skip += 1
            else:
                n_steps += n
            if n_done % 25 == 0 or n_done == len(todo):
                el = time.time() - t0
                rate = n_done / max(el, 1e-9)
                print(f"  {n_done}/{len(todo)}  写入 {n_steps} 步  跳过 {n_skip} 条  "
                      f"{el:.0f}s  预计剩余 {(len(todo) - n_done) / max(rate, 1e-9):.0f}s",
                      flush=True)

    el = time.time() - t0
    print(f"\n完成：{n_done} 条（写入 {n_steps} 步，跳过 {n_skip} 条已存在），用时 {el:.0f}s")
    if errs:
        print(f"⚠️ {len(errs)} 条出错：")
        for ep, msg in errs[:20]:
            print(f"    episode_{ep}: {msg}")

    # ── ④ 体积核算 ─────────────────────────────────────────────────────
    tok_bytes = n_steps * cfg.token_per_image * cfg.event_dim * np.dtype(cfg.event_cache_dtype).itemsize
    sum_bytes = n_steps * cfg.event_dim * 4
    print(f"\n本次写入体积：event_tok {tok_bytes / 1e9:.2f} GB + "
          f"event_sum {sum_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
