"""扫描 pkl，取出每条 episode 的 exec_start_idx，落盘成 meta/exec_start_map.json。

**为什么需要这个脚本**：v1.3 §5.1 要求区分两个阶段 ——

    t <  exec_start_idx ：执行前的 conditioned video 阶段，用 visual-only 边界分数
    t >= exec_start_idx ：执行阶段，用标准三项边界分数

两个阶段在 exec_start_idx 处强制切换（finalize 当前 Event + reset 阈值统计）。
而 exec_start_idx **只在 pkl 里**，features/ 目录下没有 —— features 里只有
token_emb_{step}.npy，看不出哪一步是视频/执行的分界。

**默认全量扫描**（`--samples 0`）。早期版本默认随机抽 3 万个样本，理由
「exec_start_idx 是每条 episode 的常数，抽 3 万 ≈ 每条命中 75 次」在 Counting 上
成立，但**在 1600 episode 的 16 任务集上是每集 6.3% 覆盖率**：execution pkl 少的
episode 会一次都抽不到 → 落进 episodes_missing → build_event_cache 拒绝 → 一直
到 precompute_eventgraph_memory 才以「event cache 不完整」炸出来（实测 8 条：
809/1005/1009/1038/1040/1044/1096/1537，数据本身没问题）。代价错位太远，
所以覆盖优先：全量扫描 47.6 万个 pkl 在 32 进程下约 3 分钟（Counting 约 1 分钟）。

`--samples N`（>0）仍然保留，用于快速试跑/小数据集。这种情况下：
  · 仍然交叉验证 exec_start_idx 在 episode 内恒定（两个不同值 = 假设错了，报错）；
  · 但**只要还有 episode 没扫到就不会写文件，并以退出码 2 结束** —— 宁可让流水线
    在第一步停住，也不要写出一份「看起来成功」的部分 map。确实想保留部分结果时
    显式加 `--allow-missing`（会正常写出，payload 里 complete=false）。

用法：
    python scripts/scan_step_index.py --dataset-path data/realrobot_preprocessed
    python scripts/scan_step_index.py --dataset-path .../realrobot_preprocessed --num-workers 32
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

# 纯 CPU/numpy，强制 CPU 后端避免 import 时抢 GPU（训练可能已占满显存）
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np


def _read_one(path: str):
    """读一个 pkl，只取三个标量。不碰 image/wrist_image（400 KB 里的 393 KB）。"""
    try:
        with open(path, "rb") as f:
            d = pickle.load(f)
    except Exception as e:                      # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    try:
        return (
            int(np.asarray(d["epis_idx"]).reshape(-1)[0]),
            int(np.asarray(d["step_idx"]).reshape(-1)[0]),
            int(np.asarray(d["exec_start_idx"]).reshape(-1)[0]),
        ), None
    except (KeyError, ValueError, IndexError) as e:
        return None, f"字段缺失/异常: {type(e).__name__}: {e}"


def _worker(chunk: list[str]):
    out, errs = [], 0
    for p in chunk:
        rec, err = _read_one(p)
        if rec is None:
            errs += 1
        else:
            out.append(rec)
    return out, errs


def _chunks(seq: list, n: int):
    k, m = divmod(len(seq), n)
    start = 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        yield seq[start:start + size]
        start += size


def main() -> None:
    ap = argparse.ArgumentParser(description="扫描 pkl 取出每条 episode 的 exec_start_idx")
    ap.add_argument("--dataset-path", default="data/realrobot_preprocessed")
    ap.add_argument("--samples", type=int, default=0,
                    help="随机抽多少个 pkl。0（默认）= 全量扫描，保证每条 episode 都被覆盖；"
                         ">0 时是抽样试跑，若仍有 episode 没扫到则退出码 2 且不写文件")
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-missing", action="store_true",
                    help="允许 episodes_missing 非空并照常写出（默认不写、退出码 2）")
    ap.add_argument("--out", default=None,
                    help="输出路径，默认 <dataset-path>/meta/exec_start_map.json")
    args = ap.parse_args()

    data_dir = os.path.join(args.dataset_path, "data")
    feat_dir = os.path.join(args.dataset_path, "features")
    if not os.path.isdir(data_dir):
        sys.exit(f"找不到 {data_dir}")

    # ls 一个 18.9 万元素的目录本身就慢；用 os.scandir 一次拿完
    names = [e.name for e in os.scandir(data_dir) if e.name.endswith(".pkl")]
    n_total = len(names)
    if n_total == 0:
        sys.exit(f"{data_dir} 下没有 pkl")
    print(f"pkl 总数 {n_total}，本集目录 {feat_dir}")

    rng = random.Random(args.seed)
    k = n_total if args.samples <= 0 else min(args.samples, n_total)
    if k == n_total:
        # 全量：不洗牌。文件名有序 => 两次运行的结果顺序一致，出错也好复现。
        picked = [os.path.join(data_dir, n) for n in names]
        print(f"全量扫描 {k} 个 pkl，{args.num_workers} 进程读")
    else:
        picked = [os.path.join(data_dir, n) for n in rng.sample(names, k)]
        print(f"随机抽 {k} 个 pkl（{k / n_total:.1%}），{args.num_workers} 进程读")

    records: list[tuple[int, int, int]] = []
    n_err = 0
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        for out, errs in ex.map(_worker, _chunks(picked, args.num_workers * 4)):
            records.extend(out)
            n_err += errs
    print(f"读到 {len(records)} 条记录，{n_err} 个失败")

    # 按 episode 汇总，检查 exec_start_idx 在 episode 内是否恒定
    by_ep: dict[int, set[int]] = defaultdict(set)
    step_counts: dict[int, int] = defaultdict(int)
    for ep, step, esi in records:
        by_ep[ep].add(esi)
        step_counts[ep] += 1

    inconsistent = {ep: sorted(v) for ep, v in by_ep.items() if len(v) > 1}
    if inconsistent:
        print(f"\n✗ {len(inconsistent)} 条 episode 的 exec_start_idx 不恒定：")
        for ep, vals in list(inconsistent.items())[:10]:
            print(f"    episode_{ep}: {vals}")
        sys.exit("exec_start_idx 不是 episode 常数 —— 本脚本的假设不成立，"
                 "需要先查清语义再继续。")

    found_eps = sorted(by_ep)
    # 与 features 目录对照：哪些 episode 没扫到
    have_feat = {int(os.path.basename(p).split("_", 1)[1])
                 for p in glob.glob(os.path.join(feat_dir, "episode_*"))
                 if os.path.isdir(p)}
    missing = sorted(have_feat - set(found_eps))
    extra = sorted(set(found_eps) - have_feat)

    exec_map = {ep: next(iter(v)) for ep, v in by_ep.items()}
    per_ep_hits = sorted(step_counts.values())

    print(f"\n覆盖 episode：{len(found_eps)} / features 目录 {len(have_feat)}")
    if missing:
        print(f"  ⚠️ 未扫到的 episode {len(missing)} 条：{missing[:20]}{' ...' if len(missing) > 20 else ''}")
        print(f"     这些 episode 的 exec_start_idx 未知 → build_event_cache 会拒绝处理它们。"
              f"（每条 episode 命中数 min={per_ep_hits[0]} max={per_ep_hits[-1]}；"
              f"命中 0 次 = execution pkl 太少且没被抽中）")
    if extra:
        print(f"  ⚠️ features 里没有的 episode {len(extra)} 条：{extra[:10]}")

    vals = np.asarray(list(exec_map.values()))
    if len(vals):
        print(f"\nexec_start_idx 分布：min={vals.min()} p25={np.percentile(vals,25):.0f} "
              f"median={np.median(vals):.0f} p75={np.percentile(vals,75):.0f} max={vals.max()}")

    out_path = args.out or os.path.join(args.dataset_path, "meta", "exec_start_map.json")
    if missing and not args.allow_missing:
        # ⚠️ 这里必须硬失败，而且**不写文件**。教训（2026-10-12）：30000 抽样在 1600
        #    episode 上漏了 8 条，脚本照常写出并退出 0，流水线一路跑到
        #    precompute_eventgraph_memory 才以「event cache 不完整」炸掉 —— 失败点
        #    离病根两步远。宁可现在退出码 2，也不要留一份「看起来成功」的部分 map。
        print(f"\n✗ 覆盖不完整：{len(missing)} 条 episode 的 exec_start_idx 未知（见上）。")
        print(f"  重跑本脚本且**不带 --samples**（默认全量扫描，保证覆盖），"
              f"或加大 --samples。")
        print(f"  确实要保留这份部分结果：加 --allow-missing。")
        print(f"  {out_path} 未写出，磁盘上的旧文件保持原样。")
        sys.exit(2)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    payload = {
        "format": "event_graph.exec_start_map.v1",
        "n_pkl_total": n_total,
        "n_sampled": k,
        "n_records": len(records),
        "n_errors": n_err,
        "seed": args.seed,
        "exec_start_idx": {str(ep): v for ep, v in sorted(exec_map.items())},
        "episodes_missing": missing,
        # complete=false 只在 --allow-missing 下才可能出现（否则上面就退出了）
        "complete": not missing,
    }
    tmp = f"{out_path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, out_path)
    print(f"\n已写入 {out_path}")


if __name__ == "__main__":
    main()
