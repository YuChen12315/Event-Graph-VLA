#!/usr/bin/env python
"""把 Event-Graph 每帧真正用到的 payload 抽成每 episode 一个连续数组。

为什么需要
----------
训练时每个样本要读 32 个长期节点 + 8 个近期帧的 2048-d payload。原来每帧都是一个
单独的 token_emb_*.npy：约 590 KB 的 pickle dict，里面 8x8 / 4x4 / 2x2 三套视觉
token 都在，而 Event-Graph 只用 4x4 的那 180 KB。结果是每个样本要 open + 反序列化
40 次；实测热缓存下仍要 ~14 ms/样本，而且 40 个小文件把 cluster FS 的延迟全部暴露
出来（冷启动 ~70 ms/样本）。

这里一次性把每集的 4x4 视觉 token / 位置编码 / 关节角按帧号连续写成三个 .npy。
训练侧用 mmap + fancy index 一次取多帧；数据页由所有 worker 共享的 OS page cache
提供，不需要每个 worker 各复制一份。

输出（每个 episode）
--------------------
    features/episode_<id>/eg_payload/
        img.npy    (T, 16, 2048) float32   —— image_emb_4x4
        pos.npy    (T, 16, 768)  float32   —— pos_emb_4x4
        state.npy  (T, 8)        float32   —— state_emb
        meta.json                          —— 最后写；存在即表示缓存完整

⚠️ 只抽 4x4。Event-Graph 的 token_per_image=16，其它分辨率（8x8/2x2）不参与；
   不要拿这份缓存去喂别的 memory 类型。

用法
----
    .venv/bin/python scripts/build_eg_payload_cache.py \
        --dataset-path data/realrobot_preprocessed --workers 32

    # 只重建部分 episode
    .venv/bin/python scripts/build_eg_payload_cache.py \
        --dataset-path data/realrobot_preprocessed \
        --episodes 0 1 2 --force
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:  # token_emb 里的 image_emb_4x4 是 bfloat16；显式注册，避免 pickle 解析踩环境
    import ml_dtypes  # noqa: F401
except ImportError:  # pragma: no cover - jax 依赖里一定有
    pass

PAYLOAD_VERSION = 1
_FRAME_RE = re.compile(r"token_emb_(\d+)\.npy$")
# 4x4 视觉 token + 位置编码 + 关节角，float32 每帧的字节数（用于磁盘预估）。
# 关节角按 42 维（真机）预留；RoboMME 的 8 维只会更小，预估偏保守不影响正确性。
_BYTES_PER_FRAME = (16 * 2048 + 16 * 768 + 42) * 4


def list_episodes(feature_dir: Path) -> list[int]:
    eps = []
    for p in feature_dir.iterdir():
        if p.is_dir() and p.name.startswith("episode_"):
            try:
                eps.append(int(p.name.split("_", 1)[1]))
            except ValueError:
                continue
    return sorted(eps)


def frame_ids(ep_dir: Path) -> list[int]:
    ids = []
    for p in ep_dir.glob("token_emb_*.npy"):
        m = _FRAME_RE.fullmatch(p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def read_frame(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """一个 token_emb 文件 → (img [16,2048], pos [16,768], state [d3])，float32。"""
    with open(path, "rb") as f:
        feat = np.load(f, allow_pickle=True).item()
    img = np.asarray(feat["image_emb_4x4"], np.float32)
    if img.ndim == 3:                      # 部分导出带一个 view 维
        img = img[0]
    pos = np.asarray(feat["pos_emb_4x4"], np.float32)
    if pos.ndim == 3:
        pos = pos[0]
    state = np.asarray(feat["state_emb"], np.float32).reshape(-1)
    return img, pos, state


def _atomic_save(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def build_episode(ep_dir: Path, pool: ThreadPoolExecutor, *, force: bool) -> str:
    """返回 "skip" / "built"。失败抛异常，由调用方汇总。"""
    out_dir = ep_dir / "eg_payload"
    meta_p = out_dir / "meta.json"
    fids = frame_ids(ep_dir)
    if not fids:
        raise RuntimeError("没有 token_emb_*.npy")
    T = len(fids)
    if fids != list(range(T)):
        missing = sorted(set(range(fids[-1] + 1)) - set(fids))[:5]
        raise RuntimeError(f"帧号不连续 0..{fids[-1]}，缺 {missing} ...")

    steps_p = ep_dir / "eventgraph" / "steps.npz"
    if steps_p.exists():
        with np.load(steps_p) as s:
            n_steps = int(s["step_ids"].shape[0])
        if n_steps != T:
            raise RuntimeError(f"steps.npz 有 {n_steps} 帧，token_emb 有 {T} 帧")

    if meta_p.exists() and not force:
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if int(meta.get("version", -1)) == PAYLOAD_VERSION and int(meta["n_frames"]) == T:
                return "skip"
        except Exception:
            pass                        # 旧 meta 坏了 → 重建

    img0, pos0, state0 = read_frame(ep_dir / f"token_emb_{fids[0]}.npy")
    img = np.empty((T, *img0.shape), np.float32)
    pos = np.empty((T, *pos0.shape), np.float32)
    state = np.empty((T, *state0.shape), np.float32)
    img[0], pos[0], state[0] = img0, pos0, state0

    futures = {
        pool.submit(read_frame, ep_dir / f"token_emb_{f}.npy"): f
        for f in fids[1:]
    }
    for fut in as_completed(futures):
        f = futures[fut]
        a, b, c = fut.result()
        if a.shape != img0.shape or b.shape != pos0.shape or c.shape != state0.shape:
            raise RuntimeError(
                f"帧 {f} 形状不一致：{a.shape}/{b.shape}/{c.shape} != "
                f"{img0.shape}/{pos0.shape}/{state0.shape}")
        img[f], pos[f], state[f] = a, b, c

    out_dir.mkdir(parents=True, exist_ok=True)
    # 先删旧 meta：中途失败时调用方会看到「没有 meta」并回退，而不是读到半份缓存
    if meta_p.exists():
        meta_p.unlink()
    _atomic_save(out_dir / "img.npy", img)
    _atomic_save(out_dir / "pos.npy", pos)
    _atomic_save(out_dir / "state.npy", state)

    # 回读抽检 2 帧，防 squeeze/state 顺序类的静默错位
    for f in {0, T // 2, T - 1}:
        a, b, c = read_frame(ep_dir / f"token_emb_{f}.npy")
        if not (np.array_equal(img[f], a) and np.array_equal(pos[f], b)
                and np.array_equal(state[f], c)):
            raise RuntimeError(f"回读校验失败：帧 {f} 与 token_emb 不一致")

    meta = {
        "version": PAYLOAD_VERSION,
        "n_frames": T,
        "img_shape": list(img0.shape),
        "pos_shape": list(pos0.shape),
        "state_shape": list(state0.shape),
        "dtype": "float32",
        "source": "token_emb_*.npy:image_emb_4x4,pos_emb_4x4,state_emb",
    }
    _atomic_write_text(meta_p, json.dumps(meta, ensure_ascii=False, indent=2))
    return "built"


def episode_n_frames(ep_dir: Path) -> int:
    steps_p = ep_dir / "eventgraph" / "steps.npz"
    if steps_p.exists():
        with np.load(steps_p) as s:
            return int(s["step_ids"].shape[0])
    return len(frame_ids(ep_dir))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-path", required=True,
                    help="预处理数据集根目录（其下有 features/episode_*）")
    ap.add_argument("--workers", type=int, default=32,
                    help="并发读取 token_emb 的线程数（I/O 型，默认 32）")
    ap.add_argument("--episodes", type=int, nargs="*", default=None,
                    help="只处理这些 episode（默认全部）")
    ap.add_argument("--force", action="store_true", help="已有缓存也重建")
    ap.add_argument("--dry-run", action="store_true",
                    help="只估计帧数/磁盘占用，不写文件")
    args = ap.parse_args()

    root = Path(args.dataset_path)
    feature_dir = root / "features"
    if not feature_dir.is_dir():
        print(f"[error] 找不到 {feature_dir}", file=sys.stderr)
        return 2

    episodes = list_episodes(feature_dir)
    if args.episodes:
        wanted = set(args.episodes)
        episodes = [e for e in episodes if e in wanted]
        missing = sorted(wanted - set(episodes))
        if missing:
            print(f"[error] 这些 episode 不存在：{missing[:10]}", file=sys.stderr)
            return 2
    if not episodes:
        print("[error] 没有可处理的 episode", file=sys.stderr)
        return 2

    total_frames = sum(episode_n_frames(feature_dir / f"episode_{e}") for e in episodes)
    need = total_frames * _BYTES_PER_FRAME
    free = shutil.disk_usage(feature_dir).free
    print(f"[plan] episodes={len(episodes)} frames={total_frames} "
          f"cache={need / 1e9:.1f} GB free={free / 1e9:.1f} GB", flush=True)
    if need + 5e9 > free:
        print("[error] 磁盘空间不足；先清理或分批跑 --episodes", file=sys.stderr)
        return 2
    if args.dry_run:
        return 0

    t_start = time.time()
    built = skipped = 0
    failures: list[tuple[int, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for i, ep in enumerate(episodes, 1):
            ep_dir = feature_dir / f"episode_{ep}"
            t0 = time.time()
            try:
                status = build_episode(ep_dir, pool, force=args.force)
            except Exception as exc:
                failures.append((ep, repr(exc)))
                print(f"[fail] ep {ep}: {exc}", file=sys.stderr, flush=True)
                continue
            if status == "skip":
                skipped += 1
            else:
                built += 1
            if i % 20 == 0 or i == len(episodes):
                elapsed = time.time() - t_start
                rate = i / max(elapsed, 1e-9)
                eta = (len(episodes) - i) / max(rate, 1e-9)
                print(f"[{i}/{len(episodes)}] built={built} skipped={skipped} "
                      f"fail={len(failures)} last_ep={ep} "
                      f"last={time.time() - t0:.1f}s elapsed={elapsed / 60:.1f}min "
                      f"eta={eta / 60:.1f}min", flush=True)

    print(f"[done] built={built} skipped={skipped} failed={len(failures)} "
          f"elapsed={(time.time() - t_start) / 60:.1f}min", flush=True)
    if failures:
        print("[failed episodes]", failures[:20], file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
