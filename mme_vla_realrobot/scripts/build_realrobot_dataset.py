"""把 Astribot 真机采集的 LeRobot v3.0 数据转成 π0.5 / Event-Graph 训练格式。

原始结构（每个任务文件夹一套）::

    Real_Memory_Data/
      A/
        meta/info.json
        meta/episodes/chunk-000/file-*.parquet
        meta/tasks.parquet
        data/chunk-000/file-*.parquet
        videos/head_camera.obs.rgb/chunk-000/file-*.mp4
        videos/stereo_camera.obs.rgb/...        # 不用

输出（与 RoboMME 预处理数据集完全一致）::

    <dataset-path>/
      data/<sample_id>.pkl                      # 每帧一个
      features/episode_<epis_idx>/token_emb_<t>.npy
      meta/stats.json                           # execution_samples / total_samples
      meta/episodes.json                        # epis_idx -> folder/local episode/length/prompt
      meta/state_stats.json                     # 42 维 raw state mean/std（填 EG yaml 用）

42 维约定（用户确认）::

    state  = joint_pos[0:18] + left_hand.obs(12) + right_hand.obs(12)
    action = joints_position_command[0:18] + left_hand.act(12) + right_hand.act(12)
    # joint_pos[0:18] = 腰 4 + 左臂 7 + 右臂 7；底盘 3 维、头部 2 维在采集中锁死，丢弃

只使用 head 相机；视频按行索引与 parquet 对齐（不做时间戳重采样）。

依赖：repo 自带的 polars（读 parquet）+ imageio-ffmpeg（解码 mp4）+ jax
（SigLIP / MemoryBuffer）。`--check-only` / `--dry-run` 不会 import jax。

用法::

    # 只检查格式
    python scripts/build_realrobot_dataset.py --check-only

    # 试转前 2 条
    python scripts/build_realrobot_dataset.py --episodes 0 1 --yes

    # 全量 A（GPU 被占时可加 --cpu，但会慢很多）
    python scripts/build_realrobot_dataset.py --folders A --yes

    # 采集新增 B 之后增量追加（已转换的 (folder, local episode) 自动跳过）
    python scripts/build_realrobot_dataset.py --folders A B --yes
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

ACTION_CHUNK_HORIZON = 20
STATE_DIM = 42          # 4 waist + 7 + 7 arms + 12 + 12 hands
ACTION_DIM = 42
IMAGE_SIZE = 224
HEAD_VIDEO_KEY = "head_camera.obs.rgb"
FEATURE_KEYS = ("image_emb_4x4", "pos_emb_4x4", "state_emb")

RAW_DATA_COLUMNS = (
    "episode_index",
    "frame_index",
    "failure",
    "astribot.obs.joint_pos",
    "astribot.obs.joints_position_command",
    "left_hand.obs.joint_pos",
    "right_hand.obs.joint_pos",
    "left_hand.act.joint_pos",
    "right_hand.act.joint_pos",
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def discover_folders(raw_root: Path) -> list[Path]:
    folders = sorted(
        p for p in raw_root.iterdir()
        if p.is_dir() and (p / "meta" / "info.json").is_file()
    )
    if not folders:
        sys.exit(f"{raw_root} 下没有找到任何含 meta/info.json 的文件夹")
    return folders


def load_info(folder: Path) -> dict:
    with open(folder / "meta" / "info.json", encoding="utf-8") as f:
        return json.load(f)


def load_episodes(folder: Path) -> list[dict]:
    files = sorted((folder / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not files:
        sys.exit(f"{folder}/meta/episodes 下没有 parquet")
    cols = [
        "episode_index", "tasks", "length",
        "data/chunk_index", "data/file_index",
        "dataset_from_index", "dataset_to_index",
        f"videos/{HEAD_VIDEO_KEY}/chunk_index",
        f"videos/{HEAD_VIDEO_KEY}/file_index",
        f"videos/{HEAD_VIDEO_KEY}/from_timestamp",
        f"videos/{HEAD_VIDEO_KEY}/to_timestamp",
    ]
    frames = [pl.read_parquet(p, columns=cols) for p in files]
    df = pl.concat(frames, how="vertical")
    rows = df.sort("episode_index").to_dicts()
    return rows


def load_prompts(path: Path | None) -> dict[str, str]:
    if path is None or not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        sys.exit(f"{path} 必须是 {{文件夹: 指令}} 的 JSON 对象")
    return {str(k): str(v).strip().lower() for k, v in raw.items()}


def build_state_action(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    joint = df["astribot.obs.joint_pos"].to_numpy()                 # (T, 23)
    cmd = df["astribot.obs.joints_position_command"].to_numpy()     # (T, 20)
    lh_obs = df["left_hand.obs.joint_pos"].to_numpy()               # (T, 12)
    rh_obs = df["right_hand.obs.joint_pos"].to_numpy()              # (T, 12)
    lh_act = df["left_hand.act.joint_pos"].to_numpy()               # (T, 12)
    rh_act = df["right_hand.act.joint_pos"].to_numpy()              # (T, 12)

    state = np.concatenate([joint[:, :18], lh_obs, rh_obs], axis=1).astype(np.float32)
    action = np.concatenate([cmd[:, :18], lh_act, rh_act], axis=1).astype(np.float32)
    if state.shape[1] != STATE_DIM or action.shape[1] != ACTION_DIM:
        raise ValueError(f"state/action 维度不是 {STATE_DIM}: {state.shape} {action.shape}")
    return state, action


def build_action_chunks(actions: np.ndarray, horizon: int = ACTION_CHUNK_HORIZON) -> np.ndarray:
    """(T, D) -> (T, horizon, D)，episode 末尾复制最后一个有效 action。"""
    t = actions.shape[0]
    idx = np.arange(t)[:, None] + np.arange(horizon)[None, :]
    idx = np.minimum(idx, t - 1)
    return actions[idx]


def read_raw_episode(folder: Path, info: dict, ep: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = folder / info["data_path"].format(
        chunk_index=int(ep["data/chunk_index"]),
        file_index=int(ep["data/file_index"]),
    )
    if not data_path.is_file():
        raise FileNotFoundError(f"缺少数据文件 {data_path}")
    df = pl.read_parquet(data_path, columns=list(RAW_DATA_COLUMNS))
    ep_idx = int(ep["episode_index"])
    df = df.filter(pl.col("episode_index") == ep_idx)
    if df.height != int(ep["length"]):
        raise ValueError(
            f"{folder.name} episode {ep_idx}: parquet 行数 {df.height} != 元数据长度 {ep['length']}")
    frame_index = df["frame_index"].to_numpy()
    if not np.array_equal(frame_index, np.arange(df.height)):
        raise ValueError(f"{folder.name} episode {ep_idx}: frame_index 不是 0..T-1 连续")
    state, action = build_state_action(df)
    failure = df["failure"].to_numpy()
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"{folder.name} episode {ep_idx}: state/action 含 NaN/Inf")
    return state, action, failure


def iter_video_frames(video_path: Path, start_frame: int, n_frames: int):
    import imageio

    reader = imageio.get_reader(str(video_path), "ffmpeg")
    try:
        it = reader.iter_data()
        for _ in range(start_frame):
            next(it)
        for _ in range(n_frames):
            yield np.asarray(next(it))
    finally:
        reader.close()


def resize_frames(frames: np.ndarray) -> np.ndarray:
    """(t, H, W, 3) uint8 -> (t, 224, 224, 3) uint8（保持长宽比 + 黑边）。"""
    from openpi.shared import image_tools

    out = image_tools.resize_with_pad(frames, IMAGE_SIZE, IMAGE_SIZE)
    return np.asarray(out, dtype=np.uint8)


def _existing_episodes(dataset_path: Path) -> tuple[dict, dict]:
    meta_dir = dataset_path / "meta"
    episodes = {}
    if (meta_dir / "episodes.json").is_file():
        with open(meta_dir / "episodes.json", encoding="utf-8") as f:
            episodes = json.load(f)
    stats = {"execution_samples": 0, "total_samples": 0}
    if (meta_dir / "stats.json").is_file():
        with open(meta_dir / "stats.json", encoding="utf-8") as f:
            stats = json.load(f)
    return episodes, stats


def _select_episodes(episodes: list[dict], args) -> list[dict]:
    picked = episodes
    if args.episodes:
        wanted = set(args.episodes)
        picked = [e for e in picked if int(e["episode_index"]) in wanted]
        missing = wanted - {int(e["episode_index"]) for e in picked}
        if missing:
            sys.exit(f"该文件夹没有这些本地 episode：{sorted(missing)}")
    if args.limit is not None:
        picked = picked[:args.limit]
    return picked


def _video_info(folder: Path, info: dict, ep: dict, fps: float) -> tuple[Path, int, int]:
    video_path = (
        folder / info["video_path"].format(
            video_key=HEAD_VIDEO_KEY,
            chunk_index=int(ep[f"videos/{HEAD_VIDEO_KEY}/chunk_index"]),
            file_index=int(ep[f"videos/{HEAD_VIDEO_KEY}/file_index"]),
        )
    )
    start = int(round(float(ep[f"videos/{HEAD_VIDEO_KEY}/from_timestamp"]) * fps))
    end = int(round(float(ep[f"videos/{HEAD_VIDEO_KEY}/to_timestamp"]) * fps))
    return video_path, start, end - start


def check_folder(folder: Path, args, prompts: dict[str, str]) -> dict:
    info = load_info(folder)
    if info.get("codebase_version") != "v3.0":
        _log(f"  ⚠️ {folder.name}: codebase_version={info.get('codebase_version')}，本脚本按 v3.0 解析")
    fps = float(info.get("fps", 30))
    episodes = _select_episodes(load_episodes(folder), args)
    total_frames = 0
    checked_videos = 0
    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        state, action, failure = read_raw_episode(folder, info, ep)
        video_path, start, n_video = _video_info(folder, info, ep, fps)
        if not video_path.is_file():
            raise FileNotFoundError(f"{folder.name} episode {ep_idx}: 缺少视频 {video_path}")
        if abs(n_video - state.shape[0]) > 1:
            raise ValueError(
                f"{folder.name} episode {ep_idx}: 视频时长换算 {n_video} 帧 != parquet {state.shape[0]} 帧")
        if not args.skip_video_check:
            import imageio
            reader = imageio.get_reader(str(video_path), "ffmpeg")
            try:
                n_decoded = int(reader.count_frames())
            finally:
                reader.close()
            if n_decoded != state.shape[0]:
                raise ValueError(
                    f"{folder.name} episode {ep_idx}: 视频解码 {n_decoded} 帧 != parquet {state.shape[0]} 帧")
            checked_videos += 1
        total_frames += state.shape[0]
        _log(
            f"  {folder.name}/{ep_idx:03d}: {state.shape[0]} 帧, "
            f"state{state.shape} action{action.shape}, failure={int(failure.sum())}, "
            f"prompt='{prompts.get(folder.name, folder.name.lower())}'")
    return {
        "folder": folder.name,
        "episodes": len(episodes),
        "frames": total_frames,
        "videos_checked": checked_videos,
        "fps": fps,
    }


def estimate_bytes(frames: int) -> int:
    pkl = IMAGE_SIZE * IMAGE_SIZE * 3 + ACTION_CHUNK_HORIZON * ACTION_DIM * 4 + STATE_DIM * 4 + 2048
    feat = 16 * 2048 * 4 + 16 * 768 * 4 + STATE_DIM * 4 + 1024   # float32，与 MemoryBuffer 实测一致
    return int(frames * (pkl + feat))


def write_episode(
    dataset_path: Path,
    folder: Path,
    info: dict,
    ep: dict,
    epis_idx: int,
    sample_start: int,
    prompt: str,
    mem_buffer,
    chunk: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    """写一条 episode 的 pkl + features。返回 (帧数, state_sum, state_sumsq)。"""
    state, action, failure = read_raw_episode(folder, info, ep)
    t_total = state.shape[0]
    action_chunks = build_action_chunks(action)
    video_path, video_start, n_video = _video_info(folder, info, ep, float(info.get("fps", 30)))
    if n_video != t_total:
        raise ValueError(
            f"{folder.name} episode {int(ep['episode_index'])}: 视频时长换算 {n_video} 帧 "
            f"!= parquet {t_total} 帧")

    data_dir = dataset_path / "data"
    feat_dir = dataset_path / "features" / f"episode_{epis_idx}"
    feat_dir.mkdir(parents=True, exist_ok=True)
    for stale in feat_dir.glob("token_emb_*.npy"):
        stale.unlink()

    frames = np.empty((t_total, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    it = iter_video_frames(video_path, video_start, n_video)
    for start in range(0, t_total, chunk):
        end = min(start + chunk, t_total)
        raw = np.stack([next(it) for _ in range(start, end)], axis=0)
        frames[start:end] = resize_frames(raw)

        states = state[start:end]
        # (t, v, h, w, 3) —— MemoryBuffer 期望的布局
        mem_buffer.add_buffer(frames[start:end, None], states, list(range(start, end)),
                              exec_start_idx=0)
        for t in range(start, end):
            feat = mem_buffer.get_history_feats(t)
            np.save(feat_dir / f"token_emb_{t}.npy", {k: feat[k] for k in FEATURE_KEYS})
        mem_buffer.clear()

        for t in range(start, end):
            frame_dict = {
                "image": frames[t],
                "state": state[t],
                "actions": action_chunks[t],
                "is_demo": np.array([False], dtype=np.bool_),
                "exec_start_idx": np.array([0], dtype=np.int32),
                "step_idx": np.array([t], dtype=np.int32),
                "epis_idx": np.array([epis_idx], dtype=np.int32),
                "prompt": prompt,
                "simple_subgoal": "",
                "grounded_subgoal": "",
                "simple_subgoal_online": "",
                "grounded_subgoal_online": "",
                "progress": float(t / max(t_total - 1, 1)),
                "failure": int(failure[t]),
            }
            pkl_path = data_dir / f"{sample_start + t}.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(frame_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return t_total, state.sum(axis=0, dtype=np.float64), (state.astype(np.float64) ** 2).sum(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Astribot 真机数据 -> π0.5 / Event-Graph 训练格式")
    ap.add_argument(
        "--raw-root",
        default=str(Path(__file__).resolve().parents[2] / "Real_Memory_Data"),
        help="原始 LeRobot 根目录（默认：与本项目同级的 Real_Memory_Data）",
    )
    ap.add_argument("--dataset-path", default="data/realrobot_preprocessed")
    ap.add_argument("--folders", nargs="*", default=None,
                    help="只处理这些文件夹（默认全部）")
    ap.add_argument("--episodes", type=int, nargs="*", default=None,
                    help="每个文件夹内只处理这些本地 episode_index")
    ap.add_argument("--limit", type=int, default=None,
                    help="每个文件夹最多处理前 N 条（试跑用）")
    ap.add_argument("--prompts", default=None,
                    help="文件夹 -> 指令 的 JSON，默认 <raw-root>/prompts.json")
    ap.add_argument("--chunk", type=int, default=32,
                    help="每批送进 SigLIP 的帧数")
    ap.add_argument("--cpu", action="store_true", help="强制 CPU 编码（GPU 被占时用，慢很多）")
    ap.add_argument("--check-only", action="store_true", help="只校验格式，不写任何文件")
    ap.add_argument("--dry-run", action="store_true", help="校验并打印磁盘预估，不写文件")
    ap.add_argument("--skip-video-check", action="store_true",
                    help="--check-only 时不解码视频核对帧数（只核对时间戳换算）")
    ap.add_argument("--overwrite", action="store_true", help="删掉已有输出目录后全量重建")
    ap.add_argument("--yes", action="store_true", help="确认真正写入（否则等同 --dry-run）")
    args = ap.parse_args()

    if args.cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    raw_root = Path(args.raw_root).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not raw_root.is_dir():
        sys.exit(f"找不到原始数据目录 {raw_root}")

    prompts_path = Path(args.prompts).expanduser() if args.prompts else raw_root / "prompts.json"
    prompts = load_prompts(prompts_path)
    if prompts:
        _log(f"指令文件: {prompts_path} ({len(prompts)} 条)")
    else:
        _log(f"指令文件不存在/为空: {prompts_path}，回退为文件夹名")

    folders = discover_folders(raw_root)
    if args.folders:
        wanted = set(args.folders)
        folders = [p for p in folders if p.name in wanted]
        missing = wanted - {p.name for p in folders}
        if missing:
            sys.exit(f"找不到文件夹：{sorted(missing)}")

    _log("=" * 70)
    _log("格式检查")
    _log("=" * 70)
    summaries = [check_folder(f, args, prompts) for f in folders]
    total_frames = sum(s["frames"] for s in summaries)
    _log("-" * 70)
    for s in summaries:
        _log(f"  {s['folder']}: {s['episodes']} 条, {s['frames']} 帧, {s['fps']:g} fps, "
             f"视频核对 {s['videos_checked']} 条")
    _log(f"  合计: {sum(s['episodes'] for s in summaries)} 条, {total_frames} 帧")
    _log(f"  预计磁盘: {estimate_bytes(total_frames) / 1e9:.1f} GB")

    if args.check_only or not args.yes:
        if not args.check_only:
            _log("\n未加 --yes：仅校验，不写文件。确认无误后加 --yes 执行转换。")
        return

    if args.overwrite and dataset_path.exists():
        _log(f"--overwrite: 删除 {dataset_path}")
        shutil.rmtree(dataset_path)
    dataset_path.mkdir(parents=True, exist_ok=True)
    (dataset_path / "data").mkdir(exist_ok=True)
    (dataset_path / "features").mkdir(exist_ok=True)
    (dataset_path / "meta").mkdir(exist_ok=True)

    episodes_meta, stats = _existing_episodes(dataset_path)
    done = {(v["folder"], int(v["episode"])) for v in episodes_meta.values()}
    next_epis_idx = max((int(k) for k in episodes_meta), default=-1) + 1
    next_sample_id = int(stats.get("execution_samples", 0))
    state_sum = np.zeros(STATE_DIM, dtype=np.float64)
    state_sumsq = np.zeros(STATE_DIM, dtype=np.float64)
    state_count = 0
    # 续算已有统计
    old_stats_path = dataset_path / "meta" / "state_stats.json"
    if old_stats_path.is_file() and next_sample_id > 0:
        with open(old_stats_path, encoding="utf-8") as f:
            old = json.load(f)
        state_count = int(old.get("count", 0))
        if state_count:
            mean = np.asarray(old["mean"], dtype=np.float64)
            std = np.asarray(old["std"], dtype=np.float64)
            state_sum = mean * state_count
            state_sumsq = (std ** 2 + mean ** 2) * state_count

    if next_sample_id:
        _log(f"\n增量：已有 {len(episodes_meta)} 条 / {next_sample_id} 帧，从 epis_idx={next_epis_idx} 继续")

    # 延迟 import：--check-only / --dry-run 不该初始化 JAX
    from mme_vla_suite.shared.mem_buffer import MemoryBuffer

    mem_buffer = MemoryBuffer(
        num_views=1,
        img_emb_dim=2048,
        pos_emb_dim=768,
        state_emb_dim=STATE_DIM,
        compute_token_drop_score=False,
        token_drop_stride=8,
        prepare_buffer=True,
    )

    t0 = time.time()
    written_eps = written_frames = skipped = 0
    for folder in folders:
        info = load_info(folder)
        fps = float(info.get("fps", 30))
        episodes = _select_episodes(load_episodes(folder), args)
        prompt = prompts.get(folder.name, folder.name.lower())
        for ep in episodes:
            local_idx = int(ep["episode_index"])
            if (folder.name, local_idx) in done:
                skipped += 1
                continue
            epis_idx = next_epis_idx
            n, ssum, ssq = write_episode(
                dataset_path, folder, info, ep, epis_idx, next_sample_id,
                prompt, mem_buffer, args.chunk)
            episodes_meta[str(epis_idx)] = {
                "folder": folder.name,
                "episode": local_idx,
                "length": n,
                "prompt": prompt,
                "fps": fps,
            }
            next_epis_idx += 1
            next_sample_id += n
            state_sum += ssum
            state_sumsq += ssq
            state_count += n
            written_eps += 1
            written_frames += n
            stats = {"execution_samples": next_sample_id, "total_samples": next_sample_id}
            _atomic_json(dataset_path / "meta" / "episodes.json", episodes_meta)
            _atomic_json(dataset_path / "meta" / "stats.json", stats)
            mean = state_sum / max(state_count, 1)
            var = np.maximum(state_sumsq / max(state_count, 1) - mean ** 2, 0.0)
            _atomic_json(dataset_path / "meta" / "state_stats.json", {
                "count": state_count,
                "mean": mean.tolist(),
                "std": np.sqrt(var).tolist(),
            })
            elapsed = time.time() - t0
            _log(f"[{written_eps}] {folder.name}/{local_idx:03d} -> episode_{epis_idx}: "
                 f"{n} 帧, 累计 {written_frames} 帧, {elapsed:.0f}s")

    mean = state_sum / max(state_count, 1)
    var = np.maximum(state_sumsq / max(state_count, 1) - mean ** 2, 0.0)
    _atomic_json(dataset_path / "meta" / "state_stats.json", {
        "count": state_count,
        "mean": mean.tolist(),
        "std": np.sqrt(var).tolist(),
    })
    _atomic_json(dataset_path / "meta" / "stats.json",
                 {"execution_samples": next_sample_id, "total_samples": next_sample_id})
    _atomic_json(dataset_path / "meta" / "episodes.json", episodes_meta)

    _log("\n完成：")
    _log(f"  新写 {written_eps} 条 / {written_frames} 帧，跳过已存在 {skipped} 条")
    _log(f"  数据集共 {len(episodes_meta)} 条 / {next_sample_id} 帧")
    _log(f"  输出: {dataset_path}")


if __name__ == "__main__":
    main()
