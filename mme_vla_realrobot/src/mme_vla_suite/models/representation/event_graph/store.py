"""Episode 级连续 event cache 存储（v1.2 §1.4 / v1.3 §10 Phase 1）。

**本模块存的是「原料」，不是「Event」。** 这一点必须分清：

    Event 是任意长度的语义片段（由 Phase 2 的边界检测切出来，3 帧到上百帧都可能）。
    L=8 是「从每个 Event 里抽几帧」的采样预算，**不是 Event 的长度**。
    哪 8 帧要等边界检测跑完才知道，所以缓存不可能只存 8 帧/Event —— 它存全部 T 帧。

每个 episode 只开 2 个文件：

    event_tok_{ep}.npy   [T, 16, 512] fp16   全部 T 帧的 PCA token
    event_sum_{ep}.npy   [T, 512]     fp32   全部 T 帧的 f_t
    event_meta_{ep}.json                      步骤号 / exec_start_idx / 版本指纹

存全部帧之所以可行，是因为降维 + 压精度：一集 550 帧 × 16 × 512 × 2B = 9 MB。
对比不降维的 [550,16,2048] fp32 = 72 MB，以及按步物化 96 节点 payload 的
[96,8,16,2048] fp32 = 12.6 MB/样本（v1.1 §3）。

为什么必须连续存：现有 features/episode_*/ 下每个 episode 有 2201 个小 npy。
Phase 3 的 dataset loader 每个样本要按 event_frame_ids 读 L=8 帧 —— 8000 样本 × 8 =
6.4 万次小文件 open，dataloader 会直接崩。v1.1 §8 明文禁止「在 dataloader 中每个
sample 随机打开数百个小 .npy」。memmap 之后每集只开 2 个 fd。

为什么 f_t 要落盘而不是现算：Control Plane 每步都要 f_t（边界检测的输入），
现算等于每步多一次 16×2048 matmul。f_t 只有 512 维、每集几百 KB，存下来更划算。
落盘的 f_t 与在线现算的 f_t 由 frame_summary.parity_check 保证逐位一致。

磁盘占用（counting 全集 189,035 步）：event_tok 3.1 GB + event_sum 0.4 GB ≈ 3.5 GB。
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np

from .config import GraphMemoryConfig
from .frame_summary import FrozenFrameSummary

__all__ = ["EventFeatureStore", "CACHE_FORMAT", "list_episodes", "episode_step_ids"]

CACHE_FORMAT = "event_graph.cache.v1"

# ⚠️ 沿用仓库既有约定：frame_id == 该 episode 内的行号 == token_emb_{step}.npy 的 step 号。
# 实测 counting 数据集每条 episode 的 step 都是 0..T-1 连续（T 因 episode 而异，
# 366/472/550 不等，总计 189,035 与 pkl 数吻合）。构建时 assert 连续性 —— 若将来数据集
# 出现空洞，宁可在这里炸掉，也不要静默读错行。
_STEP_CONTIGUITY_ASSERT = True


def list_episodes(feature_dir: str) -> list[int]:
    """列出现有 episode 编号，升序。"""
    eps = []
    for p in glob.glob(os.path.join(feature_dir, "episode_*")):
        if os.path.isdir(p):
            try:
                eps.append(int(os.path.basename(p).split("_", 1)[1]))
            except ValueError:
                continue
    return sorted(eps)


def episode_step_ids(feature_dir: str, epis_idx: int) -> list[int]:
    """一条 episode 的全部决策步号，升序。"""
    d = os.path.join(feature_dir, f"episode_{epis_idx}")
    files = glob.glob(os.path.join(d, "token_emb_*.npy"))
    return sorted(int(os.path.basename(p).rsplit("_", 1)[1].split(".")[0]) for p in files)


def _cache_paths(feature_dir: str, epis_idx: int) -> tuple[str, str, str]:
    d = os.path.join(feature_dir, f"episode_{epis_idx}")
    return (os.path.join(d, f"event_tok_{epis_idx}.npy"),
            os.path.join(d, f"event_sum_{epis_idx}.npy"),
            os.path.join(d, f"event_meta_{epis_idx}.json"))


@dataclass
class EventMeta:
    epis_idx: int
    n_steps: int
    step_ids: list[int]
    exec_start_idx: int
    event_dim: int
    token_per_image: int
    event_cache_dtype: str
    cache_fingerprint: str
    pca_asset_version: str
    built_at: str
    format: str = CACHE_FORMAT

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=True)


class EventFeatureStore:
    """一条 episode 的 event cache 读句柄（memmap）。

    典型用法（训练侧按索引读）：::

        store = EventFeatureStore(feature_dir, epis_idx, cfg, pca=fs)
        tok = store.pca_tokens(event_frame_ids)    # (N, L, 16, 512) fp16 → Data Plane
        f   = store.summary(rep_frame_ids)         # (N, 512) fp32       → Control Plane
        s   = store.all_summaries()                # (T, 512)            → 整集回放

    注意 pca_tokens 收的是**帧号**，返回的是那几帧的 token；「哪 8 帧」由 Phase 2 的
    边界检测决定（v1.2 §1.3 的 sample_event_frames），本模块不参与切分。

    frame_id 就是 episode 内的行号（见模块顶部说明）。
    """

    def __init__(
        self,
        feature_dir: str,
        epis_idx: int,
        cfg: GraphMemoryConfig,
        pca: FrozenFrameSummary | None = None,
        *,
        verify: bool = True,
    ):
        self.feature_dir = feature_dir
        self.epis_idx = int(epis_idx)
        self.cfg = cfg
        self._tok_path, self._sum_path, self._meta_path = _cache_paths(feature_dir, epis_idx)

        # 三个文件一起查，且把 episode 号和**完整路径**写进错误里。
        # 否则 np.load 在几百个 worker 里炸出来只有一句 "No such file or directory"，
        # 连是哪一集、找的是哪个路径都看不出来 —— 排查成本远高于这三行。
        missing = [p for p in (self._meta_path, self._tok_path, self._sum_path)
                   if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"episode {epis_idx} 的 event cache 不完整，缺 {len(missing)} 个文件：\n"
                + "".join(f"  缺: {p}\n" for p in missing)
                + f"  （event cache 目录 = {os.path.dirname(self._meta_path)}）\n"
                f"  先跑 scripts/build_event_cache.py。若三件套齐全却仍报错，"
                f"多半是启动时的工作目录不对 —— --dataset-path 是相对路径。")
        with open(self._meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if raw.get("format") != CACHE_FORMAT:
            raise ValueError(f"cache 格式不匹配：{raw.get('format')} != {CACHE_FORMAT}")
        self.meta = EventMeta(**raw)

        if verify:
            self._verify(cfg, pca)

        # mmap_mode='r'：只读，按 frame_id 切片时才真正读盘
        self._tok = np.load(self._tok_path, mmap_mode="r")
        self._sum = np.load(self._sum_path, mmap_mode="r")

    def __repr__(self) -> str:
        return (f"EventFeatureStore(ep={self.epis_idx}, T={self.meta.n_steps}, "
                f"exec_start={self.meta.exec_start_idx}, "
                f"pca={self.meta.pca_asset_version})")

    # ── 版本校验 ────────────────────────────────────────────────────────
    def _verify(self, cfg: GraphMemoryConfig, pca: FrozenFrameSummary | None) -> None:
        """cache 是否仍然对应当前配置与当前 PCA 资产。

        两个层级缺一不可（见 frame_summary._hash_asset 的说明）：
          cache_fingerprint  —— 配置字段（event_dim / dtype / token_per_image）变没变
          pca_asset_version  —— 那把「尺子」变没变

        改 kappa、lambda_b 这类 Control Plane 参数**不进指纹**：它们只影响 Phase 2 的
        切分决策，不影响缓存里的像素。否则每调一次参就要重建 3.5 GB，没法干活。
        """
        want = cfg.cache_fingerprint()
        if self.meta.cache_fingerprint != want:
            raise ValueError(
                f"episode {self.epis_idx} 的 cache 指纹与当前配置不符：\n"
                f"  cache: {self.meta.cache_fingerprint}\n"
                f"  配置 : {want}\n"
                f"影响 cache 内容的字段（event_dim / event_cache_dtype / token_per_image / "
                f"pca_asset_version）任一改动都必须重建 cache。")
        if pca is not None and self.meta.pca_asset_version != pca.asset_version:
            raise ValueError(
                f"episode {self.epis_idx} 的 cache 用了另一把 PCA 尺子：\n"
                f"  cache: {self.meta.pca_asset_version}\n"
                f"  当前 : {pca.asset_version}\n"
                f"重拟合 PCA 会让缓存语义静默失效，必须重建 cache。")

    # ── 形状自检 ────────────────────────────────────────────────────────
    @property
    def n_steps(self) -> int:
        return self.meta.n_steps

    @property
    def exec_start_idx(self) -> int:
        """执行阶段起始步。v1.3 §5.1：t < exec_start_idx 是 conditioned video 阶段。

        注意 v1.3 §5.1 相对 v1.0 的变化：video 阶段**必须进入** Control Plane 与长期
        记忆，只是在 video 阶段改用 visual-only 的边界分数，并在 exec_start_idx 处
        强制 finalize 当前 Event + 切换 phase。不要拿这个字段去过滤掉 video 帧。
        """
        return self.meta.exec_start_idx

    def _check(self) -> None:
        exp_tok = (self.meta.n_steps, self.cfg.token_per_image, self.cfg.event_dim)
        exp_sum = (self.meta.n_steps, self.cfg.event_dim)
        if self._tok.shape != exp_tok:
            raise ValueError(f"event_tok 形状 {self._tok.shape} != 期望 {exp_tok}")
        if self._sum.shape != exp_sum:
            raise ValueError(f"event_sum 形状 {self._sum.shape} != 期望 {exp_sum}")

    def _as_rows(self, frame_ids) -> np.ndarray:
        idx = np.asarray(frame_ids)
        if not np.issubdtype(idx.dtype, np.integer):
            raise TypeError(f"frame_id 必须是整数，得到 {idx.dtype}")
        # 负索引会被 numpy 静默当成「从末尾数」—— 这是最难查的一类越界：
        # frame_id 算错成 -1 不会报错，会安静地读到最后一帧。
        if idx.size and (idx.min() < 0 or idx.max() >= self.meta.n_steps):
            raise IndexError(
                f"frame_id 越界：范围 [{idx.min()}, {idx.max()}]，"
                f"本 episode 有 {self.meta.n_steps} 步")
        return idx

    # ── 读 ─────────────────────────────────────────────────────────────
    def pca_tokens(self, frame_ids) -> np.ndarray:
        """PCA token → Data Plane 的 L=8 输入。

        frame_ids : (...,) int  任意形状，通常来自 Phase 2 的 event_frame_ids [N, L]
        返回      : (..., 16, 512) fp16
        """
        self._check()
        return self._tok[self._as_rows(frame_ids)]

    def summary(self, frame_ids) -> np.ndarray:
        """帧摘要 f_t → Control Plane 输入。

        frame_ids : (...,) int
        返回      : (..., 512) fp32（已 L2 归一化）
        """
        self._check()
        return self._sum[self._as_rows(frame_ids)]

    def all_pca_tokens(self) -> np.ndarray:
        """整集 PCA token (T, 16, 512)。回放用。"""
        self._check()
        return np.asarray(self._tok)

    def all_summaries(self) -> np.ndarray:
        """整集 f_t (T, 512)。回放用 —— Control Plane 顺序推进时吃这个。"""
        self._check()
        return np.asarray(self._sum)

    def summary_range(self, start: int, end: int) -> np.ndarray:
        """[start, end] 闭区间。对齐 v1.3 §2.3 的 Event 帧区间语义（含两端）。"""
        if start > end:
            raise ValueError(f"start={start} > end={end}")
        return self.summary(np.arange(start, end + 1))


# ══════════════════════════════════════════════════════════════════════
# 写入侧（供 build_event_cache.py 调用）
# ══════════════════════════════════════════════════════════════════════
def write_episode_cache(
    feature_dir: str,
    epis_idx: int,
    cfg: GraphMemoryConfig,
    pca: FrozenFrameSummary,
    step_ids: list[int],
    exec_start_idx: int,
    *,
    force: bool = False,
) -> tuple[int, bool]:
    """读原始 token → 投影 → 落盘 event_tok / event_sum / event_meta。

    返回 (写入步数, 是否跳过)。断点续跑：meta 已存在且指纹一致时跳过。

    整集读进来再一次投影，比逐帧读小文件快得多 —— 小文件 open 是主要成本，
    不是计算。一集 550 步的原始 token 是 72 MB，内存无压力。
    """
    tok_path, sum_path, meta_path = _cache_paths(feature_dir, epis_idx)

    if os.path.exists(meta_path) and not force:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                old = json.load(f)
            if (old.get("cache_fingerprint") == cfg.cache_fingerprint()
                    and old.get("pca_asset_version") == pca.asset_version
                    and old.get("n_steps") == len(step_ids)):
                return 0, True
        except (json.JSONDecodeError, OSError):
            pass    # 残缺的 meta → 重写

    if _STEP_CONTIGUITY_ASSERT and step_ids != list(range(len(step_ids))):
        raise ValueError(
            f"episode {epis_idx} 的步号不是 0..T-1 连续："
            f"{step_ids[:5]}...{step_ids[-3:]}。store.py 约定 frame_id == 行号，"
            f"需要改成显式的 step→row 映射表后再构建。")

    raw = _read_raw_tokens(feature_dir, epis_idx, step_ids, cfg)   # (T, 16, 2048) fp32
    tok = pca.project(raw, cache_dtype=cfg.event_cache_dtype)      # (T, 16, 512) fp16
    summ = pca.summarize(tok)                                      # (T, 512) fp32

    _atomic_save(tok_path, tok)
    _atomic_save(sum_path, summ)
    _atomic_write(meta_path, EventMeta(
        epis_idx=epis_idx,
        n_steps=len(step_ids),
        step_ids=[int(s) for s in step_ids],
        exec_start_idx=int(exec_start_idx),
        event_dim=cfg.event_dim,
        token_per_image=cfg.token_per_image,
        event_cache_dtype=cfg.event_cache_dtype,
        cache_fingerprint=cfg.cache_fingerprint(),
        pca_asset_version=pca.asset_version,
        built_at=_now(),
    ).to_json())
    return len(step_ids), False


def _read_raw_tokens(
    feature_dir: str, epis_idx: int, step_ids: list[int], cfg: GraphMemoryConfig,
    spatial_key: str = "image_emb_4x4",
) -> np.ndarray:
    """读一条 episode 的原始视觉 token，拼成 (T, 16, 2048) fp32。"""
    d = os.path.join(feature_dir, f"episode_{epis_idx}")
    out = np.empty((len(step_ids), cfg.token_per_image, 2048), dtype=np.float32)
    for i, s in enumerate(step_ids):
        p = os.path.join(d, f"token_emb_{s}.npy")
        with open(p, "rb") as fh:
            feat = np.load(fh, allow_pickle=True).item()
        arr = np.asarray(feat[spatial_key], dtype=np.float32)
        if arr.ndim == 3:       # 部分导出带一个 view 维
            arr = arr[0]
        if arr.shape != (cfg.token_per_image, 2048):
            raise ValueError(
                f"{p} 的 {spatial_key} 形状是 {arr.shape}，期望 "
                f"({cfg.token_per_image}, 2048)")
        out[i] = arr
    return out


# ── 原子写：中断时不会留下半个文件 ──────────────────────────────────────
# 构建脚本支持断点续跑（已存在的跳过），留半个文件会被当成「已完成」跳过 —— 那是灾难。
def _atomic_save(path: str, arr: np.ndarray) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def _atomic_write(path: str, text: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def _now() -> str:
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")
