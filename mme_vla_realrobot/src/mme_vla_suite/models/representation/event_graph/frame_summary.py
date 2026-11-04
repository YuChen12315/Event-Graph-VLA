"""冻结 2048→512 投影 + 无参数帧摘要 f_t（v1.3 §2.1）。

    f_t = Normalize( (1/16) Σ_p PCA(P_t,p) ),   f_t ∈ R^512

三步流水：
    ① fit      —— 离线造一把 2048→512 的"尺子"，存成文件，以后只读不造
    ② project  —— 每帧 16 个 2048 维 token → 16 个 512 维，存成 fp16
    ③ summarize—— 16 个 512 维求平均再归一化 → 1 个 512 维 f_t

这是 Control Plane 唯一的视觉输入。它必须满足 v1.3 §2.1：

    同一 observation 在预计算脚本、training replay 和 online server 中必然得到
    同一 Control Plane 输入，不受 LoRA、task encoder 或训练 checkpoint 影响。

为此做了 4 处刻意"别扭"的写法，保证离线整段算与在线单帧算逐位一致
（v1.2 §3.3 / v1.3 §10 Phase 1 验收）：

1. 尺子冻结并落盘。离线与在线 load 同一个 .npz，绝不各自拟合。PCA 的特征向量
   符号是任意的，重拟合会整列翻号 → cache 静默失效，所以拟合时固定符号规则。

2. **矩阵乘按单帧做，不按 batch 做。** 这是刻意的反优化：
   (16,2048)@(2048,512) 与 (550*16,2048)@(2048,512) 的底层分块/累加路径由 BLAS 决定，
   不同后端、不同线程数下没有逐行相同的契约。**本机实测（numpy 1.26.4，默认 BLAS，
   550 帧一集）两者逐位相等，最大差 0.000e+00** —— 也就是说在当前环境下这条防护
   尚未体现出必要性。仍然保留，是因为离线预处理与在线 policy server 未必跑在同一
   线程配置下，而代价实测仅 0.12s vs 0.08s（400 集相差 0.8 分钟 vs 0.5 分钟）。
   用这点开销换掉一个"取决于 BLAS 实现"的假设是划算的。

3. 16 个 token 的池化用显式顺序累加，不用 `mean(axis=)`。`mean` 的成对求和分块
   策略依赖数组整体布局；显式 p=0..15 累加与形状无关。

4. 512 维的 L2 范数用 `np.sum(f*f)`，不用 `np.dot` —— 后者走 BLAS，
   线程数/构建不同会给出不同结果。

本模块的等价性保证是：**给定逐位相同的输入 token 与相同的 numpy 构建，输出逐位
相同**。上游 SigLIP 是否逐位确定（在线重编码 vs 离线缓存）不在本模块职责内。
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = ["FrozenFrameSummary", "PCA_ASSET_FORMAT"]

PCA_ASSET_FORMAT = "event_graph.pca.v1"


def _l2_normalize(f: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """按最后一维做 L2 归一化（模块 docstring 第 4 条：不走 BLAS）。"""
    sq = np.multiply(f, f)
    norm = np.sqrt(np.sum(sq, axis=-1, dtype=np.float32))
    inv = np.float32(1.0) / np.maximum(norm, np.float32(eps))
    return f * inv[..., None]


@dataclass
class FrozenFrameSummary:
    """冻结的 2048→512 线性投影 + 帧摘要。

    proj (2048, 512) float32 列已单位化的投影矩阵 W；mean (2048,) float32 训练集
    视觉 token 均值。两者 + 拟合超参一起决定 asset_version，变了就得重建 cache。
    """

    proj: np.ndarray
    mean: np.ndarray
    dim_in: int
    dim_out: int
    asset_version: str

    # ── ① 造尺子 ────────────────────────────────────────────────────────
    @classmethod
    def fit(
        cls,
        tokens: np.ndarray,
        dim_out: int = 512,
        *,
        n_fit_samples: int | None = None,
        seed: int = 0,
    ) -> "FrozenFrameSummary":
        """在训练集视觉 token 上拟合 PCA（v1.2 §1.2：必须离线固定并冻结）。

        tokens 是 (N, 2048) 的**原始（未投影）**视觉 token；dim_out 固定 512
        （§8）；n_fit_samples 给定时先按 seed 随机下采样，seed 必须固定并记入资产
        metadata，否则重拟合不可复现。

        走协方差 + eigh（2048×2048）而不是对 N×2048 做 SVD：内存与时间都可控。

        符号规则：每个主成分取其绝对值最大分量为正。不固定符号的话，重拟合会整列
        翻号 → 缓存里的 token 全部取反 → cos(f_i,f_j) 看起来还正常，但 f_t 参与的
        加法结构全被破坏，属于最难查的一类静默错误。
        """
        tokens = np.ascontiguousarray(tokens, dtype=np.float32)
        if tokens.ndim != 2:
            raise ValueError(f"期望 (N, dim_in) 的 token 矩阵，得到 {tokens.shape}")
        dim_in = tokens.shape[1]
        if dim_out > dim_in:
            raise ValueError(f"dim_out={dim_out} 超过 dim_in={dim_in}")
        # 秩缺陷防护。样本数 ≤ dim_out 时协方差秩 < dim_out，前 dim_out 个主成分必然
        # 解释 100% 方差 —— 这是秩缺陷的假象，不是真的没有信息损失。实测踩过：
        # 512 个样本拟合 512 维，解释方差读出 1.0000。
        if tokens.shape[0] <= dim_out:
            raise ValueError(
                f"拟合样本数 {tokens.shape[0]} 不超过输出维度 {dim_out}，协方差秩不足。"
                f"此时前 {dim_out} 个主成分会假装解释 100% 方差。需要 > {dim_out} 个样本。")
        if tokens.shape[0] < 4 * dim_in:
            warnings.warn(
                f"拟合样本数 {tokens.shape[0]} 少于 4×dim_in={4 * dim_in}，"
                f"协方差尾部特征值不可靠，解释方差比例会被高估。"
                f"建议每个 episode 多采几步或增加 episode 数。",
                RuntimeWarning, stacklevel=2)

        if n_fit_samples is not None and n_fit_samples < tokens.shape[0]:
            rng = np.random.default_rng(seed)
            idx = rng.choice(tokens.shape[0], size=n_fit_samples, replace=False)
            idx.sort()          # 排序只影响可复现的展示，不影响协方差
            tokens = tokens[idx]

        mean = tokens.mean(axis=0).astype(np.float32)
        centered = tokens - mean
        # 协方差用 float64 累加：2048 维下 float32 累加十几万项的相对误差约 1e-4，
        # 足以让接近简并的特征向量方向发生可见漂移 —— 特征向量一漂，cache 就废了。
        cov = centered.T.astype(np.float64) @ centered.astype(np.float64)
        cov /= max(1, tokens.shape[0] - 1)

        eigvals, eigvecs = np.linalg.eigh(cov)          # 升序
        order = np.argsort(eigvals)[::-1][:dim_out]
        proj = eigvecs[:, order].astype(np.float32)
        eigvals = eigvals[order]

        # 符号固定（见 docstring）
        pivot = np.argmax(np.abs(proj), axis=0)
        signs = np.sign(proj[pivot, np.arange(dim_out)])
        signs[signs == 0] = 1.0
        proj = np.ascontiguousarray(proj * signs[np.newaxis, :])

        asset_version = _hash_asset(proj, mean, seed, tokens.shape[0])
        obj = cls(proj=proj, mean=mean, dim_in=dim_in, dim_out=dim_out,
                  asset_version=asset_version)
        # 诊断量：回答"512 维够不够"。不参与序列化，只给拟合脚本打印。
        obj._n_fit = int(tokens.shape[0])
        obj._seed = int(seed)
        obj._kept_var = float(eigvals.sum())
        obj._total_var = float(np.trace(cov))
        obj._explained = (obj._kept_var / obj._total_var) if obj._total_var > 0 else 0.0
        return obj

    # ── 尺子的存取 ──────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(
            path,
            format=np.asarray(PCA_ASSET_FORMAT),
            proj=self.proj,
            mean=self.mean,
            dim_in=np.asarray(self.dim_in),
            dim_out=np.asarray(self.dim_out),
            asset_version=np.asarray(self.asset_version),
            fit_meta=np.asarray(json.dumps({
                "n_fit": getattr(self, "_n_fit", None),
                "seed": getattr(self, "_seed", None),
                "kept_var": getattr(self, "_kept_var", None),
                "total_var": getattr(self, "_total_var", None),
                "explained_variance_ratio": getattr(self, "_explained", None),
            })),
        )

    @classmethod
    def load(cls, path: str) -> "FrozenFrameSummary":
        """读尺子。**文件不存在直接报错，不现场拟合** —— 见模块 docstring 第 1 条。"""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"冻结 PCA 资产不存在：{path}\n"
                f"Event-Graph 的 Control Plane 不允许现场拟合 —— 先跑 "
                f"scripts/build_event_cache.py 生成资产。")
        with np.load(path, allow_pickle=False) as z:
            fmt = str(z["format"])
            if fmt != PCA_ASSET_FORMAT:
                raise ValueError(f"PCA 资产格式不匹配：{fmt} != {PCA_ASSET_FORMAT}")
            return cls(
                proj=np.ascontiguousarray(z["proj"], dtype=np.float32),
                mean=np.ascontiguousarray(z["mean"], dtype=np.float32),
                dim_in=int(z["dim_in"]),
                dim_out=int(z["dim_out"]),
                asset_version=str(z["asset_version"]),
            )

    def fit_meta(self, path: str) -> dict:
        """读回 fit 时写下的诊断信息（解释方差等）。"""
        with np.load(path, allow_pickle=False) as z:
            return json.loads(str(z["fit_meta"])) if "fit_meta" in z else {}

    # ── ② 量尺子 ────────────────────────────────────────────────────────
    def project(self, tokens: np.ndarray, cache_dtype: str = "float16") -> np.ndarray:
        """原始视觉 token → PCA token（即 event cache 的内容）。

        tokens : (..., 16, 2048) 任意浮点
        返回   : (..., 16, dim_out) 指定 dtype

        逐帧做（模块 docstring 第 2 条）。返回形状与输入前导维一致。
        """
        tokens = np.asarray(tokens, dtype=np.float32)
        if tokens.shape[-1] != self.dim_in:
            raise ValueError(f"最后一维应为 {self.dim_in}，得到 {tokens.shape[-1]}")
        lead = tokens.shape[:-2]
        n_tok = tokens.shape[-2]
        flat = tokens.reshape(-1, n_tok, self.dim_in)

        out_dt = np.dtype(cache_dtype)
        out = np.empty((flat.shape[0], n_tok, self.dim_out), dtype=np.float32)
        for f in range(flat.shape[0]):
            # 固定 shape (16, 2048) @ (2048, 512) —— 与单帧在线路径完全一致。
            # 实测与整批 matmul 逐位相同（见模块 docstring 第 2 条），此处是防 BLAS
            # 后端/线程数差异，不是防已观测到的偏差。
            out[f] = (flat[f] - self.mean[np.newaxis, :]) @ self.proj
        return out.reshape(*lead, n_tok, self.dim_out).astype(out_dt, copy=False)

    # ── ③ 压成一句话 ────────────────────────────────────────────────────
    def summarize(self, pca_tokens: np.ndarray) -> np.ndarray:
        """PCA token → 帧摘要 f_t（v1.3 §2.1）。

        pca_tokens : (..., 16, dim_out) —— **必须是 cache 里的那份精度**（默认 fp16）
        返回       : (..., dim_out) float32，已 L2 归一化

        先升到 float32 再池化：缓存是 fp16，若在 fp16 上池化会引入与在线不同的舍入。
        """
        x = np.asarray(pca_tokens, dtype=np.float32)
        if x.shape[-1] != self.dim_out:
            raise ValueError(f"最后一维应为 {self.dim_out}，得到 {x.shape[-1]}")
        n_tok = x.shape[-2]

        lead = x.shape[:-2]
        flat = x.reshape(-1, n_tok, self.dim_out)
        acc = np.zeros((flat.shape[0], self.dim_out), dtype=np.float32)
        # 显式顺序累加（模块 docstring 第 3 条）：与数组整体布局无关
        for p in range(n_tok):
            acc += flat[:, p, :]
        acc /= np.float32(n_tok)
        return _l2_normalize(acc).reshape(*lead, self.dim_out)

    def summarize_raw(self, tokens: np.ndarray, cache_dtype: str = "float16") -> np.ndarray:
        """便捷路径：原始 token → f_t，等价于 summarize(project(tokens))。

        在线路径用这个 —— **必须**先落回 cache 精度再池化，否则与离线读数不一致。
        """
        return self.summarize(self.project(tokens, cache_dtype=cache_dtype))

    # ── 验收 ────────────────────────────────────────────────────────────
    def parity_check(
        self,
        tokens: np.ndarray,
        frame_index: int = 0,
        cache_dtype: str = "float16",
    ) -> tuple[bool, float]:
        """离线（整段一次算） vs 在线（取单帧单独算）是否逐位一致。

        这正是 v1.3 §10 Phase 1 的验收项。返回 (是否逐位一致, 最大绝对差)。

        tokens : (T, 16, 2048) 一段连续帧的原始 token

        用 tobytes() 而不是 np.array_equal：后者认为 -0.0 == 0.0，会漏掉符号位差异，
        而符号位差异恰恰是 PCA 符号问题的表征。
        """
        t = np.asarray(tokens, dtype=np.float32)
        if t.ndim != 3 or t.shape[1] != 16:
            raise ValueError(f"期望 (T,16,dim_in)，得到 {t.shape}")
        if not 0 <= frame_index < t.shape[0]:
            raise IndexError(f"frame_index={frame_index} 越界（T={t.shape[0]}）")

        off_all = self.summarize_raw(t, cache_dtype=cache_dtype)[frame_index]
        on_one = self.summarize_raw(t[frame_index:frame_index + 1], cache_dtype=cache_dtype)[0]
        if off_all.tobytes() == on_one.tobytes():
            return True, 0.0
        return False, float(np.abs(off_all - on_one).max())


def _hash_asset(proj: np.ndarray, mean: np.ndarray, seed: int, n_fit: int) -> str:
    """PCA 资产指纹（v1.3 §10 Phase 1 验收：cache 版本写入 metadata）。

    只比对这 16 位十六进制，不需要比对 2048×512 的浮点矩阵。
    注意与 GraphMemoryConfig.cache_fingerprint() 是两个层级：
      本函数管"尺子变没变"；cache_fingerprint() 管"配置字段变没变"。
    """
    h = hashlib.sha256()
    h.update(PCA_ASSET_FORMAT.encode())
    h.update(np.ascontiguousarray(proj, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(mean, dtype=np.float32).tobytes())
    h.update(json.dumps({"seed": int(seed), "n_fit": int(n_fit)}, sort_keys=True).encode())
    return h.hexdigest()[:16]


def sample_training_tokens(
    feature_dir: str,
    episodes: Sequence[int],
    steps_per_episode: int,
    *,
    seed: int = 0,
    spatial_key: str = "image_emb_4x4",
) -> np.ndarray:
    """从训练集随机子集采样原始视觉 token，供 `FrozenFrameSummary.fit` 使用。

    返回 (N, 2048) float32。每个 episode 均匀抽 `steps_per_episode` 个决策步，
    每步取该帧**全部 16 个** token —— f_t 的定义就是这 16 个求平均，若每帧只取 1 个
    token，协方差会低估帧内空间分布。

    按步号整数排序：glob 的字典序下 token_emb_100 排在 token_emb_20 前面，
    直接取"前 k 个"会让采样偏向小步号。
    """
    import glob
    rng = np.random.default_rng(seed)
    chunks: list[np.ndarray] = []
    for ep in episodes:
        ep_dir = os.path.join(feature_dir, f"episode_{ep}")
        files = sorted(
            glob.glob(os.path.join(ep_dir, "token_emb_*.npy")),
            key=lambda p: int(os.path.basename(p).rsplit("_", 1)[1].split(".")[0]),
        )
        if not files:
            continue
        k = min(steps_per_episode, len(files))
        pick = np.sort(rng.choice(len(files), size=k, replace=False))
        for i in pick:
            with open(files[int(i)], "rb") as fh:
                feat = np.load(fh, allow_pickle=True).item()
            arr = np.asarray(feat[spatial_key], dtype=np.float32)
            if arr.ndim == 3:       # 部分导出带一个 view 维
                arr = arr[0]
            chunks.append(arr)      # (16, 2048)
    if not chunks:
        raise RuntimeError(f"没有从 {feature_dir} 采到任何 token")
    return np.concatenate(chunks, axis=0)
