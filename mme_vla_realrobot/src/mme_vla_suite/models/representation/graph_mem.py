"""TCGM 的记忆选择器（图 + DPP，任务条件）。

把「选哪 32 帧」做成模型前向里的一个可学习模块。

图建立在两根干净的轴上（都用现成特征，不学习、不依赖 progress）：
- 视觉轴：图像 SigLIP 特征（判断"两帧看起来像不像" → 去冗余）；
- 相位/时序轴：3D 位置编码 pos_emb（判断"两帧在不在同一时刻/位置" → 保持计数/时间覆盖）。

二者由任务条件权重 α = σ(MLP_α(task_vec)) 融合，拼成组合特征 φ，跑贪心 k-DPP
（quality = 时效衰减，diversity = φ 的几何展开）选 K 帧。

为什么相位轴用 pos_emb 而不是 TCN 学习：验证发现原始 SigLIP 特征逐 episode 的第一主成分
与 progress 相关 0.84（相位已在原始特征里），而 TCN 时间对比在周期/计数任务上会把相位流形
折叠掉（相关掉到 0.09），因为「swing 第 1 次」与「swing 第 3 次」视觉相同却时间很远，
TCN 被迫用噪声去分。故相位轴改用现成的 3D 位置编码（单调、无 train/eval 漂移、免费）。

可训练信号：L_div（选中帧成对相似度，训练 α）。tcn_loss 保留仅作验证/分析用，模型不再使用。
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import einops

import openpi.shared.array_typing as at
from mme_vla_suite.models.representation.utils import kernel_init


def _cosine_matrix(x: at.Float[at.Array, "b n d"]) -> at.Float[at.Array, "b n n"]:
    """成对余弦相似度矩阵 (b, n, n)。"""
    x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    return jnp.einsum("bnd,bmd->bnm", x, x)


def tcn_loss(
    z: at.Float[at.Array, "b n d"],
    times: at.Float[at.Array, "b n"],
    mask: at.Bool[at.Array, "b n"],
    tau: float = 0.1,
    pos_window: float = 0.05,
) -> at.Float[at.Array, ""]:
    """时间对比损失（InfoNCE）。仅用于验证/分析，模型已不用 TCN 相位轴。"""
    sim = _cosine_matrix(z) / tau
    n = z.shape[1]
    eye = jnp.eye(n, dtype=jnp.bool_)[None]
    valid = mask[:, :, None] & mask[:, None, :] & (~eye)
    time_dist = jnp.abs(times[:, :, None] - times[:, None, :])
    pos_mask = valid & (time_dist <= pos_window)
    pos_lse = jax.nn.logsumexp(jnp.where(pos_mask, sim, -1e9), axis=-1)
    all_lse = jax.nn.logsumexp(jnp.where(valid, sim, -1e9), axis=-1)
    has_pos = pos_mask.any(axis=-1)
    per_anchor = all_lse - pos_lse
    return jnp.sum(jnp.where(has_pos, per_anchor, 0.0)) / (jnp.sum(has_pos) + 1e-8)


def greedy_kdpp_indices_jax(
    phi: at.Float[at.Array, "b n d"],
    quality: at.Float[at.Array, "b n"],
    k: int,
) -> at.Int[at.Array, "b k"]:
    """贪心 k-DPP（L-ensemble L_ij = q_i q_j <φ_i, φ_j>）。

    每步选 q_i^2 * residual_i^2 最大的帧，residual 是候选相对已选子空间的正交距离平方。
    Python 静态展开，等价于「quality 加权的最远点采样」。
    """
    b, n, d = phi.shape
    phi = phi / (jnp.linalg.norm(phi, axis=-1, keepdims=True) + 1e-8)
    q = quality

    selected_mask = jnp.zeros((b, n), dtype=jnp.bool_)
    hard_list = []
    basis = []
    residual_sq = jnp.ones((b, n), dtype=phi.dtype)

    for _ in range(k):
        scores = (q * q) * residual_sq
        scores = jnp.where(selected_mask, -1.0, scores)
        best = jnp.argmax(scores, axis=-1)
        hard_list.append(best)
        selected_mask = selected_mask | jax.nn.one_hot(best, n).astype(jnp.bool_)

        best_phi = jnp.take_along_axis(phi, best[:, None, None], axis=1)[:, 0, :]
        v = best_phi
        for u in basis:
            v = v - jnp.sum(u * v, axis=-1, keepdims=True) * u
        vn = jnp.linalg.norm(v, axis=-1, keepdims=True)
        u_new = v / (vn + 1e-8)
        basis.append(u_new)

        proj = jnp.einsum("bnd,bd->bn", phi, u_new)
        residual_sq = jnp.maximum(residual_sq - proj * proj, 0.0)

    return jnp.stack(hard_list, axis=-1)


class GraphMemorySelector(nnx.Module):
    """图 + DPP 记忆选择器：任务条件融合（视觉 × 时序） + 选 K 帧。"""

    def __init__(self, config, rngs: nnx.Rngs, dtype, task_dim: int):
        self.cfg = config
        self.dtype = dtype

        self.top_k = int(config.perceptual_memory.get("top_k", config.budget // config.token_per_image))
        self.recency_decay = float(config.perceptual_memory.get("recency_decay", 0.5))

        # 任务条件融合权重：task_vec -> α ∈ (0,1)
        self.alpha_head = nnx.Linear(
            task_dim, 1, rngs=rngs, dtype=dtype, kernel_init=kernel_init
        )

    def __call__(
        self,
        candidate_image_emb: at.Float[at.Array, "b w p d1"],
        candidate_mask: at.Bool[at.Array, "b w"],
        candidate_pos_emb: at.Float[at.Array, "b w p d2"],
        candidate_state_emb: at.Float[at.Array, "b w d3"],
        task_vec: at.Float[at.Array, "b d"],
    ):
        b, w, p, d1 = candidate_image_emb.shape

        # 视觉轴：mean-pool 图像 patch token，L2 归一
        frame_feat = candidate_image_emb.mean(axis=2)  # (b, w, d1)
        frame_feat_norm = frame_feat / (jnp.linalg.norm(frame_feat, axis=-1, keepdims=True) + 1e-8)

        # 相位/时序轴：mean-pool 3D 位置编码，L2 归一
        pos_feat = candidate_pos_emb.mean(axis=2)  # (b, w, d2)
        pos_feat_norm = pos_feat / (jnp.linalg.norm(pos_feat, axis=-1, keepdims=True) + 1e-8)

        # 任务条件 α ∈ (0,1)
        alpha = jax.nn.sigmoid(self.alpha_head(task_vec))  # (b, 1)

        # φ = [√α·视觉, √(1-α)·时序]，<φ_i,φ_j> = α·视觉相似 + (1-α)·时序相似
        phi = jnp.concatenate(
            [
                jnp.sqrt(alpha)[:, :, None] * frame_feat_norm,
                jnp.sqrt(1.0 - alpha)[:, :, None] * pos_feat_norm,
            ],
            axis=-1,
        )  # (b, w, d1 + d2)

        # quality = 时效衰减（最新帧 q=1，最老帧 q=recency_decay）
        times = jnp.arange(w, dtype=jnp.float32)[None] / max(w, 1)
        recency = jnp.power(self.recency_decay, 1.0 - times)
        recency = jnp.where(candidate_mask, recency, 0.0)
        recency = jnp.broadcast_to(recency, (b, w))

        # 贪心 k-DPP（非可微，前向给出离散选择）
        hard_idx = greedy_kdpp_indices_jax(phi, recency, self.top_k)  # (b, k)

        # gather 选中的帧（完整 patch 特征）
        k = self.top_k
        pos_d = candidate_pos_emb.shape[-1]
        state_d = candidate_state_emb.shape[-1]
        sel_image = jnp.take_along_axis(
            candidate_image_emb, jnp.broadcast_to(hard_idx[:, :, None, None], (b, k, p, d1)), axis=1
        )
        sel_pos = jnp.take_along_axis(
            candidate_pos_emb, jnp.broadcast_to(hard_idx[:, :, None, None], (b, k, p, pos_d)), axis=1
        )
        sel_state = jnp.take_along_axis(
            candidate_state_emb, jnp.broadcast_to(hard_idx[:, :, None], (b, k, state_d)), axis=1
        )

        # 展平成 (b, k*p, d)，喂给 FeatureEncoder
        sel_image_flat = einops.rearrange(sel_image, "b k p d -> b (k p) d")
        sel_pos_flat = einops.rearrange(sel_pos, "b k p d -> b (k p) d")
        sel_state_flat = einops.repeat(sel_state, "b k d -> b (k p) d", p=p)
        sel_mask_flat = jnp.ones((b, k * p), dtype=jnp.bool_)

        # 供辅助损失（L_div）的统计量
        selection_indicator = jax.nn.one_hot(hard_idx, w).sum(axis=1)  # (b, w)，k-hot
        A = jnp.einsum("bnd,bmd->bnm", phi, phi)  # (b, w, w)，α 融合相似度
        stats = {
            "A": A,
            "M": jax.lax.stop_gradient(selection_indicator),
        }

        return sel_image_flat, sel_pos_flat, sel_state_flat, sel_mask_flat, stats
