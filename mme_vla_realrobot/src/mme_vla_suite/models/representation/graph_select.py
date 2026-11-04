"""TCGM v3/v4：语言引导的图选帧器。

由 `perceptual_memory.type == "graph_select"` 且 `perceptual_memory.variant` 选择两种变体：

- variant = "hard"（v4-A，yaml: perceptual-graphselect-modul_v2.yaml）
  GNN 打分 → **STE 硬 top-k** → 直接选 top_k 个**真实帧**。
  前向是精确 gather（真实帧，信息零损失），反向走 softmax 权重（STE），选帧可训练。
  训练/推理跑完全相同的代码路径（无 RNG、无 train/eval 分支）。

- variant = "soft"（v4-B，yaml: perceptual-graphselect-modul_v3.yaml）
  GNN → DiffPool 软分配 S = softmax(assign_head) → top_k 个**软原型**（记忆量更大）。
  相比 v3 原版补了三项护栏 + 一个修复：
    ① entropy 损失：推 S 行近 one-hot（防"所有原型≈全局均值"的糊化塌缩）；
    ② balance 损失：||SᵀS/‖SᵀS‖_F − I/√k‖²，推 k 个原型互不重复（防原型退化重复）；
    ③ temporal-spread 损失：约束 k 个原型的时间重心铺满 [0,1]（v1 隐式赢在"时间铺开"，这里显式化）；
    ④ 修复：原型按列 mass 归一（原 v3 是加权**和**，不同原型尺度不一致）+ 死原型真 mask。

两者共用的前端（保证 A/B 可对比）：
- 候选池：由 buffer 统一产出（prepare_candidate_graph_dpp，与 graph_dpp 相同），无 train/eval 漂移；
- 节点特征：pooled 图像（可选）× pooled 3D 位置 × task_vec（语言引导）；
- 图：时间链邻接（自环 + 相邻帧），row-normalized；
- 消息传递：H' = silu(A_norm @ H @ W)。

`use_visual_axis`（默认 false）控制节点特征是否含视觉轴。v1(α 塌到 0)/v2(α=0.5 视觉核)
的对比证据表明：视觉相似是"选记忆"的**错误轴**（把视觉相同的事件如 swing 第1次 vs 第3次
合并/去重），时间位置多样性才是对的轴。故两个变体默认都关掉视觉轴；
若想复现"带视觉轴"的版本，把 yaml 里 use_visual_axis 改成 true 即可（一行，无代码改动）。

语言引导（借鉴 videograph / VideoGraph）：task_vec 同时注入节点特征和打分/分配头。
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import einops

import openpi.shared.array_typing as at
from mme_vla_suite.models.representation.utils import kernel_init


def _temporal_adjacency(w: int) -> jnp.ndarray:
    """时间链邻接（自环 + 相邻帧），row-normalized，返回 (w, w)。"""
    A = jnp.eye(w) + jnp.eye(w, k=1) + jnp.eye(w, k=-1)
    deg = A.sum(axis=-1, keepdims=True)
    return A / (deg + 1e-8)


def _temporal_spread_loss(S: at.Float[at.Array, "b w k"]) -> at.Float[at.Array, ""]:
    """护栏③：约束 k 个原型的时间重心在 [0,1] 上均匀铺开。

    候选帧按时间升序（hybrid_frame_sampling_indices 返回 earlier + recent，均升序），
    故**候选下标即时间轴**，无需额外的 time 输入。
    做法：算每个原型的质量加权时间重心 → 排序 → 与均匀位置 i/(k-1) 做 MSE。
    """
    b, w, k = S.shape
    t = jnp.linspace(0.0, 1.0, w, dtype=S.dtype)[None, :, None]   # (1, w, 1)
    mass = S.sum(axis=1) + 1e-8                                    # (b, k)
    centroid = jnp.sum(S * t, axis=1) / mass                       # (b, k)
    target = jnp.linspace(0.0, 1.0, k, dtype=S.dtype)[None, :]     # (1, k)
    return jnp.mean((jnp.sort(centroid, axis=-1) - target) ** 2)


def _balance_loss(S: at.Float[at.Array, "b w k"]) -> at.Float[at.Array, ""]:
    """护栏②：鼓励 k 个原型互不重复（DiffPool 的 balance 正则）。"""
    k = S.shape[-1]
    P = jnp.einsum("bwk,bwl->bkl", S, S)                                   # (b, k, k)
    P = P / (jnp.linalg.norm(P, axis=(-2, -1), keepdims=True) + 1e-8)
    I = jnp.eye(k, dtype=S.dtype) / jnp.sqrt(k)
    return jnp.mean(jnp.sum((P - I) ** 2, axis=(-2, -1)))


def _entropy_loss(
    S: at.Float[at.Array, "b w k"], mask: at.Bool[at.Array, "b w"]
) -> at.Float[at.Array, ""]:
    """护栏①：鼓励每帧近乎硬分配（S 行近 one-hot），防软融合把细节糊掉。"""
    H = -jnp.sum(S * jnp.log(S + 1e-8), axis=-1)                # (b, w)
    H = jnp.where(mask, H, 0.0)
    return jnp.sum(H) / (jnp.sum(mask) + 1e-8)


class GraphConv(nnx.Module):
    """图卷积：H' = silu(A_norm @ H @ W)。"""

    def __init__(self, in_dim, out_dim, rngs, dtype):
        self.linear = nnx.Linear(in_dim, out_dim, rngs=rngs, dtype=dtype, kernel_init=kernel_init)

    def __call__(self, A, H):
        return nnx.silu(self.linear(A @ H))


class GraphFrameSelector(nnx.Module):
    """语言引导的 GNN 选帧器：hard（STE top-k 选真实帧） / soft（DiffPool 软原型）。"""

    def __init__(self, config, rngs: nnx.Rngs, dtype, task_dim: int):
        self.cfg = config
        self.dtype = dtype
        pm = config.perceptual_memory

        self.variant = str(pm.get("variant", "soft"))
        assert self.variant in ("hard", "soft"), f"未知 variant: {self.variant}"
        self.use_visual_axis = bool(pm.get("use_visual_axis", True))

        img_dim = config.memory_feature.img.input_dim
        pos_dim = config.memory_feature.pos.input_dim
        hidden = pm.get("hidden_dim", 256)
        n_layers = pm.get("num_layers", 2)
        self.top_k = int(pm.get("top_k", config.budget // config.token_per_image))
        self.mask_threshold = float(pm.get("mask_threshold", 0.0))

        # 护栏权重（仅 soft 变体生效；hard 变体恒为 0）
        self.w_entropy = float(pm.get("entropy_weight", 0.0))
        self.w_balance = float(pm.get("balance_weight", 0.0))
        self.w_spread = float(pm.get("spread_weight", 0.0))

        # 节点特征维度：视觉轴(可选) + 位置轴 + 任务向量（顺序必须与 __call__ 一致）
        dims = []
        if self.use_visual_axis:
            dims.append(img_dim)
        dims.append(pos_dim)
        dims.append(task_dim)
        self.node_proj = nnx.Linear(sum(dims), hidden, rngs=rngs, dtype=dtype, kernel_init=kernel_init)

        # 用 dict（字符串 key）而非 list，否则 nnx 参数树里的整数下标会让 weight_loader 的 flatten_dict 崩溃
        self.gnn = {f"layer_{i}": GraphConv(hidden, hidden, rngs, dtype) for i in range(n_layers)}

        if self.variant == "hard":
            # 打分头：frame → 关键帧 logit（语言引导）
            self.score_head = nnx.Linear(hidden + task_dim, 1, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        else:
            # 软分配头：frame → 原型（语言引导）
            self.assign_head = nnx.Linear(
                hidden + task_dim, self.top_k, rngs=rngs, dtype=dtype, kernel_init=kernel_init
            )

    def _encode_nodes(self, candidate_image_emb, candidate_pos_emb, task_vec):
        """节点特征 → GNN 消息传递。返回 (b, w, hidden)。"""
        b, w = candidate_image_emb.shape[0], candidate_image_emb.shape[1]
        feats = []
        if self.use_visual_axis:
            feats.append(candidate_image_emb.mean(axis=2))          # (b, w, img_dim)
        feats.append(candidate_pos_emb.mean(axis=2))                # (b, w, pos_dim)
        feats.append(jnp.broadcast_to(task_vec[:, None, :], (b, w, task_vec.shape[-1])))
        h = self.node_proj(jnp.concatenate(feats, axis=-1))
        A = _temporal_adjacency(w)
        for gconv in self.gnn.values():
            h = gconv(A, h)
        return h

    def _flatten(self, sel_image, sel_pos, sel_state, slot_valid, p):
        """(b,k,p,d) → (b, k*p, d) 展平 + 掩码重复到 token 级。"""
        k = sel_image.shape[1]
        sel_image_flat = einops.rearrange(sel_image, "b k p d -> b (k p) d")
        sel_pos_flat = einops.rearrange(sel_pos, "b k p d -> b (k p) d")
        sel_state_flat = einops.repeat(sel_state, "b k d -> b (k p) d", p=p)
        sel_mask_flat = einops.repeat(slot_valid, "b k -> b (k p)", p=p)
        return sel_image_flat, sel_pos_flat, sel_state_flat, sel_mask_flat

    def __call__(
        self,
        candidate_image_emb: at.Float[at.Array, "b w p d1"],
        candidate_mask: at.Bool[at.Array, "b w"],
        candidate_pos_emb: at.Float[at.Array, "b w p d2"],
        candidate_state_emb: at.Float[at.Array, "b w d3"],
        task_vec: at.Float[at.Array, "b d"],
    ):
        b, w, p, d1 = candidate_image_emb.shape
        k = self.top_k
        assert k <= w, f"top_k({k}) 必须 <= 候选窗口({w})"

        task = jnp.broadcast_to(task_vec[:, None, :], (b, w, task_vec.shape[-1]))
        h = self._encode_nodes(candidate_image_emb, candidate_pos_emb, task_vec)
        head_in = jnp.concatenate([h, task], axis=-1)               # (b, w, hidden + task_dim)

        if self.variant == "hard":
            out = self._forward_hard(
                head_in, candidate_image_emb, candidate_pos_emb, candidate_state_emb, candidate_mask, p
            )
        else:
            out = self._forward_soft(
                head_in, candidate_image_emb, candidate_pos_emb, candidate_state_emb, candidate_mask, p
            )
        # dtype 对齐下游约定（img 随候选，pos/state 随候选），避免 one_hot/softmax 等算子
        # 把整条链上采成 fp32（历史上 buffer 的 _nodes_to_memory 就为此显式指定过 dtype）。
        sel_img, sel_pos, sel_state, sel_mask, stats = out
        return (
            sel_img.astype(candidate_image_emb.dtype),
            sel_pos.astype(candidate_pos_emb.dtype),
            sel_state.astype(candidate_state_emb.dtype),
            sel_mask,
            stats,
        )

    # ---------------- v4-A：STE 硬 top-k，选真实帧 ----------------
    def _forward_hard(self, head_in, cand_img, cand_pos, cand_state, cand_mask, p):
        b, w = cand_mask.shape
        k = self.top_k

        logits = self.score_head(head_in)[..., 0]                   # (b, w)
        logits = jnp.where(cand_mask, logits, -1e9)                 # 屏蔽 padding

        # 硬 top-k（确定性，无 RNG；前向给出精确的真实帧选择）
        _, top_idx = jax.lax.top_k(logits, k)                       # (b, k)，降序
        m_hard = jnp.zeros((b, w), dtype=logits.dtype)
        m_hard = m_hard.at[jnp.arange(b)[:, None], top_idx].set(1.0)  # (b, w)，最多 k 个 1
        # 有效帧不足 k 时 top_k 会落到 padding 上 → 与 mask 相与，保证只标记真实存在的帧
        m_hard = m_hard * cand_mask.astype(logits.dtype)

        # STE：前向 = 硬 mask，反向 = softmax 权重（让打分头可训练）
        m_soft = jax.nn.softmax(logits, axis=-1)
        m_soft = jnp.where(cand_mask, m_soft, 0.0)
        m = m_soft + jax.lax.stop_gradient(m_hard - m_soft)          # (b, w)

        # 按时间顺序给槽位编号（候选下标序 = 时间序），构造 (b, k, w) 的选择矩阵
        rank = jnp.cumsum(m_hard, axis=-1) - 1.0                     # (b, w)：选中的为 0..k-1
        is_sel = m_hard > 0.5
        slot = jax.nn.one_hot(rank.astype(jnp.int32), k) * is_sel[..., None]   # (b, w, k)
        S = jnp.einsum("bwk,bw->bkw", slot, m)                       # (b, k, w)

        # 前向 S 是精确 one-hot → 等价于 gather 真实帧（信息零损失）
        sel_image = jnp.einsum("bkw,bwpd->bkpd", S, cand_img)
        sel_pos = jnp.einsum("bkw,bwpd->bkpd", S, cand_pos)
        sel_state = jnp.einsum("bkw,bwd->bkd", S, cand_state)

        n_sel = m_hard.sum(axis=-1)                                   # (b,)
        slot_valid = jnp.arange(k)[None, :] < n_sel[:, None]          # (b, k)

        return (*self._flatten(sel_image, sel_pos, sel_state, slot_valid, p), {"aux_loss": jnp.asarray(0.0)})

    # ---------------- v4-B：DiffPool 软分配 + 三项护栏 ----------------
    def _forward_soft(self, head_in, cand_img, cand_pos, cand_state, cand_mask, p):
        b, w = cand_mask.shape
        k = self.top_k

        S = jax.nn.softmax(self.assign_head(head_in), axis=-1)        # (b, w, k)
        S = jnp.where(cand_mask[..., None], S, 0.0)                   # 屏蔽 padding

        # 修复①：按列 mass 归一 → 原型是**加权均值**（原 v3 是加权和，不同原型尺度不一致）
        col_mass = S.sum(axis=1) + 1e-8                               # (b, k)
        S_norm = S / col_mass[:, None, :]                             # (b, w, k)

        sel_image = jnp.einsum("bwk,bwpd->bkpd", S_norm, cand_img)    # (b, k, p, d1)
        sel_pos = jnp.einsum("bwk,bwpd->bkpd", S_norm, cand_pos)
        sel_state = jnp.einsum("bwk,bwd->bkd", S_norm, cand_state)

        # 修复②：死/低质量原型 → 真 mask（原 v3 无条件全 1，零向量当有效记忆）。
        # 用**相对**阈值（相对平均质量）而非绝对阈值：episode 起始帧数 n < k 时，
        # 绝对阈值会把所有原型都屏蔽掉，导致下游 attn softmax 全 -inf → NaN。
        n_valid = cand_mask.sum(axis=-1, keepdims=True).astype(col_mass.dtype)  # (b, 1)
        slot_valid = col_mass > self.mask_threshold * (n_valid / k)             # (b, k)

        # 护栏：entropy① + balance② + temporal-spread③
        aux = (
            self.w_entropy * _entropy_loss(S, cand_mask)
            + self.w_balance * _balance_loss(S)
            + self.w_spread * _temporal_spread_loss(S)
        )
        return (*self._flatten(sel_image, sel_pos, sel_state, slot_valid, p), {"aux_loss": aux})
