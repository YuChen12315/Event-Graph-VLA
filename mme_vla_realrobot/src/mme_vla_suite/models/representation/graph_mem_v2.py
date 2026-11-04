"""GraphMem v2 —— 基于关系图的在线关键帧记忆。

唯一实现依据：`Papers/GraphMem_Implementation_Spec_v2.docx`（v2.0，2026-09-20）。
本文件与同目录的 `graph_mem.py`（TCGM 的 graph_dpp 选择器）**没有关系**，是两条独立路线。

规格 §20 要求的五个模块，在本文件里的落点：

  FrameEncoder   §6        visual/state 投影（Linear→GELU→Linear，256 维）+ 时间编码 + 来源编码
  GraphBuilder   §8/§9     节点列表 → 3 类邻接掩码 [3,N,N] + 6 维边属性 [N,N,6]
  RelationalGNN  §11/§12   2 层 × 3 关系分支 × 4 头 × 64 维，任务条件 + 边条件
  KeepScorer     §13       候选 hidden ⊕ instruction（512）→ 128 → 1，不做 sigmoid
  MemoryBank     §3/§14    8+24 容量状态机、去重、最低分淘汰、§14.1 低置信退回、§15 导出
  GraphMem       ——        上面四个 nnx 模块的容器，给 MemoryBank 提供一个纯函数式的评分回调

两条与代码库其它记忆模块不同的硬约定（都来自 v2 的实测，不是风格选择）：

1. **导出槽位按时间升序，槽位 31 固定为最新帧，不足时在"最老端"补零**（§15 / C1）。
   现有 `mem_buffer._right_pad_candidate` 是**右**填充（补在最新端），这里必须相反。
   依据：E1 冻结下游按槽位序号经 RoPE 读时间，只翻转帧序的代价（Δ=+0.0165）
   大于换掉整批帧（Δ=+0.0091）。槽位序号必须与"距今多久"单调对应。

2. **退化为等距帧集是合法且必须实现的路径**（§14.1 / C2）。
   喂错记忆的代价（+0.0286）大于把内容清零（+0.0175），也大于完全不给内容（+0.0144）。
   所以评分头不可靠时硬选 24 帧不是中性选择，而是有明确负作用的选择。

适用范围（§14.1 明确限定）：上述负向代价排序只在 `integration_type = modulation` 上验证过。
换到 context / expert 接口后需重新标定再启用退回路径。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.shared.array_typing as at
from mme_vla_suite.models.representation.utils import kernel_init

# ---------------------------------------------------------------- 固定容量（§1）
RECENT_CAP = 8          # 近期缓存帧数
LONG_CAP = 24           # 长期关键帧库帧数
TOTAL_CAP = RECENT_CAP + LONG_CAP          # 32：导出槽位数
N_CAND = LONG_CAP + 1                      # 25：溢出时的长期候选数
MAX_NODES = N_CAND + RECENT_CAP            # 33：图的临时最大规模

# 边属性 6 维的顺序（§9）
E_DT, E_VIS_SIM, E_VIS_CHG, E_ST_CHG, E_ST_CMP, E_SAME_SRC = range(6)
EDGE_DIM = 6

N_RELATIONS = 3         # 0 自环 / 1 时间 / 2 视觉 kNN（§8）


# ============================================================ §8 / §9 建图
def _even_indices(n: int, k: int) -> list[int]:
    """在 [0, n-1] 上均匀取 k 个下标（n <= k 时全取）。与 data_utils.even_sampling_indices 同口径。"""
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    return np.linspace(0, n - 1, k, dtype=np.int32).tolist()


def build_graph(
    globals_: np.ndarray,        # (N, Dv)  pooled 全局视觉特征
    states: np.ndarray,          # (N, Ds)  归一化本体状态
    state_masks: np.ndarray,     # (N, Ds)  逐维有效性
    timestamps: np.ndarray,      # (N,)     来源内递增（帧索引或秒）
    source_ids: np.ndarray,      # (N,)
    valid: np.ndarray,           # (N,)     padding 有效位
    n_candidates: int,           # 前多少个是长期候选（只它们参与淘汰）
    time_scale: float = 10.0,
    knn_k: int = 4,
    vis_change_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """建图。返回 (mask, edge_attr)：

      mask      [3, N, N] bool —— mask[r, i, j] 为 True 表示关系 r 下节点 i 接收 j 的消息
      edge_attr [N, N, 6] f32 —— 6 维边属性，顺序见 E_* 常量

    节点顺序由调用方保证（§8）：25 个长期候选在前、8 个近期节点在后。
    """
    N = globals_.shape[0]
    src_i = source_ids[:, None]
    src_j = source_ids[None, :]
    same_src = (src_i == src_j) & valid[:, None] & valid[None, :]

    # ---------------- 关系 0：自环（每个有效节点连接自己）
    mask = np.zeros((N_RELATIONS, N, N), dtype=bool)
    mask[0] = np.eye(N, dtype=bool) & valid[:, None] & valid[None, :]

    # ---------------- 关系 1：同来源时间邻居（双向，连"相邻保留节点"而非原视频连续帧）
    # 删除节点后重新连边，必须保留真实时间跨度（§8）——所以只在当前保留集合内找相邻。
    t = np.where(valid, timestamps, np.inf)
    for s in np.unique(source_ids[valid]):
        idx = np.where(valid & (source_ids == s))[0]
        if idx.size == 0:
            continue
        idx = idx[np.argsort(t[idx], kind="stable")]
        prev, nxt = idx[:-1], idx[1:]
        mask[1, prev, nxt] = True
        mask[1, nxt, prev] = True

    # ---------------- 关系 2：视觉 kNN（有向，不要求互相选择；排除自环与 padding）
    g = np.where(valid[:, None], globals_, 0.0).astype(np.float64)
    norm = np.linalg.norm(g, axis=-1, keepdims=True)
    gn = g / np.maximum(norm, 1e-8)
    sim_raw = gn @ gn.T                               # (N, N) 余弦相似度（边属性用这个）
    # kNN 的选择用屏蔽版：排除自环与 padding。**不能**把 -inf 泄进边属性（②要用真实余弦）
    sim = np.where(mask[0] | ~valid[None, :], -np.inf, sim_raw)
    k = min(knn_k, max(N - 1, 0))
    if k > 0:
        nbr = np.argsort(-sim, axis=-1)[:, :k]        # 每个节点自己的 top-k
        for i in range(N):
            if not valid[i]:
                continue
            # 近邻不足 k 个时取实际数量（§8）：被 -inf 挡掉的都是自环/padding
            keep = [j for j in nbr[i] if np.isfinite(sim[i, j])]
            mask[2, i, keep] = True

    mask &= valid[:, None] & valid[None, :]           # 任何关系都不许把 padding 当邻居

    # ---------------- 6 维边属性
    edge = np.zeros((N, N, EDGE_DIM), dtype=np.float32)

    # ① 带符号时间差：(time_i - time_j) / time_scale，截断 -10..10；跨来源置零
    dt = (timestamps[:, None] - timestamps[None, :]) / float(time_scale)
    edge[..., E_DT] = np.where(same_src, np.clip(dt, -10.0, 10.0), 0.0)

    # ② 视觉相似度：归一化全局特征的余弦（用未屏蔽的 sim_raw；自环处也如实给出余弦）
    edge[..., E_VIS_SIM] = sim_raw

    # ③ 视觉变化：两特征差值的 RMS，按训练统计定标
    diff = g[:, None, :] - g[None, :, :]                       # (N, N, Dv)
    edge[..., E_VIS_CHG] = (
        np.sqrt(np.mean(diff * diff, axis=-1)) / max(float(vis_change_scale), 1e-8)
    )

    # ④ 状态变化：只在双方共同有效的维度上算 RMS；跨来源或无共同维度置零
    st_i = np.where(state_masks, states, 0.0)
    common = state_masks[:, None, :] & state_masks[None, :, :]   # (N, N, Ds)
    n_common = common.sum(axis=-1)
    sd = st_i[:, None, :] - st_i[None, :, :]
    rms = np.sqrt(np.sum(np.where(common, sd * sd, 0.0), axis=-1)
                  / np.maximum(n_common, 1))
    edge[..., E_ST_CHG] = np.where(same_src & (n_common > 0), rms, 0.0)

    # ⑤ 状态可比标志：同来源且至少一个共同有效维度（区分"没有变化"与"无法比较"）
    edge[..., E_ST_CMP] = (same_src & (n_common > 0)).astype(np.float32)

    # ⑥ 同来源标志
    edge[..., E_SAME_SRC] = same_src.astype(np.float32)

    # 只有有效节点对上的边属性有意义；其余清零，避免污染
    edge *= (valid[:, None] & valid[None, :])[..., None]
    return mask, edge


# ============================================================ §6 节点特征
def sinusoidal_encoding(x: np.ndarray, dim: int) -> np.ndarray:
    """标准正弦编码。输入标量序列 (N,)，输出 (N, dim)。dim 必须是偶数。"""
    assert dim % 2 == 0, f"sinusoidal 维度必须为偶数，收到 {dim}"
    half = dim // 2
    freqs = np.exp(-math.log(10000.0) * np.arange(half) / half)     # (half,)
    ang = np.asarray(x, dtype=np.float64)[:, None] * freqs[None, :]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=-1).astype(np.float32)


class MLP(nnx.Module):
    """Linear → GELU → Linear（§6 指定的投影形状）。"""

    def __init__(self, in_dim, out_dim, rngs, dtype):
        self.l1 = nnx.Linear(in_dim, out_dim, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.l2 = nnx.Linear(out_dim, out_dim, rngs=rngs, dtype=dtype, kernel_init=kernel_init)

    def __call__(self, x):
        return self.l2(nnx.gelu(self.l1(x)))


class FrameEncoder(nnx.Module):
    """§6：x = LayerNorm(visual + state + time + source_embedding)，padding 节点清零。

    规格 §20 的 FrameEncoder 指"冻结视觉编码器 → 基础节点"。在本仓库里视觉编码是策略自己的
    SigLIP（上游已完成），本模块只承担 §6 剩下的三路投影 + 归一化。
    """

    def __init__(self, config, rngs, dtype):
        gm = config.perceptual_memory
        hidden = int(gm.get("hidden_dim", 256))
        vis_dim = int(config.memory_feature.img.input_dim)
        state_dim = int(config.memory_feature.state.input_dim)
        self.num_sources = int(gm.get("num_sources", 1))
        self.visual_proj = MLP(vis_dim, hidden, rngs, dtype)
        # proprio 与 state_mask 拼接后投影（§6）
        self.state_proj = MLP(2 * state_dim, hidden, rngs, dtype)
        # 来源编码：对 one-hot 做线性投影（用 Linear 而非 Embed，和本仓库其它模块一致）
        self.source_proj = nnx.Linear(self.num_sources, hidden, rngs=rngs, dtype=dtype,
                                      kernel_init=kernel_init)
        self.norm = nnx.LayerNorm(hidden, rngs=rngs, dtype=dtype)

    def __call__(self, globals_, states, state_masks, time_enc, source_ids, valid):
        vis = self.visual_proj(globals_)                                  # (N, H)
        st = self.state_proj(jnp.concatenate([states, state_masks], axis=-1))
        onehot = jax.nn.one_hot(source_ids, self.num_sources, dtype=vis.dtype)
        src = self.source_proj(onehot)
        x = self.norm(vis + st + time_enc + src)
        return x * valid[:, None]


# ============================================================ §11 / §12 GNN
def _masked_softmax(logits, mask):
    """safe_masked_softmax：无邻居行返回**全零**，不产生 NaN（§12）。

    注意不能用 -inf 再乘 0：fp16/bf16 下 exp(-inf)=0 没问题，但后续 0/0 会出 NaN。
    这里用 -1e30 做加性屏蔽，并在分母上做保护，无邻居行显式置零。
    """
    neg = jnp.asarray(-1e30, dtype=logits.dtype)
    logits = jnp.where(mask, logits, neg)
    m = jax.lax.stop_gradient(jnp.max(logits, axis=-1, keepdims=True))
    m = jnp.where(jnp.isfinite(m), m, 0.0)                 # 整行都被屏蔽时 max = -1e30 仍是有限值
    w = jnp.where(mask, jnp.exp(logits - m), 0.0)
    s = jnp.sum(w, axis=-1, keepdims=True)
    return jnp.where(s > 0, w / jnp.maximum(s, 1e-12), 0.0)


class EdgeBias(nnx.Module):
    """§11 边条件：6 → 32 → heads 的 MLP，产出逐头 attention bias。"""

    def __init__(self, edge_dim, hidden, heads, rngs, dtype):
        self.l1 = nnx.Linear(edge_dim, hidden, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.l2 = nnx.Linear(hidden, heads, rngs=rngs, dtype=dtype, kernel_init=kernel_init)

    def __call__(self, edge_attr):
        return self.l2(nnx.gelu(self.l1(edge_attr)))


class RelationalLayer(nnx.Module):
    """§12 的单层：3 个关系分支各自 Q/K/V，消息按关系拼接后投影回 256。"""

    def __init__(self, hidden, heads, task_dim, edge_dim, rngs, dtype):
        assert hidden % heads == 0
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        # 用字符串 key 而非整数下标：nnx 参数树里的整数 key 会让 weight_loader 的 flatten_dict 崩溃
        self.wq = {f"r{r}": nnx.Linear(hidden, hidden, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
                   for r in range(N_RELATIONS)}
        self.wk = {f"r{r}": nnx.Linear(hidden, hidden, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
                   for r in range(N_RELATIONS)}
        self.wv = {f"r{r}": nnx.Linear(hidden, hidden, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
                   for r in range(N_RELATIONS)}
        # 任务条件：每层把任务向量投影后加到各关系的 Query（§11）
        self.w_task = {f"r{r}": nnx.Linear(task_dim, hidden, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
                       for r in range(N_RELATIONS)}
        self.edge_mlp = {f"r{r}": EdgeBias(edge_dim, 32, heads, rngs, dtype)
                         for r in range(N_RELATIONS)}
        self.w_out = nnx.Linear(hidden * N_RELATIONS, hidden, rngs=rngs, dtype=dtype,
                                kernel_init=kernel_init)
        self.norm1 = nnx.LayerNorm(hidden, rngs=rngs, dtype=dtype)
        self.ffn_in = nnx.Linear(hidden, hidden * 2, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.ffn_out = nnx.Linear(hidden * 2, hidden, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.norm2 = nnx.LayerNorm(hidden, rngs=rngs, dtype=dtype)

    def _split(self, x):
        return einops.rearrange(x, "n (h d) -> h n d", h=self.heads)

    def __call__(self, h, instruction, masks, edge_attr, valid):
        # logits 与 softmax 用 float32 算，再转回消息 dtype（§12 的混合精度建议）
        logits_dtype = jnp.float32
        instr = instruction[None, :]                     # (1, task_dim) 广播到所有节点
        messages = []
        for r in range(N_RELATIONS):
            key = f"r{r}"
            # Q/K/V: [heads, N, head_dim]（§12 伪代码的形状约定）
            q = self._split(self.wq[key](h) + self.w_task[key](instr)).astype(logits_dtype)
            k = self._split(self.wk[key](h)).astype(logits_dtype)
            v = self._split(self.wv[key](h))

            logits = jnp.einsum("hnd,hmd->hnm", q, k) / math.sqrt(self.head_dim)
            # edge_mlp: (N,N,heads) → (heads,N,N)
            bias = einops.rearrange(self.edge_mlp[key](edge_attr), "n m h -> h n m")
            logits = logits + bias.astype(logits_dtype)

            alpha = _masked_softmax(logits, masks[r])     # (heads, N, N)
            messages.append(einops.rearrange(
                jnp.einsum("hnm,hmd->hnd", alpha.astype(v.dtype), v), "h n d -> n (h d)"))

        h = h + self.w_out(jnp.concatenate(messages, axis=-1))
        h = self.norm1(h)
        h = h + self.ffn_out(nnx.gelu(self.ffn_in(h)))
        h = self.norm2(h)
        return h * valid[:, None]                          # padding 输出强制清零（§11）


class RelationalGNN(nnx.Module):
    """§11：2 层，两层参数不共享。"""

    def __init__(self, config, rngs, dtype):
        gm = config.perceptual_memory
        hidden = int(gm.get("hidden_dim", 256))
        heads = int(gm.get("num_heads", 4))
        layers = int(gm.get("num_layers", 2))
        self.layers = {
            f"layer_{i}": RelationalLayer(hidden, heads, hidden, EDGE_DIM, rngs, dtype)
            for i in range(layers)
        }
        self.dropout = float(gm.get("dropout", 0.0))
        assert self.dropout == 0.0, "§11 固定 dropout=0"

    def __call__(self, x, instruction, masks, edge_attr, valid):
        h = x
        for layer in self.layers.values():
            h = layer(h, instruction, masks, edge_attr, valid)
        return h


class KeepScorer(nnx.Module):
    """§13：concat(hidden, instruction) → 128 → 1，输出未归一化标量（不做 sigmoid）。"""

    def __init__(self, config, rngs, dtype):
        hidden = int(config.perceptual_memory.get("hidden_dim", 256))
        self.l1 = nnx.Linear(hidden * 2, 128, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.l2 = nnx.Linear(128, 1, rngs=rngs, dtype=dtype, kernel_init=kernel_init)

    def __call__(self, h, instruction):
        b = jnp.broadcast_to(instruction[None, :], (h.shape[0], instruction.shape[-1]))
        return self.l2(nnx.gelu(self.l1(jnp.concatenate([h, b], axis=-1))))[..., 0]


class GraphMem(nnx.Module):
    """§20 的四个 nnx 模块容器。参数名按 §13 建议固定：frame_encoder / gnn / keep_head。"""

    def __init__(self, config, rngs, dtype):
        self.config = config
        self.hidden = int(config.perceptual_memory.get("hidden_dim", 256))
        self.frame_encoder = FrameEncoder(config, rngs, dtype)
        self.gnn = RelationalGNN(config, rngs, dtype)
        self.keep_head = KeepScorer(config, rngs, dtype)

    # ---------------------------------------------------------- 前向（纯函数式）
    @at.typecheck
    def __call__(
        self,
        globals_: at.Float[at.Array, "n dv"],
        states: at.Float[at.Array, "n ds"],
        state_masks: at.Bool[at.Array, "n ds"],
        time_enc: at.Float[at.Array, "n dh"],
        source_ids: at.Int[at.Array, " n"],
        valid: at.Bool[at.Array, " n"],
        instruction: at.Float[at.Array, " dt"],
        masks: at.Bool[at.Array, "r n n"],
        edge_attr: at.Float[at.Array, "n n e"],
    ):
        """返回 (hidden [n, H], scores [n])。按 §13，只有前 n_candidates 个的分数参与淘汰。"""
        x = self.frame_encoder(globals_, states, state_masks, time_enc, source_ids, valid)
        h = self.gnn(x, instruction, masks, edge_attr, valid)
        scores = self.keep_head(h, instruction) * valid
        return h, scores


# ============================================================ §18 排序训练
def ranking_loss(scores, utility, label_mask, margin: float = 0.0):
    """§18 成对排序损失：utility 高者分数必须至少高出 margin。

    utility 由外部效用教师给出并**停止梯度**（§17）；loss 同时更新 keep_head、GNN
    与可训练的节点投影。没有有效样本对时返回 0（而不是 NaN）。

    scores/utility/label_mask: (n_cand,)
    """
    utility = jax.lax.stop_gradient(utility)
    valid = label_mask.astype(bool)
    pairs = valid[:, None] & valid[None, :]
    pairs = pairs & ((utility[:, None] - utility[None, :]) > margin)

    delta = scores[:, None] - scores[None, :]
    terms = jnp.where(pairs, jax.nn.softplus(-delta), 0.0)
    n_pairs = jnp.sum(pairs)
    # 无有效对 → 0，且保持对 scores 的可微连接（乘 0 而不是返回常量）
    return jnp.sum(terms) / jnp.maximum(n_pairs, 1) + jnp.sum(scores) * 0.0


# ============================================================ §3 / §14 状态机
@dataclass
class FrameNode:
    """一个帧节点。基础特征一旦写入不再被 GNN 输出覆盖（§3 步骤 6）。"""

    frame_id: int                      # 去重键之一（§3 步骤 1）
    source_id: int = 0
    timestamp: float = 0.0
    global_feat: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    state: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    state_mask: np.ndarray = field(default_factory=lambda: np.zeros(1, np.float32))
    arrival: int = 0                   # 到达顺序，仅用于平局与诊断（§15）
    payload: dict = field(default_factory=dict)   # image/pos/state 原始张量，导出时用


class MemoryBank:
    """§3 状态机 + §14 淘汰 + §14.1 退回 + §15 导出。

    评分回调 `scorer(nodes, valid, n_cand, instruction) -> np.ndarray[n_cand]`：
      - 传 None 表示"评分头不可用"，此时**恒定**走 §14.1 退回路径；
      - 这也正是训练出评分头之前的合法运行模式：退回 = 等距帧集 = §19 的“均匀在线保留”对照。
    """

    def __init__(
        self,
        scorer=None,
        recent_cap: int = RECENT_CAP,
        long_cap: int = LONG_CAP,
        time_scale: float = 10.0,
        knn_k: int = 4,
        vis_change_scale: float = 1.0,
        confidence_threshold: float = 0.02,
        min_score_std: float = 0.0,
        window_span: int = 5,
        fallback: bool = True,
        fallback_only: bool = False,
    ):
        self.scorer = scorer
        self.recent_cap = recent_cap
        self.long_cap = long_cap
        self.time_scale = time_scale
        self.knn_k = knn_k
        self.vis_change_scale = vis_change_scale
        self.confidence_threshold = confidence_threshold
        self.min_score_std = min_score_std
        self.window_span = window_span
        self.fallback_enabled = fallback
        # fallback_only：恒定走退回路径。这是**未训练评分头时的合法运行模式**，
        # 而不是降级——它精确等价于 §19 要求的"均匀在线保留"对照臂。
        self.fallback_only = fallback_only
        self.reset()

    # ---------------------------------------------------------------- 生命周期
    def reset(self):
        self.recent: list[FrameNode] = []
        self.long_term: list[FrameNode] = []
        # 已离开近期队列的全部帧（§14.1 的等距退回要在这个全集上取样，而不只是当前长期库）。
        # 只存对象引用，不复制特征。
        self.left_behind: list[FrameNode] = []
        self.n_arrival = 0
        self.n_overflow = 0        # 近期队列溢出次数（= 离开近期的帧数）
        self.n_evictions = 0       # 真正触发淘汰评分的次数（§21 要求 300 帧 → 268）
        self.eviction_log: list[dict] = []
        self.last_seen: dict[int, int] = {}       # source_id → 最后已接收 frame_id（§7）
        self.fallback_active = False
        self.export_cache: dict | None = None

    # ---------------------------------------------------------------- §3 逐帧
    def observe(self, node: FrameNode, instruction: np.ndarray | None = None,
                step: int | None = None) -> dict:
        """帧到达。返回本轮更新后的 export。重复输入直接返回原状态（§3 步骤 1）。

        `step` 是**决策时刻**（这一帧进入流的时间步），只用于日志：淘汰发生时被溢出的
        是 `recent` 队首，即 `recent_cap` 步之前到达的那一帧，所以日志里的 `frame_id`
        与决策时刻相差 recent_cap。§17 的契约要 `query_step`，必须把它单独记下来，
        否则按 frame_id 当决策时刻会看错时间窗（实测候选与导出集的交集只有 1/24）。
        """
        # 步骤 1：校验 + 去重 + 乱序拒绝
        if self.duplicate(node):
            return self.export()
        last = self.last_seen.get(node.source_id)
        if last is not None and node.frame_id <= last:
            raise ValueError(
                f"同来源乱序输入被拒绝：source={node.source_id} "
                f"收到 frame_id={node.frame_id} <= 已接收 {last}"
            )

        node.arrival = self.n_arrival
        self.n_arrival += 1
        self.last_seen[node.source_id] = node.frame_id

        # 步骤 3：追加近期队列，超容量则队首进入淘汰流程
        self.recent.append(node)
        if len(self.recent) > self.recent_cap:
            candidate = self.recent.pop(0)
            self.n_overflow += 1
            self.left_behind.append(candidate)
            # 步骤 4：长期库未满则直接加入（**不评分、不淘汰**，§4）
            if len(self.long_term) < self.long_cap:
                self.long_term.append(candidate)
            else:
                self.n_evictions += 1
                self._evict(candidate, instruction, step)

        self.export_cache = self.export()
        return self.export_cache

    def duplicate(self, node: FrameNode) -> bool:
        """§7：每来源只记**最后已接收帧号**即可处理单调输入，不保留全量去重集合。

        注意：这里**不能**改成扫描 recent+long_term 里有没有同号帧。那样一来，
        "乱序的老帧"会因为它的帧号还留在库里而被误判成"重复输入"直接静默返回，
        乱序拒绝（§3 步骤 1 / §21）就永远触发不了。判定必须只看单调时钟。
        """
        return self.last_seen.get(node.source_id) == node.frame_id

    # ---------------------------------------------------------------- §14 淘汰
    def _evict(self, candidate: FrameNode, instruction, step: int | None = None):
        """C = L + [candidate]（25 个）；G = C + R（33 个）。只有 C 参与淘汰。"""
        candidates = self.long_term + [candidate]
        nodes = candidates + list(self.recent)
        n_cand = len(candidates)

        scores = self._score(nodes, n_cand, instruction)

        if scores is None:
            # §14.1 退回：不淘汰，长期库改由等距帧集构成
            self.fallback_active = True
            self._apply_fallback(candidate)
            self.eviction_log.append({
                "arrival": candidate.arrival, "frame_id": candidate.frame_id,
                "step": step,
                # §17 的效用教师契约要求产出 candidate_ids；退回臂同样要留底，
                # 否则对"均匀在线保留"这一对照臂无法测候选效用。
                "candidate_ids": [n.frame_id for n in candidates],
                "score": None, "scores": None, "dropped": None, "fallback": True,
            })
            return

        self.fallback_active = False
        # §14：分数相同优先淘汰到达顺序较早的节点 → 按 (score, arrival) 字典序取最小
        cand_arrival = np.array([n.arrival for n in candidates])
        order = np.lexsort((cand_arrival, scores))          # 主键 score，次键 arrival
        drop = int(order[0])
        log = {
            "arrival": candidate.arrival, "frame_id": candidate.frame_id,
            "step": step,
            "candidate_ids": [n.frame_id for n in candidates],
            "scores": scores.tolist(), "dropped": int(candidates[drop].frame_id),
            "fallback": False,
        }
        self.long_term = [n for i, n in enumerate(candidates) if i != drop]
        self.eviction_log.append(log)

    def _score(self, nodes: list[FrameNode], n_cand: int, instruction) -> np.ndarray | None:
        """调评分回调并做 §14.1 的置信度判定。返回 None 表示应当退回。"""
        if self.scorer is None or self.fallback_only:
            return None
        raw = np.asarray(self.scorer(nodes, n_cand, instruction), dtype=np.float64)
        if not np.all(np.isfinite(raw)):
            return None
        scores = raw[:n_cand]

        # 判据 A：淘汰决策的置信度 —— 最低分与次低分的归一化间距。
        # 间距小 = 淘汰谁基本是抛硬币 = 不该硬选（§14.1 第一种触发条件）。
        # ⚠️ 这个量是**尺度不变**的（除以了 std），所以它衡量的是"决策相对分布是否可分"，
        #    而不是"分数幅度是否正常"——后者由判据 A' 单独负责。两者都需要在训练轨迹上
        #    标定（§14.1），当前默认值是按随机初始化评分头的实测分布定的占位值。
        if n_cand >= 2:
            srt = np.sort(scores)
            std = float(np.std(scores))
            margin = (srt[1] - srt[0]) / (std + 1e-8)
            if margin < self.confidence_threshold:
                return None
            # 判据 A'：分数幅度下界。未训练／加载失败的评分头常表现为输出幅度退化，
            # 尺度不变的判据 A 抓不到这种情况（实测：N(0,1e-6) 与 N(0,1) 的 margin 分布
            # 完全一致）。阈值默认 0 = 关闭，须在训练轨迹上标定后再开。
            if self.min_score_std > 0.0 and std < self.min_score_std:
                return None

        # 判据 B：超过半数候选来自同一 arrival 窗口 → 评分没有把候选区分开
        arr = np.sort(np.array([n.arrival for n in nodes[:n_cand]]))
        if arr.size > self.window_span:
            counts = [np.sum((arr >= a) & (arr < a + self.window_span)) for a in arr]
            if max(counts) * 2 > arr.size:
                return None

        return scores

    def _apply_fallback(self, candidate: FrameNode):
        """§14.1 退回行为：长期库 = **已离开近期队列的历史**上的等距帧集。

        注意取样全集是 left_behind（全部已离开近期的帧），不是"当前长期库 + candidate"。
        后者会让等距性与实际历史无关：每轮只从 25 个里去掉一个，长期库会逐渐退化成
        一段连续区间（实测 60 帧时长期库变成 0..51、间隔 1~29），既不等距也丢失了
        "均匀覆盖全程"这个对照臂的定义。§19 的"均匀在线保留"对照就是等距帧集。

        不发生淘汰；长期库状态仍按正常路径推进，以便下一轮重新判定。
        """
        pool = self.left_behind
        if len(pool) <= self.long_cap:
            self.long_term = list(pool)
            return
        order = sorted(range(len(pool)), key=lambda i: (pool[i].timestamp, pool[i].arrival))
        picks = _even_indices(len(order), self.long_cap)
        self.long_term = [pool[order[i]] for i in picks]

    # ---------------------------------------------------------------- §15 导出
    def export(self) -> dict:
        """按 timestamp 升序输出；槽位 31 为最新帧，槽位 0 为最老；不足 32 时在**最老端**补零。

        与 mem_buffer._right_pad_candidate(补在最新端) 相反 —— 见文件头约定 1。
        """
        nodes = sorted(self.recent + self.long_term,
                       key=lambda n: (n.timestamp, n.source_id, n.arrival))
        n = len(nodes)
        assert n <= TOTAL_CAP, f"导出容量越界：{n} > {TOTAL_CAP}"
        return {
            "nodes": nodes,                       # 时间升序
            "slot_of_oldest": TOTAL_CAP - n,      # 有效槽位从 [TOTAL_CAP-n, 31]
            "valid_mask": [False] * (TOTAL_CAP - n) + [True] * n,
            "arrival_index": [n_.arrival for n_ in nodes],
            "fallback_active": bool(self.fallback_active),
            "n_overflow": self.n_overflow,
            "n_evictions": self.n_evictions,
        }

    # ---------------------------------------------------------------- §21 自检
    def assert_unique_and_bounded(self):
        ids = [(n.source_id, n.frame_id) for n in self.recent + self.long_term]
        assert len(ids) == len(set(ids)), f"帧库出现重复：{ids}"
        assert len(self.recent) <= self.recent_cap
        assert len(self.long_term) <= self.long_cap
        assert len(self.recent) + len(self.long_term) <= TOTAL_CAP


# ============================================================ 建图（推理/训练唯一入口）
def graph_tensors(nodes: list[FrameNode], n_cand: int, instruction, hidden: int,
                  time_scale: float, knn_k: int, vis_change_scale: float) -> dict:
    """把节点列表落成 GraphMem 前向所需的全部张量（numpy）。

    **§18 的排序训练与 §14 的推理共用这一条入口**，不是两处各写一遍建图。
    理由：排序标签由 §17 的教师在同一批节点上量出（Δloss 是在这个图上算的），
    若训练侧另写一遍建图，节点顺序 / 边属性稍有出入，报出来的排序指标就与部署行为
    脱钩，而且是**静默**脱钩。抽成单一入口后，一致性由构造保证。

    节点顺序由调用方保证（§8）：25 个长期候选在前、8 个近期节点在后。
    """
    N = len(nodes)
    assert N > 0, "空节点列表无法建图"
    globals_ = np.stack([n.global_feat for n in nodes]).astype(np.float32)
    states = np.stack([n.state for n in nodes]).astype(np.float32)
    state_masks = np.stack([n.state_mask for n in nodes]).astype(bool)
    timestamps = np.array([n.timestamp for n in nodes], dtype=np.float64)
    source_ids = np.array([n.source_id for n in nodes], dtype=np.int32)
    valid = np.ones(N, dtype=bool)

    masks, edge_attr = build_graph(
        globals_, states, state_masks, timestamps, source_ids, valid, n_cand,
        time_scale=time_scale, knn_k=knn_k, vis_change_scale=vis_change_scale)

    time_enc = sinusoidal_encoding(timestamps / time_scale, hidden)
    if instruction is None:
        # 离线回放路径（precompute / 教师）不传 instruction，此处统一补零。
        # ⚠️ 已知缺口：于是 keep_head 的 instruction 条件在离线路径上恒为常数，
        #    该分支学不到东西。要修就得同时改 replay_episode → MemoryBank._score →
        #    GraphScorer 三处，把任务指令嵌入穿进去；不影响 §18 的训练/部署一致性
        #    （两边都是零），故先记录不动。
        instruction = np.zeros(hidden, dtype=np.float32)

    return {
        "globals_": globals_, "states": states, "state_masks": state_masks,
        "time_enc": time_enc, "source_ids": source_ids, "valid": valid,
        "instruction": np.asarray(instruction, dtype=np.float32),
        "masks": masks, "edge_attr": edge_attr,
    }


# ============================================================ 评分回调适配器
class GraphScorer:
    """把 GraphMem(nnx) 包成 MemoryBank 能调的 numpy 回调。

    建图在 numpy 侧（可脱离 JAX 单测），GNN 只在评分时跑。
    """

    def __init__(self, model: GraphMem, config, dtype=jnp.float32):
        self.model = model
        self.config = config
        gm = config.perceptual_memory
        self.hidden = int(gm.get("hidden_dim", 256))
        self.time_scale = float(gm.get("time_scale", 10.0))
        self.knn_k = int(gm.get("knn_k", 4))
        self.vis_change_scale = float(gm.get("vis_change_scale", 1.0))
        self.dtype = dtype
        self.last_graph = None

    def __call__(self, nodes: list[FrameNode], n_cand: int, instruction) -> np.ndarray:
        t = graph_tensors(nodes, n_cand, instruction, self.hidden,
                          self.time_scale, self.knn_k, self.vis_change_scale)
        self.last_graph = (t["masks"], t["edge_attr"])

        _, scores = self.model(
            jnp.asarray(t["globals_"]), jnp.asarray(t["states"]),
            jnp.asarray(t["state_masks"]), jnp.asarray(t["time_enc"]),
            jnp.asarray(t["source_ids"]), jnp.asarray(t["valid"]),
            jnp.asarray(t["instruction"]),
            jnp.asarray(t["masks"]), jnp.asarray(t["edge_attr"]))
        return np.asarray(scores, dtype=np.float64)


def build_model(config, rngs, dtype=jnp.float32) -> GraphMem:
    """按 yaml 构造 GraphMem。参数名与 §13 建议一致。"""
    return GraphMem(config, rngs, dtype)


# ============================================================ 帧流回放（离线/训练共用）
def make_node(frame_id, image, pos, state, timestamp=None, source_id=0,
              state_mask=None) -> FrameNode:
    """从一帧的缓存特征构造节点。

    - global_feat = 对局部 token 做**池化**（§5 visual_global：逐相机池化并融合后的全局特征）
    - proprio = state；state_mask 缺省全 1（counting 数据的 8 维本体全部有效）
    - payload 保留 image/pos/state 原始张量，供导出时 gather
    """
    image = np.asarray(image, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    if timestamp is None:
        timestamp = float(frame_id)
    if state_mask is None:
        state_mask = np.ones_like(state, dtype=bool)
    return FrameNode(
        frame_id=int(frame_id), source_id=int(source_id), timestamp=float(timestamp),
        global_feat=image.mean(axis=0).astype(np.float32),
        state=state, state_mask=np.asarray(state_mask, dtype=bool),
        payload={"image": image, "pos": np.asarray(pos, dtype=np.float32), "state": state},
    )


def replay_episode(frames: list[dict], cfg, scorer=None, instruction=None,
                   upto: int | None = None, record_every: int = 1) -> dict:
    """把一条 episode 的帧流按 §3 逐帧喂给 MemoryBank，返回每个评测时刻的导出。

    frames: 时间升序的帧列表，每项含 image/pos/state（frame_id 缺省用下标）。
    返回 {step_idx: export}，供预计算脚本落盘、或训练时直接取某个时刻的记忆。

    注意这是**因果**回放：时刻 s 的导出只依赖 frames[:s+1]（§10）。
    """
    gm = cfg.perceptual_memory
    bank = MemoryBank(
        scorer=scorer,
        recent_cap=int(gm.get("recent_cap", RECENT_CAP)),
        long_cap=int(gm.get("long_cap", LONG_CAP)),
        time_scale=float(gm.get("time_scale", 10.0)),
        knn_k=int(gm.get("knn_k", 4)),
        vis_change_scale=float(gm.get("vis_change_scale", 1.0)),
        confidence_threshold=float(gm.get("confidence_threshold", 0.02)),
        min_score_std=float(gm.get("min_score_std", 0.0)),
        window_span=int(gm.get("window_span", 5)),
        fallback_only=bool(gm.get("fallback_only", False)),
    )
    last = len(frames) if upto is None else min(upto + 1, len(frames))
    out = {}
    for t in range(last):
        f = frames[t]
        node = make_node(
            frame_id=f.get("frame_id", t), image=f["image"], pos=f["pos"], state=f["state"],
            timestamp=f.get("timestamp", f.get("frame_id", t)),
            source_id=f.get("source_id", 0), state_mask=f.get("state_mask"))
        bank.observe(node, instruction, step=t)
        if t % record_every == 0 or t == last - 1:
            out[t] = bank.export()
    out["_bank"] = bank
    return out


def slots_to_memory(export: dict, token_per_image: int, total: int = TOTAL_CAP):
    """把 export 的节点列表落成固定张量，形状与 `mem_buffer._nodes_to_memory` 对齐：

        img   (total*p, d1)  pos (total*p, d2)  state (total*p, d3)  mask (total*p,)

    但**填充方向相反**：这里 padding 落在**最老端**（槽位 0 侧），槽位 31 恒为最新帧。
    `_nodes_to_memory` 是右填充（补在最新端）。见文件头约定 1 —— 下游按槽位序号
    经 RoPE 读时间，方向搞反等于把整条时间轴反了。

    返回 (img, pos, state, mask, export)；帧级 mask 重复到 token 级。
    """
    nodes = export["nodes"]
    n = len(nodes)
    assert n <= total, f"导出帧数 {n} 超过槽位数 {total}"
    # 空导出必须显式拒绝：下面的 else 分支拿不到 token 维度，会静默产出 (total, 1)
    # 的张量，与正常的 (total*P, D) 形状不符，错误会一路飘到下游才炸。
    assert n > 0, "空导出无法推断 token 维度；调用方应先保证至少有一帧"
    if n:
        img = np.stack([nd.payload["image"] for nd in nodes]).astype(np.float32)   # (n, p, d1)
        pos = np.stack([nd.payload["pos"] for nd in nodes]).astype(np.float32)     # (n, p, d2)
        state = np.stack([nd.payload["state"] for nd in nodes]).astype(np.float32)  # (n, d3)
    else:
        img = np.zeros((0, 1, 1), np.float32)
        pos = np.zeros((0, 1, 1), np.float32)
        state = np.zeros((0, 1), np.float32)

    pad = total - n
    p_, d1 = img.shape[1:]
    img = np.concatenate([np.zeros((pad, p_, d1), np.float32), img], axis=0)
    pos = np.concatenate([np.zeros((pad, *pos.shape[1:]), np.float32), pos], axis=0)
    state = np.concatenate([np.zeros((pad, *state.shape[1:]), np.float32), state], axis=0)
    frame_mask = np.concatenate([np.zeros(pad, bool), np.ones(n, bool)])          # (total,)

    return (img.reshape(-1, d1), pos.reshape(-1, pos.shape[-1]),
            np.repeat(state, token_per_image, axis=0),
            np.repeat(frame_mask, token_per_image), export)
