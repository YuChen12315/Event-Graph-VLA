"""Event-Graph Data Plane（v1.3 §7 改造版 v1.4）—— 唯一可学习部分。

    L=8 的 512-d FP16 frozen tokens + state
      → v_pool（指令条件化逐帧池化）
      → Event GRU（吃 8 帧，出 h_event ∈ R^512）
      → 2 层 relation-aware GNN（Temporal Edge + Semantic Edge）
      → 一个 Event Node 一组 token：该节点**真实代表帧**的 2048-d payload 原样铺开
      → max_nodes×16 = 512 个长期 token（空组整组屏蔽）
      + 最近 8 帧 × 16 = 128 个近期 token
      → 640 个 token 交给既有的 mem_encoder

---------------------------------------------------------------------------
两个容易搞错的地方
---------------------------------------------------------------------------
① **长期段 = 一个 Event Node 一组真实画面，不跨事件混合。**
   旧版是「24 个可学 Query 对全部有效节点做 gated read，再用那张 24×N 的权重表
   去加权混合真实 payload」——那是个凸组合，所以**没有任何一帧真实画面能原样
   到达 action expert**，每个 token 都是「全部事件的加权平均」。
   新版直接取消这层混合：实测每个样本平均只有 ~6 个可见事件（Counting 400 条
   episode / 189,035 步：中位数 6、均值 7.14、最大 29），所以「每节点一组」
   根本不需要靠混合来省容量。节点隐状态 h 只作为**零初始化的残差**叠在该节点
   的 16 个 token 上（slot_out），不许替代真实画面。
   §11 禁止用 512-d cache 顶替代表帧的 2048-d payload —— 这条没变。

② **v_pool 不下沉时间信息。** 指令 Query 只对**每一帧内部的 16 个视觉 token**
   做 cross-attention（8 帧共享同一套参数），8 帧之间的先后关系交给 GRU。
   如果让 attention 直接把 8 帧一起池化掉，GRU 就没东西可学了。

（原 ③「gated read 用 sigmoid 不是 softmax」随 24 槽读取器一同删除：那条设计
  针对的是「纯 softmax 会把跨事件的权重强制配平」，而现在**已经没有跨事件
  混合**了，不需要门控。Evidence Mass（帧数 / 证据量）没有丢，见 __call__ ①。）

---------------------------------------------------------------------------
逐帧因果性
---------------------------------------------------------------------------
本模块是逐样本前向的纯函数，不持有任何跨帧状态 —— 图内容是上游
EventGraphController（无参数、stop-gradient）按帧喂进来的。同一个
GraphState 进来，必然得到同一批 token。
"""

from __future__ import annotations

import math

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from mme_vla_suite.models.representation.utils import kernel_init


__all__ = ["EventGraphMemory", "REL_TEMPORAL", "REL_SEMANTIC", "N_RELATIONS"]


# 边类型下标（GNN 每个关系一套 Q/K/V）
REL_TEMPORAL = 0
REL_SEMANTIC = 1
N_RELATIONS = 2

# 边属性维度：[是时间边, 是语义边, 余弦相似度, 源节点的 boundary_strength]
EDGE_DIM = 4

_NEG = -1e30   # 加性屏蔽用的「负无穷」（别用真 -inf：后面 0 * inf 会出 NaN）

# ⚠️ 必须在模块级只求值一次。orthogonal() 每次调用都返回一个**新的闭包对象**，
#    而 kernel_init 是 NodeDef 的静态字段、按对象身份比对：写在 __init__ 里的话，
#    eval_shape 那次建出来的模型和真跑那次建出来的模型会持有两个不同的函数对象，
#    于是 jax.jit(out_shardings=...) 报 "different pytree metadata"。
_ORTHO = nnx.initializers.orthogonal()


def _masked_softmax(logits: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """带屏蔽的 softmax：整行都被屏蔽时返回**全零**，不产生 NaN。

    不能直接 `-inf` 后再乘 mask —— exp(-inf)=0 没问题，但归一化时 0/0 会出 NaN。
    这里先减行最大值（防 exp 溢出），再对分母做保护。
    """
    neg = jnp.asarray(_NEG, dtype=logits.dtype)
    logits = jnp.where(mask, logits, neg)
    m = jax.lax.stop_gradient(jnp.max(logits, axis=-1, keepdims=True))
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    w = jnp.where(mask, jnp.exp(logits - m), 0.0)
    s = jnp.sum(w, axis=-1, keepdims=True)
    return jnp.where(s > 0, w / jnp.maximum(s, 1e-12), 0.0)


class _GRUCell(nnx.Module):
    """标准 GRU 单元（r/z/n 三个门）。

    只用两个 nnx.Linear 手写，不用 flax 的 nnx.GRUCell：后者会在模块上留一个
    `self.rngs` 属性，于是 nnx.state(model) 里多出一个 key<fry> 叶子，训练脚本的
    EMA（`ema_decay * old + ...`）直接 TypeError。GRUCell 的 rngs 只服务于
    initialize_carry，而这里每步都是 jnp.zeros 起步，本来就用不上。
    """

    def __init__(self, d: int, rngs: nnx.Rngs, dtype):
        self.w_i = nnx.Linear(d, 3 * d, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        # 循环核用正交初始化（与 nnx.GRUCell 的默认一致），不加 bias
        self.w_h = nnx.Linear(d, 3 * d, rngs=rngs, dtype=dtype, use_bias=False,
                              kernel_init=_ORTHO)

    def __call__(self, h, x):
        """h, x: [B, N, d] → 新 h [B, N, d]。"""
        xi = einops.rearrange(self.w_i(x), "b n (g d) -> g b n d", g=3)
        hh = einops.rearrange(self.w_h(h), "b n (g d) -> g b n d", g=3)
        r = jax.nn.sigmoid(xi[0] + hh[0])
        z = jax.nn.sigmoid(xi[1] + hh[1])
        n = jnp.tanh(xi[2] + r * hh[2])
        return (1.0 - z) * n + z * h


# ══════════════════════════════════════════════════════════════════════
# ① + ② v_pool 与 Event GRU
# ══════════════════════════════════════════════════════════════════════
class EventEncoder(nnx.Module):
    """一个 Event 的 L=8 帧 → h_event。

    v_pool：同一个指令 Query 分别对每一帧的 P=16 个视觉 token 做 cross-attention，
            8 帧共享参数 ⇒ L 个 task-conditioned frame feature。
            （指令只在 Q 里出现一次，不是逐帧各生成一个 Query。）
    fusion：把该帧的 state 拼上去过一个线性层。
    GRU  ：按时间顺序吃过 L 帧，取末态当 h_event —— 帧间先后关系在这里学。
    """

    def __init__(self, d: int, token_dim: int, state_dim: int, task_dim: int,
                 heads: int, rngs: nnx.Rngs, dtype):
        assert d % heads == 0, f"d={d} 必须被 heads={heads} 整除"
        self.d = d
        self.heads = heads
        self.head_dim = d // heads

        self.q_proj = nnx.Linear(task_dim, d, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        # k 不加 bias：给 K 的每一维加常数，等于给该帧全部 16 个 token 的 logits
        # 同时加同一个数，softmax 下完全抵消 —— 是个永远拿不到梯度、白占参数的死支。
        self.k_proj = nnx.Linear(token_dim, d, rngs=rngs, dtype=dtype,
                                 kernel_init=kernel_init, use_bias=False)
        self.v_proj = nnx.Linear(token_dim, d, rngs=rngs, dtype=dtype, kernel_init=kernel_init)

        # 帧级融合：池化结果 ⊕ 该帧 state
        self.state_proj = nnx.Linear(state_dim, d, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.frame_proj = nnx.Linear(d + d, d, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.frame_norm = nnx.LayerNorm(d, rngs=rngs, dtype=dtype)

        self.gru = _GRUCell(d, rngs, dtype)
        self.out_norm = nnx.LayerNorm(d, rngs=rngs, dtype=dtype)

    def __call__(self, tokens, frame_mask, state, node_mask, task_vec, *, train: bool = True):
        """
        tokens      [B, N, L, P, Dt]  该帧的 16 个 512-d PCA token
        frame_mask  [B, N, L]         哪些采样帧是真实存在的
        state       [B, N, L, S]      逐帧归一化 state
        node_mask   [B, N]            哪些槽位是真实节点
        task_vec    [B, W]            指令向量
        返回 h_event [B, N, d]

        train=True 时计算用 float32：fp16 会把训练早期约 1e-10 的梯度冲成 0。
        train=False（sample_actions）时用 bfloat16，和部署权重的 dtype 一致。
        """
        B, N, L, P, _ = tokens.shape
        d, H, dh = self.d, self.heads, self.head_dim
        compute = jnp.float32 if train else jnp.bfloat16

        # ⚠️ 训练先把 fp16 的输入升到 f32 再算，别让任何内部激活落在 fp16 上。
        #    磁盘上的 512-d 冻结 token 存成 fp16 是为了省带宽，但**计算**留在 fp16
        #    会在反向时把小的梯度冲成精确的 0：fp16 的最小次正规数是 6e-8，初始化
        #    阶段回传到这里的梯度只有 ~1e-10（24 槽读取器接近饱和 + slot 残差零初始化
        #    ⇒ 整支记忆的梯度本来就小）。实测现象是「GNN 的参数（bf16，次正规到 9e-41）
        #    有梯度、encoder 的参数全是精确 0」—— 梯度没断，是被 fp16 吃掉了。
        tokens = tokens.astype(compute)
        state = state.astype(compute)

        # ── v_pool：指令当 Q，每帧的 P 个 token 当 K/V（L 帧共用同一套投影）──
        q = einops.rearrange(self.q_proj(task_vec).astype(compute), "b (h d) -> b h d", h=H)
        k = einops.rearrange(self.k_proj(tokens), "b n l p (h d) -> b h n l p d", h=H)
        v = einops.rearrange(self.v_proj(tokens), "b n l p (h d) -> b h n l p d", h=H)

        # token 级屏蔽：padding 帧的 K 清零（V 靠后面 attn=0 自然不参与）
        # ⚠️ k 已经分头成 (b,h,n,l,p,d)，mask 必须在 h 那一维插 1，写成 "b 1 n l 1 1"。
        #    写成 "b n l 1 1 1" 时，只要 N == L 就能**广播成功**（mask 的 n 位会被当
        #    成 h 位用），于是不报错、只是屏蔽错了帧 —— 冒烟测试里 N=L=8 正好撞上，
        #    换到真实配置 N=96/L=8 才炸出来。
        tok_ok = einops.rearrange(frame_mask, "b n l -> b 1 n l 1 1")
        k = jnp.where(tok_ok, k, 0.0)

        logits = jnp.einsum("bhd,bhnlpd->bhnlp", q, k) / math.sqrt(dh)  # [B,H,N,L,P]
        mask_p = einops.repeat(frame_mask, "b n l -> b h n l p", h=H, p=P)
        attn = _masked_softmax(logits, mask_p)                          # [B,H,N,L,P]
        pooled = jnp.einsum("bhnlp,bhnlpd->bhnld", attn.astype(v.dtype), v)  # [B,H,N,L,dh]
        pooled = einops.rearrange(pooled, "b h n l d -> b n l (h d)")   # [B,N,L,d]

        # ── 帧级融合：拼上该帧 state ──
        st = nnx.silu(self.state_proj(state))                       # [B,N,L,S] -> [B,N,L,d]
        x = self.frame_norm(self.frame_proj(jnp.concatenate([pooled, st], axis=-1)))
        x = x.astype(compute) * frame_mask[..., None]               # padding 帧清零

        # ── Event GRU：按时间顺序吃 L 帧，无效帧不推进状态 ──
        h = jnp.zeros((B, N, d), compute)
        for l in range(L):
            new_h = self.gru(h, x[:, :, l, :])
            keep = frame_mask[:, :, l, None]
            h = jnp.where(keep, new_h, h)

        # 不转回 tokens.dtype（fp16）：见上面的说明，输出这一跳同样会把小梯度冲成 0。
        # 下游（GNN / 槽读取器）本来就在 f32/bf16 上算，留 f32 只是少一次有损往返。
        return self.out_norm(h) * node_mask[..., None]              # padding 节点清零


# ══════════════════════════════════════════════════════════════════════
# ③ relation-aware GNN
# ══════════════════════════════════════════════════════════════════════
class EdgeBias(nnx.Module):
    """边属性 → 逐头 attention bias 的 MLP。"""

    def __init__(self, edge_dim: int, hidden: int, heads: int, rngs: nnx.Rngs, dtype):
        self.l1 = nnx.Linear(edge_dim, hidden, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        # l2 也不加 bias：它的输出是逐头的常数，加在整行所有边上，softmax 下同样抵消。
        self.l2 = nnx.Linear(hidden, heads, rngs=rngs, dtype=dtype,
                             kernel_init=kernel_init, use_bias=False)

    def __call__(self, edge_attr):
        return self.l2(nnx.gelu(self.l1(edge_attr)))


class RelationalLayer(nnx.Module):
    """一层：时间边 / 语义边各一套 Q/K/V，消息拼接后投影回去。"""

    def __init__(self, d: int, heads: int, task_dim: int, rngs: nnx.Rngs, dtype):
        assert d % heads == 0
        self.heads = heads
        self.head_dim = d // heads
        # 用字符串 key 而非整数下标：nnx 参数树里的整数 key 会让 weight_loader 的
        # flatten_dict 崩溃（graph_mem_v2.py 踩过）。
        self.wq = {f"r{r}": nnx.Linear(d, d, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
                   for r in range(N_RELATIONS)}
        # wk 同样不加 bias（理由见 EventEncoder.k_proj）
        self.wk = {f"r{r}": nnx.Linear(d, d, rngs=rngs, dtype=dtype,
                                       kernel_init=kernel_init, use_bias=False)
                   for r in range(N_RELATIONS)}
        self.wv = {f"r{r}": nnx.Linear(d, d, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
                   for r in range(N_RELATIONS)}
        self.w_task = {f"r{r}": nnx.Linear(task_dim, d, rngs=rngs, dtype=dtype,
                                           kernel_init=kernel_init)
                       for r in range(N_RELATIONS)}
        self.edge_mlp = {f"r{r}": EdgeBias(EDGE_DIM, 32, heads, rngs, dtype)
                         for r in range(N_RELATIONS)}
        self.w_out = nnx.Linear(d * N_RELATIONS, d, rngs=rngs, dtype=dtype,
                                kernel_init=kernel_init)
        self.norm1 = nnx.LayerNorm(d, rngs=rngs, dtype=dtype)
        self.ffn_in = nnx.Linear(d, d * 2, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.ffn_out = nnx.Linear(d * 2, d, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.norm2 = nnx.LayerNorm(d, rngs=rngs, dtype=dtype)

    @staticmethod
    def _split(x, heads):
        return einops.rearrange(x, "b n (h d) -> b h n d", h=heads)

    def __call__(self, h, task_vec, adj, edge_attr, node_mask, *, train: bool = True):
        """
        h         [B, N, d]
        task_vec  [B, W]
        adj       [B, R, N, N] bool   每个关系的邻接
        edge_attr [B, N, N, E]
        node_mask [B, N]
        """
        # 训练用 float32，理由同 EventEncoder。推理走 bfloat16。
        compute = jnp.float32 if train else jnp.bfloat16
        H = self.heads
        instr = task_vec[:, None, :]                                  # [B,1,W] 广播到所有节点
        messages = []
        for r in range(N_RELATIONS):
            key = f"r{r}"
            q = self._split(self.wq[key](h) + self.w_task[key](instr), H).astype(compute)
            k = self._split(self.wk[key](h), H).astype(compute)
            v = self._split(self.wv[key](h), H)

            logits = jnp.einsum("bhnd,bhmd->bhnm", q, k) / math.sqrt(self.head_dim)
            bias = einops.rearrange(self.edge_mlp[key](edge_attr), "b n m h -> b h n m")
            logits = logits + bias.astype(compute)

            # adj 是 [B,N,N]，广播到每个头
            alpha = _masked_softmax(logits, adj[:, r][:, None, :, :]) # [B,H,N,N]
            msg = jnp.einsum("bhnm,bhmd->bhnd", alpha.astype(v.dtype), v)
            messages.append(einops.rearrange(msg, "b h n d -> b n (h d)"))

        h = h + self.w_out(jnp.concatenate(messages, axis=-1))
        h = self.norm1(h)
        h = h + self.ffn_out(nnx.gelu(self.ffn_in(h)))
        h = self.norm2(h)
        return h * node_mask[..., None]


# ══════════════════════════════════════════════════════════════════════
# 容器
# ══════════════════════════════════════════════════════════════════════
class EventGraphMemory(nnx.Module):
    """Control Plane 的 GraphState + 真实帧 payload → memory token 的原料。

    输出就是 `mem_encoder`（FeatureEncoder）要的那四个张量，形状与现有
    perceptual 记忆一致 —— 所以下游的调制注入 / Action Expert 一行都不用改。

    注意输出顺序：**长期在前（max_nodes×16），近期在后（recent_window×16）**。
    长期段一个 Event Node 一组 16 个 token，组内就是该节点真实代表帧的 payload，
    空组（可见节点没那么多）整组屏蔽；两个分支时间上互斥，靠 mask 区分。
    """

    def __init__(self, cfg, rngs: nnx.Rngs, dtype, *,
                 payload_dim: int, pos_dim: int, state_dim: int, task_dim: int,
                 heads: int = 8):
        self.cfg = cfg
        d = cfg.event_dim
        self.d = d
        self.n_groups = cfg.max_nodes   # 长期段分组数：一个 Event Node 一组
        self.P = cfg.token_per_image
        self.R = cfg.recent_window

        self.encoder = EventEncoder(d, token_dim=d, state_dim=state_dim,
                                    task_dim=task_dim, heads=heads, rngs=rngs, dtype=dtype)
        # 用 dict + 字符串 key，不要用 list：list 的下标是整数，会成为参数树里的整数
        # key，flax 的 flatten_dict 拿它拼路径时 `sep.join(path)` 直接 TypeError。
        # 仓库里其它 memory 模块也都是这个写法。
        self.gnn = {f"layer_{i}": RelationalLayer(d, heads, task_dim, rngs, dtype)
                    for i in range(cfg.gnn_layers)}

        # 节点标量特征：阶段（0=video 1=exec）+ 边界强度 + 累计抓取数 + 证据量
        # —— 四个都是 Control Plane 的输出（无参数、不吃梯度），这里只学怎么读它们。
        # ⚠️ mass_proj 是从删掉的 SlotReader 里搬过来的，别丢：实测长期记忆**唯一**
        #    明确携带的信息就是事件证据量（log1p(总 event mass) 从 384 个长期 token
        #    可线性解出 R²=0.9987，而 step+task 只有 0.551）。旧版把它加在槽向量上，
        #    现在没有槽了，就按节点加回节点特征。
        self.phase_proj = nnx.Linear(1, d, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.bstr_proj = nnx.Linear(1, d, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.grasp_proj = nnx.Linear(1, d, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.mass_proj = nnx.Linear(1, d, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.node_in = nnx.Linear(d, d, rngs=rngs, dtype=dtype, kernel_init=kernel_init)
        self.node_norm = nnx.LayerNorm(d, rngs=rngs, dtype=dtype)

        # 节点隐状态 → 本节点 16 个 token 的残差（见 __call__ ④）。
        # **零初始化**：训练开头残差恒为 0，模型行为就是纯「真实代表帧原样铺开」，
        # 也就是 §7 描述的那条；GNN 的贡献再从 0 长出来，不会一开始就把画面盖掉。
        # （冻结 ckpt 上实测：去掉这条残差路径的损失代价，比改变长期段铺哪几帧
        #   大 ~100 倍 —— GNN 这条隐状态通路是 action expert 真正在用的东西。）
        self.slot_out = nnx.Linear(d, self.P * payload_dim, rngs=rngs, dtype=dtype,
                                   kernel_init=jax.nn.initializers.zeros,
                                   bias_init=jax.nn.initializers.zeros)
        self.payload_dim = payload_dim

    # ── 建图：时间边 + 语义边 ─────────────────────────────────────────
    def _build_graph(self, h, node_mask, node_bstr):
        f32 = jnp.float32
        B, N, _ = h.shape
        pair_ok = node_mask[:, :, None] & node_mask[:, None, :]        # [B,N,N]

        idx = jnp.arange(N)
        near = jnp.abs(idx[:, None] - idx[None, :]) == 1               # [N,N]
        adj_t = pair_ok & near[None, :, :]

        # 语义边：余弦相似度 top-k（只在有效节点之间，且排除自己）
        # ⚠️ padding 行是精确的 0，而 ||h|| 在 0 处导数发散（h/||h|| = 0/0）——
        #    用 np.linalg.norm 会让整条反向通路变 NaN。这里改成 rsqrt(max(·, ε))：
        #    n2 < ε 时 maximum 的梯度是 0，不会再有 0/0。
        hf = h.astype(f32)
        n2 = jnp.sum(hf * hf, axis=-1, keepdims=True)
        hn = hf * jax.lax.rsqrt(jnp.maximum(n2, 1e-12))
        sim = jnp.einsum("bnd,bmd->bnm", hn, hn)                       # [B,N,N]
        self_sim = jnp.eye(N, dtype=bool)[None, :, :]
        cand = pair_ok & ~self_sim
        neg = jnp.asarray(_NEG, f32)
        sim_masked = jnp.where(cand, sim, neg)
        k = min(self.cfg.semantic_topk, N - 1)
        _, top_idx = jax.lax.top_k(sim_masked, k)                      # [B,N,k]（末维）
        adj_s = (jnp.sum(jax.nn.one_hot(top_idx, N, dtype=f32), axis=-2) > 0)
        adj_s = adj_s & cand

        adj = jnp.stack([adj_t, adj_s], axis=1)                        # [B,R,N,N]

        # 边属性：[时间边, 语义边, 余弦, 源节点的 boundary_strength]
        # 第 4 维按行广播：从节点 n 发往 m 的边，带上 n 那一刀切得多安静（0=不安静）。
        cos_clean = jnp.where(pair_ok, sim, 0.0)
        bstr = jnp.broadcast_to(node_bstr.astype(f32)[:, :, None], (B, N, N))
        edge_attr = jnp.stack([
            adj_t.astype(f32),
            adj_s.astype(f32),
            cos_clean,
            bstr,
        ], axis=-1)
        return adj, edge_attr

    def __call__(self, *, node_tokens, node_frame_mask, node_state, node_mask,
                 node_mass, node_bstr, node_phase, node_grasp,
                 node_payload, node_payload_mask, node_payload_pos, node_payload_state,
                 recent_payload, recent_payload_pos, recent_payload_state, recent_mask,
                 task_vec, train: bool = True):
        """
        长期一侧
          node_tokens        [B,N,L,P,d]   从 512-d FP16 event cache 切出来的 L=8 帧
          node_frame_mask    [B,N,L]
          node_state         [B,N,L,S]
          node_mask          [B,N]
          node_mass          [B,N]         节点帧数（证据量）
          node_bstr          [B,N]         boundary_strength
          node_phase         [B,N]         0=video 1=exec
          node_grasp         [B,N]         截至该节点的累计成功抓取数（Control Plane 数的）
          node_payload       [B,N,P,Dp]    代表帧的真实 2048-d payload
          node_payload_mask  [B,N]
          node_payload_pos   [B,N,P,Dpos]
          node_payload_state [B,N,S]
        近期一侧
          recent_payload     [B,R,P,Dp] / recent_payload_pos [B,R,P,Dpos]
          recent_payload_state [B,R,S] / recent_mask [B,R]
        返回 (img [B,NG*P + R*P, Dp], pos [...], state [...], mask [...])，
        NG = max_nodes（长期段分组数），P = token_per_image，R = recent_window
        """
        B, N = node_mask.shape
        P, R = self.P, self.R

        # ① 节点标量特征拼进去（阶段 + 边界强度 + 累计抓取数 + 证据量），
        #    再走 v_pool + GRU。抓取数的量纲与另外几个不同（0~5 的整数 vs 0/1），
        #    但既然走的是同一个可学 Linear，量纲差异由 kernel 自己吸收，不需要手工缩放。
        #    证据量先 log1p：它是帧数，可以到上百，线性量纲压不住（旧版也是 log1p）。
        #    padding 行的 mass 在 readout 里是精确 0，where 只是防 NaN 的护栏。
        mass = jnp.log1p(jnp.where(node_mask, node_mass, 0.0))
        scal = (self.phase_proj(node_phase[..., None])
                + self.bstr_proj(node_bstr[..., None])
                + self.grasp_proj(node_grasp[..., None])
                + self.mass_proj(mass[..., None]))
        h0 = self.node_norm(self.node_in(scal)) * node_mask[..., None]
        h = self.encoder(
            node_tokens, node_frame_mask, node_state, node_mask, task_vec, train=train
        ) + h0

        # ② relation-aware GNN
        adj, edge_attr = self._build_graph(h, node_mask, node_bstr)
        for layer in self.gnn.values():
            h = layer(h, task_vec, adj, edge_attr, node_mask, train=train)

        # ③ 一个节点一组 token：不再有跨节点的权重混合。
        #    没有 payload 的节点不能进来：它拿不出真实画面证据，铺进去等于凭空造帧。
        #    （它仍然参与 GNN 的 message passing —— 512-d token 是存在的。）
        node_read = node_mask & node_payload_mask

        # ④ 本组 16 个 token = 该节点真实代表帧的 payload + 节点隐状态的残差。
        #    这里是**恒等搬运**，不是加权和：readout.py 已经按时间升序把可见节点
        #    左对齐到 [N-k, N)，所以「第 g 组 = 第 g 行」不需要任何 gather / 索引。
        res = self.slot_out(h.astype(node_payload.dtype))               # [B,N,P*Dp]
        lt_img = node_payload + einops.rearrange(res, "b n (p d) -> b n p d", p=P)

        # 残差只加在视觉上：pos 是位置编码，state 是关节角，往里加一个学出来的
        # 视觉隐向量没有语义，只会污染它们。
        lt_img = einops.rearrange(lt_img, "b n p d -> b (n p) d")
        lt_pos = einops.rearrange(node_payload_pos, "b n p d -> b (n p) d")
        lt_state = einops.repeat(node_payload_state, "b n d -> b (n p) d", p=P)

        # ⑤ 长期段的 mask 必须**逐组**给，不能再用「有节点就整段可见」：
        #    空组零内容仍然可见的话，softmax 会去注意一堆零 token —— 实测「有效的
        #    零填充槽」代价 +0.193，是改变铺哪几帧的几千倍；整组屏蔽则没有这份代价。
        #    图上没有可读节点时整段让位给近期分支，也天然被逐组 mask 覆盖（k=0 ⇒ 全 0）。
        lt_mask = einops.repeat(node_read, "b n -> b (n p)", p=P)

        # ⑥ 近期分支：8 帧真实 token 直接铺开
        rc_img = einops.rearrange(recent_payload, "b r p d -> b (r p) d")
        rc_pos = einops.rearrange(recent_payload_pos, "b r p d -> b (r p) d")
        rc_state = einops.repeat(recent_payload_state, "b r d -> b (r p) d", p=P)
        rc_mask = einops.repeat(recent_mask, "b r -> b (r p)", p=P)

        # ⑦ 长期在前、近期在后
        img = jnp.concatenate([lt_img, rc_img], axis=1)
        pos = jnp.concatenate([lt_pos, rc_pos], axis=1)
        state = jnp.concatenate([lt_state, rc_state], axis=1)
        mask = jnp.concatenate([lt_mask, rc_mask], axis=1)
        return img, pos, state, mask
