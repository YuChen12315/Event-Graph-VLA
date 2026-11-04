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


class GRU_cell(nnx.Module):
    def __init__(self, dim: int, rngs: nnx.Rngs, dtype):
        self.w_i = nnx.Linear(dim, 3 * dim, rngs=rngs, dtype=dtype)
        self.w_h = nnx.Linear(dim, 3 * dim, rngs=rngs, dtype=dtype)
    
    def __call__(self, x, h):
        xi = einops.rearrange(self.w_i(x), "b n (g d) -> g b n d", g=3)
        hh = einops.rearrange(self.w_i(h), "b n (g d) -> g b n d", g=3)
        r = jax.nn.sigmoid(xi[0] + hh[0])
        z = jax.nn.sigmoid(xi[1] + hh[1])
        n = jnp.tanh(xi[2] + r * hh[2])
        return (1.0 - z) * n + z * h
    