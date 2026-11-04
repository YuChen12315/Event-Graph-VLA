import numpy as np
import math
import einops
import flax.nnx as nnx



def even_sampling_indices(step_idx: int, token_budget: int) -> list[int]:
    """Generate evenly spaced indices for sampling frames."""
    """
    step_idx 代表当前当前的最大索引值，即序列总长度
    token_budget 代表希望采样出的索引个数

    """
    if step_idx < token_budget: 
        return list(range(step_idx+1)) # 没够采样数，直接返回 
    else:
        return np.linspace(0, step_idx, token_budget, dtype=np.int32).tolist() # 够了就均匀采
    


# 自己设计的 “16+16” 抽帧
def hybrid_frame_sampling_indices(
    step_idx: int,
    total_frames: int = 32,
    recent_frames: int = 16,
) -> list[int]:
    """拆解采样：最近 recent_frames 帧逐帧保留，更早历史均匀采样剩余帧。

    total_frames=32, recent_frames=16 时：
    - step_idx < 32：历史不足，返回全部帧 [0..step_idx]（交给后续 right-padding 补零）
    - step_idx >= 32：返回 [0..step_idx-16] 均匀采 16 帧 + [step_idx-15..step_idx] 共 16 帧
    """
    if step_idx < total_frames:
        return list(range(step_idx + 1))

    # 最近 16 帧（含当前帧）
    recent = list(range(step_idx - recent_frames + 1, step_idx + 1))

    # 更早的历史 [0, step_idx - recent_frames] 均匀采样剩下的帧
    uniform_frames = total_frames - recent_frames
    earlier = np.linspace(0, step_idx - recent_frames, uniform_frames, dtype=np.int32).tolist()

    return earlier + recent



# 关键帧选择抽帧：近期帧 + 关键帧（子目标边界）+ 均匀上下文填充
def keyframe_sampling_indices(
    step_idx: int,
    total_frames: int = 32,
    recent_frames: int = 16,
    keyframe_indices: list[int] | None = None,
) -> list[int]:
    """近期帧逐帧保留，更早历史优先保留"关键帧"（子目标边界），其余用均匀采样填充。

    - keyframe_indices: 边界帧索引（训练用真实边界，评测用预测 progress 的跳变）。
    - 总帧数仍是 total_frames（budget//token_per_image），不足时交给 right-padding 补零。
    """
    if step_idx < total_frames:
        return list(range(step_idx + 1))

    recent = list(range(step_idx - recent_frames + 1, step_idx + 1))
    earlier_budget = total_frames - recent_frames
    earlier_hi = step_idx - recent_frames  # 早期区间上界

    keyframes = [k for k in (keyframe_indices or []) if k <= earlier_hi]
    keyframes = keyframes[-earlier_budget:]  # 取最近的 earlier_budget 个关键帧

    # 用均匀采样填充剩余槽位（排除已是关键帧的）
    fill_n = earlier_budget - len(keyframes)
    fill = even_sampling_indices(earlier_hi, fill_n) if fill_n > 0 else []
    fill = [f for f in fill if f not in set(keyframes)]

    earlier = sorted(set(keyframes) | set(fill))
    return earlier + recent



def right_padding_token_emb(
    sampled_img_emb, # (l v p d1)
    sampled_pos_emb, # (l v p d2)
    sampled_state_emb, # (l d3)
    mask, # (l)
    max_size: int):

    # 当前长度 < 目标长度（l < max_size）→ 右侧填充（Right Padding）
    # 注意：padding 必须用和输入相同的 dtype，否则会把 bf16 记忆 upcast 成 float64，导致存储/加载膨胀
    if sampled_img_emb.shape[0] < max_size:
        sampled_img_emb = np.concatenate(
            [
                sampled_img_emb,
                np.zeros(
                    (
                        max_size - sampled_img_emb.shape[0],
                        *sampled_img_emb.shape[1:],
                    ),
                    dtype=sampled_img_emb.dtype,
                ),
            ],
            axis=0,
        )
        sampled_pos_emb = np.concatenate(
            [
                sampled_pos_emb,
                np.zeros(
                    (
                        max_size - sampled_pos_emb.shape[0],
                        *sampled_pos_emb.shape[1:],
                    ),
                    dtype=sampled_pos_emb.dtype,
                ),
            ],
            axis=0,
        )
        sampled_state_emb = np.concatenate(
            [
                sampled_state_emb,
                np.zeros(
                    (
                        max_size - sampled_state_emb.shape[0],
                        *sampled_state_emb.shape[1:],
                    ),
                    dtype=sampled_state_emb.dtype,
                ),
            ],
            axis=0,
        )
        mask = np.concatenate(
            [mask, np.zeros((max_size - mask.shape[0]), dtype=np.bool_)], axis=0
        )

    # 当前长度 ≥ 目标长度（l >= max_size）→ 直接截断（Truncation）
    else:
        sampled_img_emb = sampled_img_emb[:max_size]
        sampled_pos_emb = sampled_pos_emb[:max_size]
        sampled_state_emb = sampled_state_emb[:max_size]
        mask = mask[:max_size]

    return sampled_img_emb, sampled_pos_emb, sampled_state_emb, mask



def left_padding_token_emb(
    recur_image_emb, # (l v p d1)
    recur_pos_emb, # (l v p d2)
    recur_state_emb, # (l d3)
    recur_mask, # (l)
    max_recur_steps: int
):
    if recur_image_emb.shape[0] < max_recur_steps:
        recur_image_emb = np.concatenate(
            [
                np.zeros(
                    (
                        max_recur_steps - recur_image_emb.shape[0],
                        *recur_image_emb.shape[1:],
                    )
                ),
                recur_image_emb,
            ],
            axis=0,
        )
        recur_pos_emb = np.concatenate(
            [
                np.zeros(
                    (
                        max_recur_steps - recur_pos_emb.shape[0],
                        *recur_pos_emb.shape[1:],
                    )
                ),
                recur_pos_emb,
            ],
            axis=0,
        )
        recur_state_emb = np.concatenate(
            [
                np.zeros(
                    (
                        max_recur_steps - recur_state_emb.shape[0],
                        *recur_state_emb.shape[1:],
                    )
                ),
                recur_state_emb,
            ],
            axis=0,
        )
        recur_mask = np.concatenate(
            [
                np.zeros(max_recur_steps - recur_mask.shape[0], dtype=np.bool_),
                recur_mask,
            ],
            axis=0,
        )
    else:
        recur_image_emb = recur_image_emb[-max_recur_steps:]
        recur_pos_emb = recur_pos_emb[-max_recur_steps:]
        recur_state_emb = recur_state_emb[-max_recur_steps:]
        recur_mask = recur_mask[-max_recur_steps:]

    return recur_image_emb, recur_pos_emb, recur_state_emb, recur_mask





def pool_tokens_to_size(
    tokens, # (b v p d) or (b p d)
    target_size: int = 64,
    pool_type: str = "mean",
):
    """对视觉 patch token 采样到 target_size 尺寸"""

    if len(tokens.shape) == 4:
        b, v, p, d = tokens.shape
    elif len(tokens.shape) == 3:
        b, p, d = tokens.shape
    else:
        raise ValueError(f"Invalid tokens shape: {tokens.shape}")
    if p == target_size:
        return tokens

    h = w = int(math.sqrt(p))
    if len(tokens.shape) == 4:
        tokens_2d = einops.rearrange(tokens, "b v (h w) d -> (b v) h w d", h=h, w=w)
    else:
        tokens_2d = einops.rearrange(tokens, "b (h w) d -> b h w d", h=h, w=w)

    pool_size = int(math.sqrt(p // target_size))
    if pool_type == "mean":
        pool_func = nnx.avg_pool
    elif pool_type == "max":
        pool_func = nnx.max_pool
    else:
        raise ValueError(f"Invalid pool type: {pool_type}")

    pooled = pool_func(
        tokens_2d, window_shape=(pool_size, pool_size), strides=(pool_size, pool_size)
    )
    if len(tokens.shape) == 4:
        out = einops.rearrange(pooled, "(b v) h w d -> b v (h w) d", b=b, v=v)
    else:
        out = einops.rearrange(pooled, "b h w d -> b (h w) d", b=b)

    return out


if __name__ == "__main__":
    # img_emb = np.random.randn(16, 256, 512).astype(np.float32)
    # out = pool_tokens_to_size(img_emb, 64)
    # print(out.shape)
    out = hybrid_frame_sampling_indices(18)
    print(out)
