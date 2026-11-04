import jax
import jax.numpy as jnp
import einops
import math
import heapq
from copy import deepcopy
import numpy as np
import ml_dtypes
from typing import Callable
import cv2
from collections import defaultdict


from openpi.shared import image_tools
from mme_vla_suite.shared.data_utils import *


def create_dict(indices):
    result = defaultdict(lambda: defaultdict(list)) # {step_idx: {view_idx: [patch_idx]}}
    for step_idx, view_idx, patch_idx in indices:
        result[step_idx][view_idx].append(patch_idx)
    return result


def _streaming_step(nodes_centroid, nodes_mass, nodes_t_start, nodes_t_last,
                    nodes_rep_img, nodes_rep_pos, nodes_rep_state, nodes_rep_time,
                    img, pos, state, t,
                    top_k, alpha, tau, merge_threshold, w_gap, w_span, c_min, recency_decay):
    """流式在线原型聚类的单步：把新帧 (img,pos,state,t) 合并/新增/替换进节点（就地更新）。

    B：节点存两套字段——「质心 centroid(2048) 只用于合并判断」，「代表真实帧 rep(16token) 用于动作输入」；
    C：硬合并资格 W_gap(距最近吸收帧) / W_span(节点总跨度，防链式合并跨事件) / c_min(视觉余弦下限)。
    """
    pool_new = img.reshape(-1, img.shape[-1]).mean(0).astype(np.float32)

    if not nodes_centroid:
        nodes_centroid.append(pool_new)
        nodes_mass.append(1.0)
        nodes_t_start.append(float(t)); nodes_t_last.append(float(t))
        nodes_rep_img.append(img); nodes_rep_pos.append(pos); nodes_rep_state.append(state)
        nodes_rep_time.append(float(t))
        return

    # 只在「合格」节点里挑最高分（硬资格过滤 + 软分数排序）
    best_idx, best_score = -1, -1.0
    for i in range(len(nodes_centroid)):
        visual = float(np.dot(pool_new, nodes_centroid[i]) /
                       (np.linalg.norm(pool_new) * np.linalg.norm(nodes_centroid[i]) + 1e-8))
        # 硬合并资格（C）
        if (t - nodes_t_last[i] > w_gap) or (t - nodes_t_start[i] > w_span) or (visual < c_min):
            continue
        temporal = float(np.exp(-(t - nodes_t_last[i]) / max(tau, 1e-3)))
        score = alpha * visual + (1.0 - alpha) * temporal
        if score > best_score:
            best_idx, best_score = i, score

    if best_idx >= 0 and best_score >= merge_threshold:
        # ① 合并：更新质心（按 mass 加权均值）+ 更新代表帧（离新质心更近的真实帧）
        i = best_idx
        w = nodes_mass[i]
        new_centroid = (w * nodes_centroid[i] + pool_new) / (w + 1.0)
        old_rep_pool = nodes_rep_img[i].reshape(-1, nodes_rep_img[i].shape[-1]).mean(0).astype(np.float32)
        if float(np.linalg.norm(pool_new - new_centroid)) < float(np.linalg.norm(old_rep_pool - new_centroid)):
            nodes_rep_img[i] = img; nodes_rep_pos[i] = pos; nodes_rep_state[i] = state
            nodes_rep_time[i] = float(t)
        nodes_centroid[i] = new_centroid
        nodes_mass[i] = w + 1.0
        nodes_t_last[i] = float(t)   # t_start 不变
    elif len(nodes_centroid) < top_k:
        # ② 无合格合并且有空位 → 新增
        nodes_centroid.append(pool_new)
        nodes_mass.append(1.0)
        nodes_t_start.append(float(t)); nodes_t_last.append(float(t))
        nodes_rep_img.append(img); nodes_rep_pos.append(pos); nodes_rep_state.append(state)
        nodes_rep_time.append(float(t))
    else:
        # ③ 无合格合并且满了 → 替换最不重要的（mass × 时效）
        importance = [m * (recency_decay ** (t - nodes_t_last[i])) for i, m in enumerate(nodes_mass)]
        evict = int(np.argmin(importance))
        nodes_centroid[evict] = pool_new
        nodes_mass[evict] = 1.0
        nodes_t_start[evict] = float(t); nodes_t_last[evict] = float(t)
        nodes_rep_img[evict] = img; nodes_rep_pos[evict] = pos; nodes_rep_state[evict] = state
        nodes_rep_time[evict] = float(t)


def _streaming_graph_coarsen(frame_list, top_k, alpha, tau, merge_threshold,
                             w_gap, w_span, c_min, recency_decay):
    """流式在线原型聚类：逐帧维护 ≤top_k 个原型（质心+代表帧），返回代表帧序列。

    frame_list: list[(img (p,d1), pos (p,d2), state (d3,), time (int))]
    返回 rep_img/rep_pos/rep_state/mass 四个 list（长度 ≤top_k，供 _nodes_to_memory 输出真实帧）。
    """
    nodes_centroid, nodes_mass, nodes_t_start, nodes_t_last = [], [], [], []
    nodes_rep_img, nodes_rep_pos, nodes_rep_state, nodes_rep_time = [], [], [], []

    for img, pos, state, t in frame_list:
        _streaming_step(nodes_centroid, nodes_mass, nodes_t_start, nodes_t_last,
                        nodes_rep_img, nodes_rep_pos, nodes_rep_state, nodes_rep_time,
                        img, pos, state, t, top_k, alpha, tau, merge_threshold,
                        w_gap, w_span, c_min, recency_decay)

    return nodes_rep_img, nodes_rep_pos, nodes_rep_state, nodes_mass


class MemoryBuffer:
    # We use 8x8 tokens each image for token dropping and 4x4 tokens for frame sampling.
    def __init__(
        self,
        num_views: int = 1,  # number of camera views to be used as memory, better use fixed-views
        img_emb_dim: int = 2048,  # the initila image token embedding dimension, siglip is 2048
        pos_emb_dim: int = 768, 
        state_emb_dim: int = 8,
        max_steps: int = 4096,
        compute_token_drop_score: bool = False, # set true when using token dropping
        token_drop_keptsize: int = 2048,   # max number of tokens to be kept in the heap for similarity score calculation
        token_drop_stride: int = 8,        # timestep stride for similarity score calculation, if too close, the difference will be too small
        prepare_buffer: bool = False,      # set true during online evaluation, set false during offline training
        pool_type: str = "mean",           # pooling type for the image tokens, mean or max
        vision_enc_fn: Callable | None = None,   # vision encoding function
    ):
        self.num_views = num_views
        self.img_emb_dim = img_emb_dim        
        self.pos_emb_dim = pos_emb_dim
        self.state_emb_dim = state_emb_dim

        self.pool_type = pool_type
        self.token_drop_keptsize = token_drop_keptsize
        self.token_drop_stride = token_drop_stride
        self.token_drop_last_frame = -1   
        self.scored_token_heap = []
        self.compute_token_drop_score = compute_token_drop_score
        
        self._history_feats = {}
        
        
        if prepare_buffer:
            # load models in GPUs to provide embeedings
            # used for dataset buliding and evaluation
            from mme_vla_suite.shared.posemb_3d import PosEmb3D
            if vision_enc_fn is None:
                from mme_vla_suite.shared.siglip_tokenizer import SigLipTokenizer
                siglip_tokenizer = SigLipTokenizer()
                self.vision_enc = jax.jit(siglip_tokenizer.__call__)
            else:
                self.vision_enc = vision_enc_fn
            pos_embedder = PosEmb3D(dim=pos_emb_dim)
            ranges = jnp.arange(max_steps)
            self.pos_emb_dict = {
                "8x8": np.array(pos_embedder(ranges, 8)),
                "4x4": np.array(pos_embedder(ranges, 4)),
                "2x2": np.array(pos_embedder(ranges, 2)),
            }
        else:
            # used in dataset loading mode
            self.vision_enc = None
            self.pos_emb_dict = None
        
    def add_buffer(
        self,
        images, #: (t v h w 3), np.int8
        states, #: (t d), np.float32
        step_idx_list: list[int],
        exec_start_idx: int = 0,   # 只有 EventGraph 子类用（§5.1 阶段分界）；基类忽略
    ):
        assert self.vision_enc is not None, "encode is not initialized"

        t, v, _, _, _ = images.shape
        assert v == self.num_views

        image_jnp = jnp.array(
            images.astype(np.float32) / 255.0 * 2.0 - 1.0
        )
        image_jnp = einops.rearrange(image_jnp, "t v h w c -> (t v) h w c")
        image_jnp = image_tools.resize_with_pad(image_jnp, 224, 224)
        image_jnp = einops.rearrange(image_jnp, "(t v) h w c -> t v h w c", t=t, v=v)
        output_emb = self.vision_enc(image_jnp)  # (t, v, 64, 2048)
        
        pooled_emb_8x8 = pool_tokens_to_size(output_emb, 64)  # (t, v, 64, 2048)
        pooled_emb_4x4 = pool_tokens_to_size(output_emb, 16)  # (t, v, 16, 2048)
        pooled_emb_2x2 = pool_tokens_to_size(output_emb, 4)  # (t, v, 4, 2048)
        
        for i, step_idx in enumerate(step_idx_list):
            image_emb_8x8 = jax.device_get(pooled_emb_8x8)[i]  # (v, 64, 2048)
            image_emb_4x4 = jax.device_get(pooled_emb_4x4)[i]  # (v, 16, 2048)
            image_emb_2x2 = jax.device_get(pooled_emb_2x2)[i]  # (v, 4, 2048)

            """

            关键理解 pos_emb_dict 的存储结构：之前 prepare_buffer 中预计算的位置编码，
            第一维并不是按 step_idx 排序，而是按 step_idx * num_views + view_idx 展平存储的。

            例如 num_views=2, 则第 0 步对应索引 0,1, 第 1 步对应索引 2,3……

            因此这里通过切片 step_idx*views : (step_idx+1)*views 恰好取出当前时间步下所有视角的位置编码，
            形状为 (v, 64, 768)，与视觉特征完美对齐。

            """
            pos_emb_8x8 = self.pos_emb_dict["8x8"][
                step_idx*self.num_views : (step_idx+1)*self.num_views]  # (v, 64, 768)
            pos_emb_4x4 = self.pos_emb_dict["4x4"][
                step_idx*self.num_views : (step_idx+1)*self.num_views]  # (v, 16, 768)
            pos_emb_2x2 = self.pos_emb_dict["2x2"][
                step_idx*self.num_views : (step_idx+1)*self.num_views]  # (v, 4, 768)
            
            token_emb_to_save = {
                "image_pixels": images[i].copy(), # fp32, this is for visualization
                "image_emb_8x8": image_emb_8x8, # bf16
                "image_emb_4x4": image_emb_4x4, # bf16
                "image_emb_2x2": image_emb_2x2, # bf16
                "pos_emb_8x8": pos_emb_8x8, # fp32
                "pos_emb_4x4": pos_emb_4x4, # fp32
                "pos_emb_2x2": pos_emb_2x2, # fp32
                "state_emb": states[i], # fp32
            }
            assert step_idx not in self._history_feats, f"step_idx {step_idx} already in buffer"
            self._history_feats[step_idx] = token_emb_to_save
            
            if self.compute_token_drop_score:
                """
                该函数会使用刚存入的 8×8 精细特征与历史堆中的 Token 计算相似度，并在堆满时丢弃最冗余的 Token，
                从而实现“在线关键帧筛选”。
                """
                self._process_token_drop_score(step_idx)    
            
    
    def get_history_feats(self, step_idx: int, remove_image_pixels: bool = True):
        if remove_image_pixels:
            return {k: v for k, v in self._history_feats[step_idx].items() if k != "image_pixels"}
        return self._history_feats[step_idx]

    
    def _process_token_drop_score(self, step_idx):
        if step_idx == 0: # all tokens in the first frame should be kept, only later frames are considered for token dropping
            for patch_idx in range(64): # 64 patches in 8x8 grid, which is fixed here for now
                for view_idx in range(self.num_views):
                    heapq.heappush(
                        self.scored_token_heap, (1000.0, step_idx, view_idx, patch_idx)
                    )
        
        if step_idx == self.token_drop_last_frame + self.token_drop_stride:
            prev_img = self._history_feats[max(0, self.token_drop_last_frame)]["image_pixels"]
            curr_img = self._history_feats[step_idx]["image_pixels"]
            
            prev_img = prev_img.astype(np.float32) / 255.0 * 2.0 - 1.0
            curr_img = curr_img.astype(np.float32) / 255.0 * 2.0 - 1.0

            prev_img_tokens = einops.rearrange(prev_img, "v (ph h) (pw w) c -> v (ph pw) (h w c)", ph=8, pw=8)
            curr_img_tokens = einops.rearrange(curr_img, "v (ph h) (pw w) c -> v (ph pw) (h w c)", ph=8, pw=8)
            
            for view_idx in range(self.num_views):
                # pixel works better than siglip embedding
                difference = np.abs(prev_img_tokens[view_idx] - curr_img_tokens[view_idx]).mean(axis=-1) 
                for patch_idx in range(64):
                    if difference[patch_idx] < 1e-4:
                        # too similar, skip this patch
                        continue
                    heapq.heappush(
                        self.scored_token_heap,
                        (
                            difference[patch_idx],
                            step_idx,     # timestep index
                            view_idx,     # view index
                            patch_idx,    # patch index
                        ),
                    )
                    if len(self.scored_token_heap) > self.token_drop_keptsize:
                        heapq.heappop(self.scored_token_heap)
            
            self.token_drop_last_frame += self.token_drop_stride
                
    
    def clear(self):
        self.scored_token_heap.clear()
        self.token_drop_last_frame = -1
        self._history_feats.clear()
        
        
    def get_token_dropping_indices(self):
        selected_tokens_heap = deepcopy(self.scored_token_heap)
        kept_indices = []
        while selected_tokens_heap:
            _, buffer_idx, view_idx, patch_idx = heapq.heappop(selected_tokens_heap)
            kept_indices.append((buffer_idx, view_idx, patch_idx))
                
        return kept_indices
    
    @staticmethod
    def filter_token_dropping_indices(kept_indices, step_idx, token_budget, is_sorted=True):
        kept_indices_filtered = [item for item in kept_indices if item[0] <= step_idx]
        kept_indices = kept_indices_filtered[-token_budget:]
        if is_sorted:
            kept_indices = sorted(kept_indices)
        return kept_indices
    
    
    def _prepare_token_dropping(self, history_feats, sorted_kept_indices, token_budget):
        img_emb = np.zeros((token_budget, self.img_emb_dim), dtype=np.float32)
        pos_emb = np.zeros((token_budget, self.pos_emb_dim), dtype=np.float32)
        state_emb = np.zeros((token_budget, self.state_emb_dim), dtype=np.float32)
        mask = np.zeros((token_budget), dtype=np.bool_)

        for idx, (buffer_idx, view_idx, patch_idx) in enumerate(sorted_kept_indices[:token_budget]):
            img_emb[idx] = history_feats[buffer_idx][f"image_emb_8x8"][view_idx][patch_idx]
            pos_emb[idx] = history_feats[buffer_idx][f"pos_emb_8x8"][view_idx][patch_idx]
            state_emb[idx] = history_feats[buffer_idx]["state_emb"]
            mask[idx] = True
        
        # print("effective token length: ", np.sum(mask))
                
        return img_emb, pos_emb, state_emb, mask
    

    def prepare_token_dropping(self, step_idx, token_budget, history_feats_gather_fn, kept_indices=None, *args, **kwargs):
        if kept_indices is None:
            indices_to_load = self.get_token_dropping_indices()
        else:
            indices_to_load = kept_indices
        
        sorted_kept_indices = self.filter_token_dropping_indices(
            indices_to_load, step_idx, token_budget, is_sorted=True)
        
        indices_to_load = sorted(set([item[0] for item in sorted_kept_indices]))
        # print("step_idx: ", step_idx)
        # print(f"indices_to_load (length: {len(indices_to_load)}/{len(sorted_kept_indices)}): {indices_to_load}")
        # self._visualize_token_dropping(sorted_kept_indices, step_idx)
        # import pdb; pdb.set_trace()
        
        history_feats = history_feats_gather_fn(indices_to_load, *args, **kwargs)

        return self._prepare_token_dropping(history_feats, sorted_kept_indices, token_budget)
    
    
    def _visualize_token_dropping(self, kept_indices, step_idx):
        dic = create_dict(sorted(kept_indices))
        images = [self._history_feats[idx]["image_pixels"][0] for idx in range(len(self._history_feats))]
        
        images_anno = []
        for step_idx in dic:
            img_anno = images[step_idx].copy()

            img_anno = cv2.putText(
                img_anno,
                f"{step_idx}",
                (img_anno.shape[1] // 2, img_anno.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                1,
            )
            mask = np.zeros(img_anno.shape[:2], dtype=np.uint8)

            for patch_idx in range(64):
                if patch_idx in dic[step_idx][0]:
                    continue
                h_center = patch_idx // 8 * 32 + 16
                w_center = patch_idx % 8 * 32 + 16
                mask = cv2.rectangle(
                    mask,
                    (w_center - 16, h_center - 16),
                    (w_center + 16, h_center + 16),
                    255,
                    -1,
                )
            img_anno[mask == 255] = (
                img_anno[mask == 255] * 0.3 + np.array([255, 255, 255]) * 0.7
            )
            
            images_anno.append(img_anno)
        
        import imageio, os
        imageio.mimsave(
            os.path.join(
                "debug",
                f"token_drop_anno_step_{step_idx}.mp4",
            ),
            images_anno,
            fps=10,
        )
        
    
    def get_frame_sampling_indices(self, step_idx, token_budget, token_per_image):
        """
        分母含义: token_per_image * self.num_views 代表 “存储一个时间步 (Step) 需要消耗的 Token 总数”。

        因为一个时间步包含 num_views 个视角的图像，每个图像占 token_per_image 个 Token, 
        所以相乘得到该步的总消耗。

        max_size 含义：用总预算除以单步消耗，向下取整，得到 “在不超过预算的前提下，最多能保留的历史帧数”。

        最终调用：拿着算出来的 max_size, 去调用 even_sampling_indices,
        在 [0, step_idx] 范围内均匀选出这么多个帧索引。
        """
        max_size = token_budget // (token_per_image * self.num_views)
        return even_sampling_indices(step_idx, max_size)


    # 仿照原本的get_frame_sampling_indices函数，直接改成调用自己写的 “16+16” 抽帧函数
    def get_frame_sampling_indices_hybrid(self, step_idx, token_budget, token_per_image, recent_frames):
        max_size = token_budget // (token_per_image * self.num_views)
        return hybrid_frame_sampling_indices(step_idx, max_size, recent_frames)
    
    
    def _prepare_frame_sampling(self, history_feats, indices_to_load, token_budget, token_per_image):
        spatial_size = str(int(math.sqrt(token_per_image)))
        spatial_key = f"{spatial_size}x{spatial_size}"
        max_size = token_budget // (token_per_image * self.num_views)
        
        # 加载数据：从字典中按索引抽取
        sampled_img_emb = self._load_emb(history_feats, indices_to_load, f"image_emb_{spatial_key}")
        sampled_pos_emb = self._load_emb(history_feats, indices_to_load, f"pos_emb_{spatial_key}")
        sampled_state_emb = self._load_emb(history_feats, indices_to_load, "state_emb")
        mask = np.ones((sampled_img_emb.shape[0]), dtype=np.bool_)
                
        # we use right padding to the perceptual memory
        sampled_img_emb, sampled_pos_emb, sampled_state_emb, mask = right_padding_token_emb(
            sampled_img_emb, sampled_pos_emb, sampled_state_emb, mask, max_size
        )
                
        img_emb =  np.reshape(sampled_img_emb, (-1, self.img_emb_dim))
        pos_emb = np.reshape(sampled_pos_emb, (-1, self.pos_emb_dim))
        mask = np.repeat(mask, self.num_views * token_per_image)
        state_emb = np.repeat(sampled_state_emb, self.num_views * token_per_image, axis=0)
        
        # print("effective token length: ", np.sum(mask))
                
        return img_emb, pos_emb, state_emb, mask
            
    
    def prepare_frame_sampling(self, step_idx, token_budget, token_per_image, history_feats_gather_fn,  *args, **kwargs):
        indices_to_load = self.get_frame_sampling_indices(step_idx, token_budget, token_per_image)
        # print("step_idx: ", step_idx, "indices_to_load: ", indices_to_load, "length: ", len(indices_to_load))
        # self._visualize_frame_sampling(indices_to_load, step_idx)
        history_feats = history_feats_gather_fn(indices_to_load, *args, **kwargs)
        return self._prepare_frame_sampling(history_feats, indices_to_load, token_budget, token_per_image)


    # 仿照原本的prepare_frame_sampling函数，直接改成调用自己写的 get_frame_sampling_indices_hybrid
    def prepare_frame_sampling_hybrid(self, step_idx, token_budget, token_per_image, recent_frames,
                                  history_feats_gather_fn, *args, **kwargs):
        indices_to_load = self.get_frame_sampling_indices_hybrid(
            step_idx, token_budget, token_per_image, recent_frames)
        history_feats = history_feats_gather_fn(indices_to_load, *args, **kwargs)
        return self._prepare_frame_sampling(history_feats, indices_to_load, token_budget, token_per_image)


    # 关键帧选择抽帧：近期帧 + 关键帧（子目标边界）+ 均匀上下文
    def prepare_keyframe_selection(
        self, step_idx, token_budget, token_per_image, recent_frames,
        keyframe_indices, history_feats_gather_fn, *args, **kwargs
    ):
        max_size = token_budget // (token_per_image * self.num_views)
        indices_to_load = keyframe_sampling_indices(
            step_idx, max_size, recent_frames, keyframe_indices)
        history_feats = history_feats_gather_fn(indices_to_load, *args, **kwargs)
        return self._prepare_frame_sampling(
            history_feats, indices_to_load, token_budget, token_per_image)



    # 图 + DPP 记忆选择（TCGM 学习版）：产出候选窗口帧，模型侧做可学习选择。
    # 候选 = 近期 recent_frames 帧（连续，细粒度）+ 更早历史均匀抽 (window - recent_frames) 帧（覆盖全程）。
    # 返回 (window, p, d1) img / (window, p, d2) pos / (window, d3) state / (window,) mask，right-pad 到 window。
    def prepare_candidate_graph_dpp(self, step_idx, window, recent_frames, token_per_image, history_feats_gather_fn, *args, **kwargs):
        # 近期连续 + 历史均匀 → 覆盖整段历史，又保留近期细粒度（避免只看到最近 window 帧）
        candidate_indices = hybrid_frame_sampling_indices(step_idx, window, recent_frames)

        history_feats = history_feats_gather_fn(candidate_indices, *args, **kwargs)

        spatial_key = f"{int(math.sqrt(token_per_image))}x{int(math.sqrt(token_per_image))}"
        img = self._load_emb(history_feats, candidate_indices, f"image_emb_{spatial_key}")  # (n, v, p, d1)
        pos = self._load_emb(history_feats, candidate_indices, f"pos_emb_{spatial_key}")     # (n, v, p, d2)
        state = self._load_emb(history_feats, candidate_indices, "state_emb")                # (n, d3)

        img = img[:, 0]  # (n, p, d1)，压掉 view 维
        pos = pos[:, 0]  # (n, p, d2)

        return self._right_pad_candidate(img, pos, state, window)

    @staticmethod
    def _right_pad_candidate(img, pos, state, window):
        """把 (n, ...) 的候选序列 right-pad 到 window 帧，并给出 mask。"""
        n = img.shape[0]
        if n < window:
            img = np.concatenate([img, np.zeros((window - n, *img.shape[1:]), dtype=img.dtype)], axis=0)
            pos = np.concatenate([pos, np.zeros((window - n, *pos.shape[1:]), dtype=pos.dtype)], axis=0)
            state = np.concatenate([state, np.zeros((window - n, *state.shape[1:]), dtype=state.dtype)], axis=0)
            mask = np.concatenate([np.ones(n, dtype=np.bool_), np.zeros(window - n, dtype=np.bool_)], axis=0)
        else:
            img, pos, state = img[:window], pos[:window], state[:window]
            mask = np.ones(window, dtype=np.bool_)
        return img, pos, state, mask

    # 流式在线原型聚类记忆（TCGM v2，B+C 版）：逐帧维护 ≤top_k 个原型（质心判断 + 真实代表帧输入）。
    def prepare_graph_streaming(self, step_idx, token_per_image, top_k, alpha, tau,
                                merge_threshold, w_gap, w_span, c_min, recency_decay,
                                history_feats_gather_fn, *args, **kwargs):
        spatial_key = f"{int(math.sqrt(token_per_image))}x{int(math.sqrt(token_per_image))}"
        indices = list(range(step_idx + 1))
        history_feats = history_feats_gather_fn(indices, *args, **kwargs)

        frame_list = []
        for t in indices:
            f = history_feats[t]
            img = np.asarray(f[f"image_emb_{spatial_key}"], dtype=np.float32)[0]   # (p, d1)
            pos = np.asarray(f[f"pos_emb_{spatial_key}"], dtype=np.float32)[0]     # (p, d2)
            state = np.asarray(f["state_emb"], dtype=np.float32)                    # (d3,)
            frame_list.append((img, pos, state, t))  # 原始 step 索引，τ 是原始步数尺度（与预计算一致）

        rep_img, rep_pos, rep_state, _ = _streaming_graph_coarsen(
            frame_list, top_k, alpha, tau, merge_threshold, w_gap, w_span, c_min, recency_decay)

        return self._nodes_to_memory(rep_img, rep_pos, rep_state, top_k, token_per_image)

    @staticmethod
    def _nodes_to_memory(nodes_img, nodes_pos, nodes_state, top_k, token_per_image):
        """把 ≤top_k 个原型 right-pad 到 top_k，展平成 (top_k*p, d)，dtype 对齐下游（img bf16，pos/state f32）。"""
        img = np.stack(nodes_img, axis=0)     # (n, p, d1)
        pos = np.stack(nodes_pos, axis=0)     # (n, p, d2)
        state = np.stack(nodes_state, axis=0)  # (n, d3)
        n = img.shape[0]
        if n < top_k:
            img = np.concatenate([img, np.zeros((top_k - n, *img.shape[1:]), dtype=img.dtype)], axis=0)
            pos = np.concatenate([pos, np.zeros((top_k - n, *pos.shape[1:]), dtype=pos.dtype)], axis=0)
            state = np.concatenate([state, np.zeros((top_k - n, *state.shape[1:]), dtype=state.dtype)], axis=0)
            mask = np.concatenate([np.ones(n, dtype=np.bool_), np.zeros(top_k - n, dtype=np.bool_)], axis=0)
        else:
            img, pos, state = img[:top_k], pos[:top_k], state[:top_k]
            mask = np.ones(top_k, dtype=np.bool_)

        # 下游 FeatureEncoder 用 bf16，img 对齐回 bf16；pos/state 保持 f32（和批式一致）
        # 用 ml_dtypes 纯 numpy 转 bf16（不用 jnp，避免预计算脚本跑到 GPU 上导致 OOM）
        img_flat = img.reshape(-1, img.shape[-1]).astype(ml_dtypes.bfloat16)
        pos_flat = pos.reshape(-1, pos.shape[-1])
        state_flat = np.repeat(state, token_per_image, axis=0)
        mask_flat = np.repeat(mask, token_per_image)
        return img_flat, pos_flat, state_flat, mask_flat

    def _visualize_frame_sampling(self, indices_to_load, step_idx):
        images = [self._history_feats[idx]["image_pixels"][0] for idx in indices_to_load]
        
        import imageio, os
        imageio.mimsave(
            os.path.join(
                "debug",
                f"frame_sampling_step_{step_idx}.mp4",
            ),
            images,
            fps=2,
        )
                
    @staticmethod
    def _load_emb(history_feats: dict, indices_to_load: list[int], key: str):
        return np.stack(
            [history_feats[idx][key] for idx in indices_to_load],
            axis=0,
        )
        
    def default_history_feats_gather_fn(self, indices_to_load, *args, **kwargs):
        return {idx: self._history_feats[idx] for idx in indices_to_load}


class MemoryBufferRecurrent(MemoryBuffer):
    # By default, we use 8x8 tokens each image for recurrent memory.
    def __init__(
        self,
        input_obs_horizon: int = 8, 
        max_recur_steps: int = 64, # max number of recurrent steps for all history 
        max_video_steps: int = 40, # max number of recurrent steps to load for the video-based observations for certain video-conditioned tasks
        *args, **kwargs
    ):
        
        super().__init__(*args, **kwargs)
                
        self.input_obs_horizon = input_obs_horizon
        self.max_recur_steps = max_recur_steps
        self.max_video_steps = max_video_steps

    
    def add_buffer(
        self,
        images, #: (t v h w 3), np.int8
        state, #: (t d), np.float32
        step_idx_list: list[int],
        exec_start_idx: int = 0,   # 与基类保持同签名（调用方统一传）；本类不用
    ):
        assert self.vision_enc is not None, "encode is not initialized"

        t, v, _, _, _ = images.shape
        assert v == self.num_views

        image_jnp = jnp.array(
            images.astype(np.float32) / 255.0 * 2.0 - 1.0
        )
        image_jnp = einops.rearrange(image_jnp, "t v h w c -> (t v) h w c")
        image_jnp = image_tools.resize_with_pad(image_jnp, 224, 224)
        image_jnp = einops.rearrange(image_jnp, "(t v) h w c -> t v h w c", t=t, v=v)
        output_emb = self.vision_enc(image_jnp)  # (t, v, 64, 2048)
        
        # use 8x8 tokens, since recurrent memory can take more tokens than perceptual memory
        pooled_emb_8x8 = pool_tokens_to_size(output_emb, 64)  # (t, v, 64, 2048)
        
        for i, step_idx in enumerate(step_idx_list):
            image_emb_8x8 = jax.device_get(pooled_emb_8x8)[i]  # (v, 64, 2048)
            
            pos_emb_8x8 = self.pos_emb_dict["8x8"][
                step_idx*self.num_views : (step_idx+1)*self.num_views]  # (v, 64, 768)
            
            token_emb_to_save = { # recurrent memory only uses 8x8
                # "image_pixels": images[i].copy(), # fp32
                "image_emb_8x8": image_emb_8x8,
                "pos_emb_8x8": pos_emb_8x8,
                "state_emb": state[i],
            }
            assert step_idx not in self._history_feats, f"step_idx {step_idx} already in buffer"
            self._history_feats[step_idx] = token_emb_to_save  
    
    
    def get_token_recurrent_indices(self, step_idx, exec_start_idx):
        input_obs_horizon = self.input_obs_horizon
        max_recur_steps = self.max_recur_steps
        max_video_steps = self.max_video_steps
                
        assert step_idx >=0 and step_idx >= exec_start_idx
        
        if exec_start_idx == 0:
            # no videos
            if step_idx < input_obs_horizon:
                indices_to_load = [step_idx]
            else:
                start_idx = step_idx % input_obs_horizon
                indices_to_load = list(range(start_idx, step_idx+1, input_obs_horizon))
                indices_to_load = indices_to_load[-max_recur_steps:]
        else:
            if exec_start_idx <= input_obs_horizon * 2:
                # if too short, use more granular sampling
                video_indices = list(range(0, exec_start_idx, input_obs_horizon//2))
            elif exec_start_idx <= max_video_steps * input_obs_horizon:
                # if not too short, use regular sampling
                video_indices = list(range(0, exec_start_idx, input_obs_horizon))
            else:
                # if too long, use fixed-size sampling
                video_indices = np.linspace(0, exec_start_idx - 1, max_video_steps, dtype=int).tolist()
            
            if step_idx-exec_start_idx < input_obs_horizon:
                rest_indices = [step_idx]
            else:
                start_idx = (step_idx-exec_start_idx) % input_obs_horizon + exec_start_idx
                rest_indices = list(range(start_idx, step_idx+1, input_obs_horizon))
            
            combined_indices = video_indices + rest_indices
            indices_to_load = combined_indices[-max_recur_steps:]
            
        assert len(indices_to_load) > 0 and len(indices_to_load) <= max_recur_steps
            
        return indices_to_load
    
    
    def _prepare_token_recurrent(self, history_feats, indices_to_load, padding=True):        
        recur_image_embs = []
        recur_pos_embs = []
        recur_state_embs = []
        recur_masks = []
        
        for i in indices_to_load:
            recur_image_embs.append(history_feats[i]["image_emb_8x8"])  # (v, p, d)
            recur_pos_embs.append(history_feats[i]["pos_emb_8x8"])  # (v, p, d)
            recur_state_embs.append(history_feats[i]["state_emb"])  # (d)
            recur_masks.append(True)  # (1)
        
        recur_image_emb = np.stack(recur_image_embs, axis=0)
        recur_pos_emb = np.stack(recur_pos_embs, axis=0)
        recur_state_emb = np.stack(recur_state_embs, axis=0)
        recur_mask = np.stack(recur_masks, axis=0)
        
        if padding:
            recur_image_emb, recur_pos_emb, recur_state_emb, recur_mask = left_padding_token_emb(
                recur_image_emb, recur_pos_emb, recur_state_emb, recur_mask, self.max_recur_steps
            )                
        return recur_image_emb, recur_pos_emb, recur_state_emb, recur_mask


    def prepare_token_recurrent(self, step_idx, exec_start_idx,  history_feats_gather_fn,  *args, **kwargs):
        indices_to_load = self.get_token_recurrent_indices(step_idx, exec_start_idx)
        # print("step_idx: ", step_idx, "exec_start_idx: ", exec_start_idx)
        # print(f"indices_to_load ({len(indices_to_load)}): {indices_to_load}")
        assert len(indices_to_load) > 0 and len(indices_to_load) <= self.max_recur_steps
        history_feats = history_feats_gather_fn(indices_to_load, *args, **kwargs)
        # self._visualize_token_recurrent(indices_to_load, step_idx)
        return self._prepare_token_recurrent(history_feats, indices_to_load)

    def _visualize_token_recurrent(self, indices_to_load, step_idx):
        images = [self._history_feats[idx]["image_pixels"][0] for idx in indices_to_load]
        import imageio, os
        imageio.mimsave(
            os.path.join(
                "debug",
                f"token_recurrent_step_{step_idx}.mp4",
            ),
            images,
            fps=2,
        )