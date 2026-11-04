from collections.abc import Sequence
import logging
import time
from typing import Any, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0
from mme_vla_suite.shared.mem_buffer import MemoryBuffer, MemoryBufferRecurrent

class MME_VLA_Policy:
    def __init__(
        self,
        model: HistoryPi0,
        *,
        seed: int = 42,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        norm_stats: dict[str, _transforms.NormStats] | None = None,
        use_quantiles: bool = False,
    ):
        self._model = model
        self._seed = seed
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._vision_encode = nnx_utils.module_jit(model.vision_encode)
        
        
        self.config = model.history_config
        self.mem_buffer = None

        self.state_norm_stats = norm_stats['state']
        self.use_quantiles = use_quantiles
        
        self.reset()
        
    
    def _prepare_mem_buffer(self):
        if self.config is None or self.config.representation_type == "symbolic":
            self.mem_buffer = None
        elif (
            self.config.representation_type == "perceptual"
            and self.config.perceptual_memory.type == "event_graph"
        ):
            self.mem_buffer = self._make_eventgraph_buffer()
        elif self.config.representation_type == "recurrent":
            self.mem_buffer = MemoryBufferRecurrent(
                num_views=self.config.num_views,
                img_emb_dim=self.config.memory_feature.img.input_dim,
                pos_emb_dim=self.config.memory_feature.pos.input_dim,
                state_emb_dim=self.config.memory_feature.state.input_dim,
                input_obs_horizon=self.config.streaming_obs_horizon,
                max_recur_steps=self.config.recurrent_memory.max_recur_steps,
                max_video_steps=self.config.recurrent_memory.max_pretraj_steps,
                prepare_buffer=True, vision_enc_fn=self._vision_encode,
            )
        else:
            self.mem_buffer = MemoryBuffer(
                num_views=self.config.num_views,
                img_emb_dim=self.config.memory_feature.img.input_dim,
                pos_emb_dim=self.config.memory_feature.pos.input_dim,
                state_emb_dim=self.config.memory_feature.state.input_dim,
                compute_token_drop_score = self.config.perceptual_memory.type == "token_dropping",
                token_drop_stride=self.config.streaming_obs_horizon // 2,
                prepare_buffer=True, vision_enc_fn=self._vision_encode,
            )

    def _make_eventgraph_buffer(self):
        """Event-Graph 的在线缓冲（v1.3 §10 Phase 6）。

        返回的仍是 MemoryBuffer 的子类，所以 `self.mem_buffer` 这个名字不变 —— 但
        **必须**在这里拦下来，绝不能让它落到 `_prepare_mem_buffer` 最后那个 else 去：
        那会安静地退化成 frame_sampling，v1.3 §11 明令禁止，而且不报错，只是记忆
        换成了另一种，评测数字看着还挺正常。
        """
        from mme_vla_suite.models.representation.event_graph.config import GraphMemoryConfig
        from mme_vla_suite.models.representation.event_graph.frame_summary import (
            FrozenFrameSummary,
        )
        from mme_vla_suite.models.representation.event_graph.online import (
            EventGraphMemoryBuffer,
        )

        cfg = GraphMemoryConfig.from_history_config(self.config)
        if cfg.token_per_image != 16:
            # 在线逐帧那一步只取了 image_emb_4x4（16 个 token）。其它池化尺寸在
            # memory_feature 里对不上，会拿错特征且不报错。
            raise ValueError(
                f"event_graph 在线推理要求 token_per_image=16（image_emb_4x4），"
                f"得到 {cfg.token_per_image}。")
        pca = FrozenFrameSummary.load(cfg.resolve_pca_path())
        mf = self.config.memory_feature
        return EventGraphMemoryBuffer(
            cfg, pca,
            dims=(int(mf.img.input_dim), int(mf.pos.input_dim), int(mf.state.input_dim)),
            normalize_state=self._normalize_state,
            num_views=self.config.num_views,
            img_emb_dim=int(mf.img.input_dim),
            pos_emb_dim=int(mf.pos.input_dim),
            state_emb_dim=int(mf.state.input_dim),
            vision_enc_fn=self._vision_encode,
        )

    @override
    def infer(self, obs: dict) -> dict:
        if self.config is not None and self.config.representation_type != "symbolic":
            assert len(self.mem_buffer._history_feats) > 0, \
                "history feats is empty, add buffer first"

        self._last_prompt = str(obs.get("prompt", ""))  # 记录当前任务，供 episode 结束时打印
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._prepare_history(inputs)
        inputs = self._input_transform(inputs)
        observation = HistAugObservation.from_dict(
            jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        )
        self._rng, sample_rng = jax.random.split(self._rng)
    
        start_time = time.monotonic()
        outputs = {
            "state": observation.state,
            "actions": self._sample_actions(sample_rng, observation, **self._sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["infer_time_ms"] = model_time * 1000

        return outputs
    
    @override
    def reset(self) -> None:
        # 记录上一 episode 检测到的关键帧数（keyframe_selection 验证日志）
        if getattr(self, "_keyframe_indices", None):
            logging.info(f"[keyframe] episode 检测到 {len(self._keyframe_indices)} 个关键帧 prompt='{getattr(self, '_last_prompt', '')}'")
        del self.mem_buffer
        # mem_buffer 和它的 EventGraphController 之间有引用环（controller 的
        # pca_tokens_fn 是 buffer 的绑定方法），光 del 不会立刻回收；而每个 buffer
        # 里是所有历史帧的像素+PCA token（单 episode 可达数 GB）。长跑评测时这些
        # 循环垃圾会把进程内存吃到上百 GB（2026-10-24 实测 ~2 GB/episode）。
        import gc

        gc.collect()
        self._prepare_mem_buffer()
        self.step_idx = -1
        self.exec_start_idx = 0
        self._rng = jax.random.key(self._seed)
        self._last_prompt = ""
        self._keyframe_indices = []  # 关键帧选择用的运行时关键帧索引
            
    
    def add_buffer(self, obs: dict) -> None:
        if self.mem_buffer is None:
            return
        images = obs["images"]
        states = obs["state"]
        if obs.get("exec_start_idx", 0) > 0: # has video
            self.exec_start_idx = obs["exec_start_idx"]
        
        step_idx_list = list(range(self.step_idx+1, self.step_idx + len(images) + 1))
        # exec_start_idx 由基类忽略、EventGraph 子类消费：第一批观测（condensed
        # video）必须在这批帧入图**之前**就知道阶段分界，否则那一段会被当成执行阶段。
        self.mem_buffer.add_buffer(images, states, step_idx_list,
                                   exec_start_idx=self.exec_start_idx)
        self.step_idx += len(images)

    def _normalize_state(self, state):
        if self.use_quantiles:
            return (state - self.state_norm_stats.q01) / (self.state_norm_stats.q99 - self.state_norm_stats.q01 + 1e-6) * 2.0 - 1.0
        else:
            return (state - self.state_norm_stats.mean) / (self.state_norm_stats.std + 1e-6)

    def _prepare_history(self, inputs: dict) -> dict:
        if self.config is None or self.config.representation_type == "symbolic":
            return inputs
        
        if self.config.representation_type == "recurrent":
            history_feats_gather_fn = self.mem_buffer.default_history_feats_gather_fn
            recur_image_emb, recur_pos_emb, recur_state_emb, recur_mask = \
                self.mem_buffer.prepare_token_recurrent(
                    self.step_idx, self.exec_start_idx, history_feats_gather_fn)
            inputs["recur_image_emb"] = recur_image_emb
            inputs["recur_pos_emb"] = recur_pos_emb
            inputs["recur_state_emb"] = self._normalize_state(recur_state_emb)
            inputs["recur_mask"] = recur_mask
        elif self.config.representation_type == "perceptual":
            history_feats_gather_fn = self.mem_buffer.default_history_feats_gather_fn
            token_budget = self.config.budget

            if self.config.perceptual_memory.type == "event_graph":   # ← Event-Graph（v1.3）
                # 唯一走这条路的分支：**不回退 frame_sampling**（§11 / Phase 6）。
                # memory() 和训练侧 dataset 调的是同一个 readout，产出同形的 15 个张量。
                inputs.update(self.mem_buffer.memory())
            elif self.config.perceptual_memory.type == "token_dropping":
                static_image_emb, static_pos_emb, static_state_emb, static_mask = \
                    self.mem_buffer.prepare_token_dropping(
                        self.step_idx, token_budget, history_feats_gather_fn)
                inputs["static_image_emb"] = static_image_emb
                inputs["static_pos_emb"] = static_pos_emb
                inputs["static_state_emb"] = self._normalize_state(static_state_emb)
                inputs["static_mask"] = static_mask
            elif self.config.perceptual_memory.type == "frame_sampling_hybrid":   # ← 新增
                token_per_image = self.config.token_per_image
                recent_frames = self.config.perceptual_memory.get("recent_frames", 16)
                static_image_emb, static_pos_emb, static_state_emb, static_mask = \
                    self.mem_buffer.prepare_frame_sampling_hybrid(
                        self.step_idx, token_budget, token_per_image, recent_frames, history_feats_gather_fn)
                inputs["static_image_emb"] = static_image_emb
                inputs["static_pos_emb"] = static_pos_emb
                inputs["static_state_emb"] = self._normalize_state(static_state_emb)
                inputs["static_mask"] = static_mask
            elif self.config.perceptual_memory.type == "keyframe_selection":   # ← 关键帧选择
                token_per_image = self.config.token_per_image
                recent_frames = self.config.perceptual_memory.get("recent_frames", 16)
                static_image_emb, static_pos_emb, static_state_emb, static_mask = \
                    self.mem_buffer.prepare_keyframe_selection(
                        self.step_idx, token_budget, token_per_image, recent_frames,
                        self._keyframe_indices, history_feats_gather_fn)
                inputs["static_image_emb"] = static_image_emb
                inputs["static_pos_emb"] = static_pos_emb
                inputs["static_state_emb"] = self._normalize_state(static_state_emb)
                inputs["static_mask"] = static_mask
            elif self.config.perceptual_memory.type in (
                "graph_dpp",
                "graph_select",
            ):   # ← 图+DPP / 图选帧(v4)
                token_per_image = self.config.token_per_image
                window = self.config.perceptual_memory.get("window", 64)
                recent_frames = self.config.perceptual_memory.get("recent_frames", 16)
                candidate_img, candidate_pos, candidate_state, candidate_mask = \
                    self.mem_buffer.prepare_candidate_graph_dpp(
                        self.step_idx, window, recent_frames, token_per_image,
                        history_feats_gather_fn)
                inputs["candidate_image_emb"] = candidate_img
                inputs["candidate_pos_emb"] = candidate_pos
                inputs["candidate_state_emb"] = self._normalize_state(candidate_state)
                inputs["candidate_mask"] = candidate_mask
            elif self.config.perceptual_memory.type == "graph_streaming":   # ← 流式图粗化（TCGM v2）
                token_per_image = self.config.token_per_image
                top_k = self.config.perceptual_memory.get("top_k", 32)
                alpha = self.config.perceptual_memory.get("alpha", 0.5)
                tau = self.config.perceptual_memory.get("temporal_tau", 100)
                merge_threshold = self.config.perceptual_memory.get("merge_threshold", 0.7)
                w_gap = self.config.perceptual_memory.get("w_gap", 50)
                w_span = self.config.perceptual_memory.get("w_span", 150)
                c_min = self.config.perceptual_memory.get("c_min", 0.3)
                recency_decay = self.config.perceptual_memory.get("recency_decay", 0.5)
                static_image_emb, static_pos_emb, static_state_emb, static_mask = \
                    self.mem_buffer.prepare_graph_streaming(
                        self.step_idx, token_per_image, top_k, alpha, tau, merge_threshold,
                        w_gap, w_span, c_min, recency_decay,
                        history_feats_gather_fn)
                inputs["static_image_emb"] = static_image_emb
                inputs["static_pos_emb"] = static_pos_emb
                inputs["static_state_emb"] = self._normalize_state(static_state_emb)
                inputs["static_mask"] = static_mask
            else:
                token_per_image = self.config.token_per_image
                static_image_emb, static_pos_emb, static_state_emb, static_mask = \
                    self.mem_buffer.prepare_frame_sampling(
                        self.step_idx, token_budget, token_per_image, history_feats_gather_fn)
                inputs["static_image_emb"] = static_image_emb
                inputs["static_pos_emb"] = static_pos_emb
                inputs["static_state_emb"] = self._normalize_state(static_state_emb)
                inputs["static_mask"] = static_mask
        else:
            raise ValueError(f"Not supported representation type: {self.config.representation_type}")
        
    
        return inputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata