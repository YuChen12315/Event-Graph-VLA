import os
import json
import logging
import numpy as np
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from pathlib import Path
import random
import re

from openpi.training import config as _config
from openpi.training.data_loader import Dataset
from mme_vla_suite.shared.mem_buffer import MemoryBuffer, MemoryBufferRecurrent
from mme_vla_suite.models.integration.history_observation import EG_FIELDS
import pickle

random.seed(0)
logger = logging.getLogger(__name__)


def load_vector_file(vector_path: str, step_idx: int) -> tuple[dict, int]:
    with open(vector_path, "rb") as f:
        return np.load(f, allow_pickle=True).item(), step_idx
    
    

class SampleDataset(Dataset):
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.stats = json.load(open(os.path.join(self.dataset_path, "meta", "stats.json")))
    
    def __len__(self):
        if "execution_samples" in self.stats:
            return self.stats["execution_samples"]
        else:
            return self.stats["total_samples"]
    
    def __getitem__(self, idx):
        with open(os.path.join(self.dataset_path,  "data", f"{idx}.pkl"), "rb") as f:
            return pickle.load(f)


class RoboMMEDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        data_config: _config.DataConfig,
        history_config: DictConfig | None,
        action_horizon: int,
        compute_norm_stats: bool = False,
    ):
        self.history_config = history_config
        
        
        self.action_horizon = action_horizon
        self.dataset = SampleDataset(dataset_path)
        # 解析成绝对路径：DataLoader worker 里 numpy memmap 打开文件后还会调用
        # os.path.abspath(fid.name) -> getcwd()。进程 cwd 一旦不可用（长期训练、
        # spawn worker、cwd 被删/挂载抖动），这里就会抛一个不带文件名的
        # FileNotFoundError，看起来像「文件缺失」但文件其实在。绝对路径不碰 cwd。
        self.feature_dir = (Path(self.dataset.dataset_path) / "features").resolve()

        # 分段线性进度（离线预计算）：{(epis_idx, step_idx): progress}，存在则覆盖 pkl 里的阶梯进度
        self._progress_linear = None
        linear_path = Path(self.dataset.dataset_path) / "meta" / "progress_linear.pkl"
        if linear_path.exists():
            with open(linear_path, "rb") as f:
                self._progress_linear = pickle.load(f)
            print(f"[dataset] 已加载分段线性进度: {linear_path}")

        # Event-Graph：按 episode 缓存 EventFeatureStore（只为 pca_tokens 查表），
        # 避免每取一个样本都重新校验一遍 cache 元信息。
        self._eg_stores = None
        self._eg_dims = None
        # cfg/PCA 资产只加载一次（4 MB × 1600 集 × 每个 worker；之前每个 episode
        # 首次访问都重读一遍）。
        self._eg_cfg = None
        self._eg_pca = None
        # nodes.npz / steps.npz 每个 episode 解压一次后常驻。NpzFile 的 __getitem__
        # 每次访问键都会重新解压该成员；pack_eg_tensors 按节点读 9 个键 × 32 个节点，
        # 不缓存的话每个样本要解压近 300 次（实测 ~14 ms/样本）。
        self._eg_tables: dict[int, tuple[dict, dict]] = {}
        # 每 episode 的紧凑 payload memmap 句柄（scripts/build_eg_payload_cache.py）。
        # 只缓存句柄，数据页由所有 worker 共享的 OS page cache 提供。
        #
        # 容量要盖住**整个数据集**，不能只留几十集：worker 是全局 shuffle 的，
        # 一个 episode 两次访问之间平均隔着「集数/每集样本数」个样本（1600 集时
        # 远大于 64），LRU 一小就几乎每样本都 miss —— 实测重新 open 三个 mmap
        # 在这套 cluster FS 上要 ~20 ms，比 payload 本身（~5 ms）还贵。
        # 句柄本身很轻：每集 3 个 mmap，1600 集 = 4800 个映射/worker，
        # 本机 max_map_count=1e8、overcommit=1，开销可以忽略。
        self._eg_payload_lru: OrderedDict[int, tuple] = OrderedDict()
        self._eg_payload_lru_size = 4096
        self._eg_payload_version = 1

        # 关键帧选择：从原始阶梯进度差分重建每个 episode 的子目标边界位置（训练用真实边界）
        self._boundary_map = None
        if (
            self.history_config is not None
            and self.history_config.representation_type == "perceptual"
            and self.history_config.perceptual_memory.type == "keyframe_selection"
        ):
            self._boundary_map = self._build_boundary_map()

        if self.history_config is not None:
            self.img_emb_dim = self.history_config.memory_feature.img.input_dim
            self.pos_emb_dim = self.history_config.memory_feature.pos.input_dim
            self.state_emb_dim = self.history_config.memory_feature.state.input_dim
            self.num_views = self.history_config.num_views
            self.streaming_obs_horizon = self.history_config.streaming_obs_horizon
        
            if self.history_config.representation_type == "perceptual":
                self.mem_buffer = MemoryBuffer(
                    num_views=self.num_views,
                    img_emb_dim=self.img_emb_dim,
                    pos_emb_dim=self.pos_emb_dim,
                    state_emb_dim=self.state_emb_dim,
                    token_drop_stride=self.streaming_obs_horizon // 2,
                )
            elif self.history_config.representation_type == "recurrent":
                self.mem_buffer = MemoryBufferRecurrent(
                    num_views=self.num_views,
                    img_emb_dim=self.img_emb_dim,
                    pos_emb_dim=self.pos_emb_dim,
                    state_emb_dim=self.state_emb_dim,
                    input_obs_horizon=self.streaming_obs_horizon,
                    max_recur_steps=self.history_config.recurrent_memory.max_recur_steps,
                    max_video_steps=self.history_config.recurrent_memory.max_pretraj_steps,
                )
            else:
                # symbolic memory does not need mem_buffer
                self.mem_buffer = None
        else:
            logger.info("=== Do not use history ===")
        
        self.compute_norm_stats = compute_norm_stats

        if not compute_norm_stats:
            self.state_norm_stats = data_config.norm_stats['state']
            self.use_quantiles = data_config.use_quantile_norm

    def __getstate__(self):
        """pickle 给 DataLoader worker 时丢掉按 episode 的缓存。

        `_eg_payload_lru` / `_eg_stores` 里是 np.memmap，而 np.memmap 是 ndarray
        子类 —— 默认 pickle 会把**整个映射文件的内容**拷进 pickle。集数一多，
        每个 worker 启动时要传几十上百 GB，表现为「DataLoader 建起来就卡死」。
        这些缓存本来就是进程本地的加速结构，worker 里第一次访问会自己重建。
        """
        state = self.__dict__.copy()
        state["_eg_stores"] = None
        state["_eg_tables"] = {}
        state["_eg_payload_lru"] = OrderedDict()
        state["_eg_cfg"] = None
        state["_eg_pca"] = None
        return state

    def _gather_history_feat(self, indices_to_load: list[int], epis_idx: int):
        history_feats = {}
        history_paths = []
        for idx in indices_to_load:
            history_feats[idx] = {}
            history_paths.append(
                os.path.join(self.feature_dir, f"episode_{epis_idx}", f"token_emb_{idx}.npy")
            )
        
        if self.history_config.representation_type == "recurrent":
            # load_vector_fn = load_vector_file_lru4096 if self.use_lru else load_vector_file
            load_vector_fn = load_vector_file
            max_workers = min(48, max(4, len(history_paths)))
        else:
            # load_vector_fn = load_vector_file_lru1024 if self.use_lru else load_vector_file
            load_vector_fn = load_vector_file
            max_workers = min(36, max(4, len(history_paths)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(load_vector_fn, path, idx): path
                for path, idx in zip(history_paths, indices_to_load)
            }
            for future in as_completed(future_to_path):
                np_dict, idx = future.result()
                history_feats[idx] = np_dict
        return history_feats
    
    
    def prepare_token_drop(self, epis_idx, step_idx):
        token_budget = self.history_config.budget
        kept_indices = json.load(open(os.path.join(self.feature_dir, f"episode_{epis_idx}", "kept_indices.json")))
        
        return self.mem_buffer.prepare_token_dropping(
            step_idx, token_budget, self._gather_history_feat, 
            kept_indices=kept_indices, epis_idx=epis_idx)
        

    def prepare_frame_sampling(self,  epis_idx,  step_idx):
        token_per_image = self.history_config.token_per_image
        token_budget = self.history_config.budget

        return self.mem_buffer.prepare_frame_sampling(
            step_idx, token_budget, token_per_image, self._gather_history_feat, 
            epis_idx=epis_idx)


    # 依旧仿照上一步创建创建属于“16x16”抽帧的函数
    def prepare_frame_sampling_hybrid(self, epis_idx, step_idx):
        token_per_image = self.history_config.token_per_image
        token_budget = self.history_config.budget
        recent_frames = self.history_config.perceptual_memory.get("recent_frames", 16)
        return self.mem_buffer.prepare_frame_sampling_hybrid(
            step_idx, token_budget, token_per_image, recent_frames,
            self._gather_history_feat, epis_idx=epis_idx)


    # 读预计算的 keyframe 记忆（若存在）；不存在返回 None，由调用方回退到现算
    def _load_keyframe_mem(self, epis_idx, step_idx):
        mem_path = self.feature_dir / f"episode_{epis_idx}" / f"keyframe_mem_{step_idx}.npy"
        if not mem_path.exists():
            return None
        with open(mem_path, "rb") as f:
            return np.load(f, allow_pickle=True).item()

    # 读预计算的流式图粗化记忆（若存在）；不存在返回 None，由调用方回退到现算
    def _load_streaming_mem(self, epis_idx, step_idx):
        mem_path = self.feature_dir / f"episode_{epis_idx}" / f"streaming_mem_{step_idx}.npy"
        if not mem_path.exists():
            return None
        with open(mem_path, "rb") as f:
            return np.load(f, allow_pickle=True).item()

    # ══════════════════════════════════════════════════════════════════
    # Event-Graph（v1.3）
    # ══════════════════════════════════════════════════════════════════
    def _eg_paths(self, epis_idx):
        d = self.feature_dir / f"episode_{epis_idx}" / "eventgraph"
        return d, d / "nodes.npz", d / "steps.npz", d / "meta.json"

    def _eg_store(self, epis_idx):
        """按 episode 缓存的 EventFeatureStore（要它的 pca_tokens 查表）。

        ⚠️ 这里读的是**冻结的 PCA token**（512-d），它只是图神经网络的输入；
        代表帧那份 2048-d Visual Payload 是另外从 token_emb 里读的，两者不能混
        —— v1.3 §11 明确禁止拿 512-d cache 顶替 2048-d payload。
        """
        if self._eg_stores is None:
            self._eg_stores = {}
        if self._eg_cfg is None:
            from mme_vla_suite.models.representation.event_graph.config import (
                GraphMemoryConfig,
            )
            from mme_vla_suite.models.representation.event_graph.frame_summary import (
                FrozenFrameSummary,
            )

            self._eg_cfg = GraphMemoryConfig.from_history_config(self.history_config)
            self._eg_pca = FrozenFrameSummary.load(self._eg_cfg.resolve_pca_path())
            self._eg_dims = (
                (int(self._eg_cfg.token_per_image), int(self.history_config.memory_feature.img.input_dim)),
                (int(self._eg_cfg.token_per_image), int(self.history_config.memory_feature.pos.input_dim)),
                (int(self.history_config.memory_feature.state.input_dim),),
            )
        if epis_idx not in self._eg_stores:
            from mme_vla_suite.models.representation.event_graph.store import (
                EventFeatureStore,
            )

            self._eg_stores[epis_idx] = EventFeatureStore(
                str(self.feature_dir), epis_idx, self._eg_cfg, pca=self._eg_pca)
        return self._eg_stores[epis_idx]

    def _eg_raw_frame(self, epis_idx, frame_id):
        """一帧的 (视觉 token [P,d1], 位置编码 [P,d2], 关节角 [d3])，全部是原始精度。

        只读这一个 npy（三个 key 在同一个文件里），不去碰任何"已经投影过"的缓存。
        """
        p = self.feature_dir / f"episode_{epis_idx}" / f"token_emb_{frame_id}.npy"
        with open(p, "rb") as f:
            feat = np.load(f, allow_pickle=True).item()
        img = np.asarray(feat["image_emb_4x4"], np.float32)
        if img.ndim == 3:                      # 部分导出带一个 view 维
            img = img[0]
        pos = np.asarray(feat["pos_emb_4x4"], np.float32)
        if pos.ndim == 3:
            pos = pos[0]
        state = np.asarray(feat["state_emb"], np.float32).reshape(-1)
        want_img, want_pos, want_state = self._eg_dims
        if img.shape != want_img or pos.shape != want_pos or state.shape != want_state:
            raise ValueError(
                f"{p} 的 token 形状是 img{img.shape}/pos{pos.shape}/state{state.shape}，"
                f"而配置要求 {want_img}/{want_pos}/{want_state}。"
                f"多半是 yaml 的 memory_feature 或 token_per_image 与实际导出不一致 ——"
                f"此时模型拿到的是错位的特征，不报错但学不出东西。")
        return img, pos, state

    def _eg_tables_for(self, epis_idx, nodes_path, steps_path):
        """把 nodes.npz / steps.npz 整个解压成普通 dict，并按 episode 常驻。

        NpzFile 不缓存成员：`z["rep_frame"]` 每调一次就重新解压一次。readout 按
        节点逐个字段取值，32 个节点 ≈ 288 次解压，实测 ~14 ms/样本 —— 全部是重复
        工作。两个文件解压后每集只有 ~0.25 MB，1600 集 × 每个 worker 也才几百 MB。
        """
        cached = self._eg_tables.get(epis_idx)
        if cached is not None:
            return cached
        with np.load(nodes_path) as zf:
            z = {k: zf[k] for k in zf.files}
        with np.load(steps_path) as sf:
            s = {k: sf[k] for k in sf.files}
        self._eg_tables[epis_idx] = (z, s)
        return z, s

    def _eg_payload_arrays(self, epis_idx, n_frames=None):
        """每 episode 的紧凑 payload：(img [T,P,d1], pos [T,P,d2], state [T,d3])。

        由 `scripts/build_eg_payload_cache.py` 生成：把 token_emb 里 Event-Graph 真正
        用到的 4x4 字段抽出来按帧号连续存放。没有缓存或形状对不上时返回 None，调用方
        回退到逐帧读 token_emb（正确但慢）。
        """
        hit = self._eg_payload_lru.get(epis_idx)
        if hit is not None:
            self._eg_payload_lru.move_to_end(epis_idx)
            return hit

        # _eg_dims 平时由 _eg_store 初始化；允许单独调用本方法（验证脚本/调试）时
        # 也能正确校验形状，而不是被误判成「缓存不可用」。
        if self._eg_dims is None:
            self._eg_store(epis_idx)

        d = self.feature_dir / f"episode_{epis_idx}" / "eg_payload"
        meta_p = d / "meta.json"
        if not meta_p.exists():
            return None
        try:
            meta = json.loads(meta_p.read_text())
            if int(meta.get("version", -1)) != self._eg_payload_version:
                raise ValueError(f"版本 {meta.get('version')} != {self._eg_payload_version}")
            if n_frames is not None and int(meta["n_frames"]) != int(n_frames):
                raise ValueError(f"缓存帧数 {meta['n_frames']} != steps 帧数 {n_frames}")
            img = np.load(d / "img.npy", mmap_mode="r")
            pos = np.load(d / "pos.npy", mmap_mode="r")
            state = np.load(d / "state.npy", mmap_mode="r")
            if img.dtype != np.float32 or pos.dtype != np.float32 or state.dtype != np.float32:
                raise ValueError(f"dtype 不是 float32: {img.dtype}/{pos.dtype}/{state.dtype}")
            want_img, want_pos, want_state = self._eg_dims
            if (tuple(img.shape[1:]) != want_img or tuple(pos.shape[1:]) != want_pos
                    or tuple(state.shape[1:]) != want_state):
                raise ValueError(
                    f"形状 {img.shape[1:]}/{pos.shape[1:]}/{state.shape[1:]} "
                    f"!= 配置 {want_img}/{want_pos}/{want_state}")
        except Exception as exc:
            logger.warning(
                "episode %d 的 eg_payload 缓存不可用（%s），回退逐帧读取 token_emb。",
                epis_idx, exc)
            return None

        self._eg_payload_lru[epis_idx] = (img, pos, state)
        while len(self._eg_payload_lru) > self._eg_payload_lru_size:
            self._eg_payload_lru.popitem(last=False)
        return img, pos, state

    def _eg_payload_many(self, epis_idx, frame_ids, n_frames=None):
        """一次取多帧 payload（训练侧批量读的入口）。"""
        ids = np.asarray(frame_ids, np.int64).reshape(-1)
        if ids.size == 0:
            want_img, want_pos, want_state = self._eg_dims
            return (np.zeros((0, *want_img), np.float32),
                    np.zeros((0, *want_pos), np.float32),
                    np.zeros((0, *want_state), np.float32))
        if n_frames is not None and (ids.min() < 0 or ids.max() >= n_frames):
            raise IndexError(
                f"episode {epis_idx} 的 frame_id 越界：[{ids.min()}, {ids.max()}]，"
                f"共 {n_frames} 帧")

        arrays = self._eg_payload_arrays(epis_idx, n_frames)
        if arrays is None:
            views = [self._eg_raw_frame(epis_idx, int(f)) for f in ids]
            return (np.stack([v[0] for v in views]),
                    np.stack([v[1] for v in views]),
                    np.stack([v[2] for v in views]))

        img, pos, state = arrays
        return (np.asarray(img[ids], np.float32),
                np.asarray(pos[ids], np.float32),
                np.asarray(state[ids], np.float32))

    def _load_eventgraph_mem(self, epis_idx, step_idx):
        """把离线落盘的「这一帧看见了哪些 Event」展开成模型要的原始节点张量。

        落盘文件只存**选择**（节点表 + 每帧可见的节点 id），几百 KB；2048-d 视觉
        payload 一律现场从 token_emb 读，不复制第二份（§11）。

        拼装本身在 `event_graph/readout.py` —— 在线推理走的是同一个函数。本方法
        只负责「离线这份状态存在哪、怎么读」，两边都不许自己再写一遍拼装。
        """
        from mme_vla_suite.models.representation.event_graph.readout import pack_eg_tensors

        d, nodes_path, steps_path, meta_path = self._eg_paths(epis_idx)
        if not meta_path.exists():
            raise FileNotFoundError(
                f"缺少 EventGraph 离线图状态：{steps_path}。请先跑 "
                f"scripts/precompute_eventgraph_memory.py --history-config "
                f"perceptual-eventgraph-modul.yaml。"
                f"（本类型不做 dataloader 内现算的回退：Controller 是逐帧推进的状态机，"
                f"每个样本从头重放 t 帧就是 O(T²)，会拖死训练。）")

        st = self._eg_store(epis_idx)
        # 三个特征维度取自 yaml 的 memory_feature（与 FeatureEncoder / 模型那边同源）
        dims = (int(self.history_config.memory_feature.img.input_dim),
                int(self.history_config.memory_feature.pos.input_dim),
                int(self.history_config.memory_feature.state.input_dim))

        z, s = self._eg_tables_for(epis_idx, nodes_path, steps_path)
        n_frames = int(s["step_ids"].shape[0])
        row = np.flatnonzero(s["step_ids"] == step_idx)
        if row.size != 1:
            raise ValueError(
                f"episode_{epis_idx}: step_idx={step_idx} 在图状态里出现 {row.size} 次"
                f"（应当恰好 1 次）。多半是 dump 与当前数据集的 step 编号口径不一致。")
        t = int(row[0])

        return pack_eg_tensors(
            cfg=st.cfg,
            node_rows=z,
            vis_ids=s["vis_ids"][s["vis_ptr"][t]:s["vis_ptr"][t + 1]].astype(np.int64),
            recent_ids=s["recent_ids"][t],
            recent_mask=s["recent_mask"][t],
            frame_states=s["states"],
            dims=dims,
            read_payload=lambda rep: self._eg_raw_frame(epis_idx, rep),
            read_payload_many=lambda fids: self._eg_payload_many(
                epis_idx, fids, n_frames),
            read_pca_tokens=st.pca_tokens,
            normalize_state=self._normalize_state,
            exec_start_idx=int(st.exec_start_idx),
        )

    def _load_graphmem_mem(self, epis_idx, step_idx):
        """GraphMem v2 的 32 槽记忆（§15 / C1：槽位 31 = 最新帧，padding 在最老端）。

        由 scripts/precompute_graphmem_memory.py 离线落盘。文件里带 fallback_active，
        供 §19 的辅助诊断统计退回比例。"""
        mem_path = self.feature_dir / f"episode_{epis_idx}" / f"graphmem_mem_{step_idx}.npy"
        if not mem_path.exists():
            return None
        with open(mem_path, "rb") as f:
            return np.load(f, allow_pickle=True).item()

    def _build_boundary_map(self) -> dict:
        """读 meta/boundary_map.pkl（离线脚本存好的）；不存在则回退读所有 pkl 差分重建。"""
        boundary_path = self.feature_dir.parent / "meta" / "boundary_map.pkl"
        if boundary_path.exists():
            with open(boundary_path, "rb") as f:
                return pickle.load(f)

        # 回退：读所有 pkl（原始阶梯进度），按 episode 分组、按 step 排序，差分出边界位置
        progress_by_ep: dict = {}
        for idx in range(len(self.dataset)):
            data = self.dataset[idx]
            epis_idx = int(np.asarray(data["epis_idx"]).item())
            step_idx = int(np.asarray(data["step_idx"]).item())
            progress = float(data["progress"])  # 原始阶梯进度（未覆盖）
            progress_by_ep.setdefault(epis_idx, {})[step_idx] = progress

        boundary_map: dict = {}
        for epis_idx, steps in progress_by_ep.items():
            boundaries = []
            prev = 0.0
            for t in sorted(steps.keys()):
                p = steps[t]
                if p > prev + 1e-6:
                    boundaries.append(t)
                prev = p
            boundary_map[epis_idx] = boundaries
        return boundary_map

    # 关键帧选择：近期帧 + 关键帧（子目标边界）+ 均匀上下文
    def prepare_keyframe_memory(self, epis_idx, step_idx):
        token_per_image = self.history_config.token_per_image
        token_budget = self.history_config.budget
        recent_frames = self.history_config.perceptual_memory.get("recent_frames", 16)
        keyframe_indices = self._boundary_map.get(epis_idx, []) if self._boundary_map is not None else []
        return self.mem_buffer.prepare_keyframe_selection(
            step_idx, token_budget, token_per_image, recent_frames,
            keyframe_indices, self._gather_history_feat, epis_idx=epis_idx)

    # 图 + DPP 记忆选择（TCGM 学习版）：产出候选窗口帧，模型侧做可学习选择
    def prepare_graph_dpp(self, epis_idx, step_idx):
        token_per_image = self.history_config.token_per_image
        window = self.history_config.perceptual_memory.get("window", 64)
        recent_frames = self.history_config.perceptual_memory.get("recent_frames", 16)
        return self.mem_buffer.prepare_candidate_graph_dpp(
            step_idx, window, recent_frames, token_per_image,
            self._gather_history_feat, epis_idx=epis_idx)

    # 流式图粗化记忆（TCGM v2）：重建 [0, step_idx]，逐帧维护 ≤top_k 原型
    def prepare_graph_streaming(self, epis_idx, step_idx):
        token_per_image = self.history_config.token_per_image
        top_k = self.history_config.perceptual_memory.get("top_k", 32)
        alpha = self.history_config.perceptual_memory.get("alpha", 0.5)
        tau = self.history_config.perceptual_memory.get("temporal_tau", 100)
        merge_threshold = self.history_config.perceptual_memory.get("merge_threshold", 0.7)
        w_gap = self.history_config.perceptual_memory.get("w_gap", 50)
        w_span = self.history_config.perceptual_memory.get("w_span", 150)
        c_min = self.history_config.perceptual_memory.get("c_min", 0.3)
        recency_decay = self.history_config.perceptual_memory.get("recency_decay", 0.5)
        return self.mem_buffer.prepare_graph_streaming(
            step_idx, token_per_image, top_k, alpha, tau, merge_threshold,
            w_gap, w_span, c_min, recency_decay,
            self._gather_history_feat, epis_idx=epis_idx)

    def prepare_token_recurrent(self, epis_idx, step_idx, exec_start_idx):
        self.mem_buffer.exec_start_idx = exec_start_idx
        
        return self.mem_buffer.prepare_token_recurrent(
            step_idx, exec_start_idx, self._gather_history_feat,
            epis_idx=epis_idx)
        
    
    def __len__(self):
        return len(self.dataset)
    
    
    def _normalize_state(self, state):
        if self.compute_norm_stats:
            return state
        else:
            if self.use_quantiles:
                return (state - self.state_norm_stats.q01) / (self.state_norm_stats.q99 - self.state_norm_stats.q01 + 1e-6) * 2.0 - 1.0
            else:
                return (state - self.state_norm_stats.mean) / (self.state_norm_stats.std + 1e-6)
    
    def _truncated_gaussian_noise(self, max_val: int, std: float = None) -> int:
        """Generate truncated Gaussian noise centered at 0, clamped to [-max_val, max_val]."""
        if std is None:
            std = max_val / 2.5  # ~95% of samples within range before clamping
        noise = random.gauss(0, std)
        noise = max(-max_val, min(max_val, noise))  # clamp to [-max_val, max_val]
        return int(round(noise))

    def add_grounding_augmentation(self, subgoal: str, noise_range: int = 8) -> str:
        if not subgoal:
            return subgoal
        
        matches = re.findall(r'at <(\d+), (\d+)>', subgoal)
        
        if len(matches) == 0 or len(matches) > 1:
            return subgoal
        
        x, y = matches[0]
        x = int(x)
        y = int(y)
        noise_x = self._truncated_gaussian_noise(noise_range)
        noise_y = self._truncated_gaussian_noise(noise_range)

        new_subgoal = subgoal.replace(f'at <{x}, {y}>', f'at <{x + noise_x}, {y + noise_y}>')
        return new_subgoal
        

    def __getitem__(self, idx):
        data = self.dataset[idx]

        # 用分段线性进度覆盖阶梯进度（若预计算了 progress_linear.pkl）
        if self._progress_linear is not None:
            epis_idx = int(np.asarray(data["epis_idx"]).item())
            step_idx = int(np.asarray(data["step_idx"]).item())
            lp = self._progress_linear.get((epis_idx, step_idx))
            if lp is not None:
                data["progress"] = lp

        data["actions"] = data["actions"][:self.action_horizon]
        
        # During online evaluation, the ground-truth subgoal may change earlier than when it was recorded.
        # To make the model robust to this temporal shift and avoid train/test distribution mismatch,
        # we randomly sample from either subgoal or subgoal_online (early change).
        if self.history_config is not None \
            and self.history_config.representation_type == "symbolic" \
            and self.history_config.symbolic_memory.type in ["simple_subgoal", "grounded_subgoal"] \
            and random.random() < 0.5: 
            data["simple_subgoal"] = data["simple_subgoal_online"]
            data["grounded_subgoal"] = data["grounded_subgoal_online"]
        data.pop("simple_subgoal_online")
        data.pop("grounded_subgoal_online")
        
        if self.history_config is not None and self.history_config.representation_type == "symbolic":
            data["grounded_subgoal"] = self.add_grounding_augmentation(data["grounded_subgoal"], noise_range=8)
            data["simple_subgoal"] = self.add_grounding_augmentation(data["simple_subgoal"], noise_range=8)
 
        
        if self.history_config is not None:
            # use history
            epis_idx = data["epis_idx"].item()
            step_idx = data["step_idx"].item()
            exec_start_idx = data["exec_start_idx"].item()

            if self.history_config.representation_type == "perceptual":
                if self.history_config.perceptual_memory.type == "token_dropping":
                    (
                        static_img_emb,
                        static_pos_emb,
                        static_state_emb,
                        static_mask, # >=64
                    ) = self.prepare_token_drop(epis_idx, step_idx)
                    data["static_image_emb"] = static_img_emb
                    data["static_pos_emb"] = static_pos_emb
                    data["static_state_emb"] = self._normalize_state(static_state_emb)
                    data["static_mask"] = static_mask
                elif self.history_config.perceptual_memory.type == "frame_sampling_hybrid":   # ← 新增分支
                    static_img_emb, static_pos_emb, static_state_emb, static_mask = \
                        self.prepare_frame_sampling_hybrid(epis_idx, step_idx)
                    data["static_image_emb"] = static_img_emb
                    data["static_pos_emb"] = static_pos_emb
                    data["static_state_emb"] = self._normalize_state(static_state_emb)
                    data["static_mask"] = static_mask
                elif self.history_config.perceptual_memory.type == "keyframe_selection":   # ← 关键帧选择
                    mem = self._load_keyframe_mem(epis_idx, step_idx)
                    if mem is not None:
                        # 直接读预计算的 keyframe 记忆（state 下面统一 normalize）
                        data["static_image_emb"] = mem["static_image_emb"]
                        data["static_pos_emb"] = mem["static_pos_emb"]
                        data["static_state_emb"] = self._normalize_state(mem["static_state_emb"])
                        data["static_mask"] = mem["static_mask"]
                    else:
                        # 回退：没有预计算文件时现算
                        static_img_emb, static_pos_emb, static_state_emb, static_mask = \
                            self.prepare_keyframe_memory(epis_idx, step_idx)
                        data["static_image_emb"] = static_img_emb
                        data["static_pos_emb"] = static_pos_emb
                        data["static_state_emb"] = self._normalize_state(static_state_emb)
                        data["static_mask"] = static_mask
                elif self.history_config.perceptual_memory.type in (
                    "graph_dpp",
                    "graph_select",
                ):   # ← 图+DPP / 图选帧(v4)
                    candidate_img, candidate_pos, candidate_state, candidate_mask = \
                        self.prepare_graph_dpp(epis_idx, step_idx)
                    data["candidate_image_emb"] = candidate_img
                    data["candidate_pos_emb"] = candidate_pos
                    data["candidate_state_emb"] = self._normalize_state(candidate_state)
                    data["candidate_mask"] = candidate_mask
                elif self.history_config.perceptual_memory.type == "graphmem":   # ← GraphMem v2（§3 状态机）
                    # 8+24 容量状态机逐帧推进，只在 episode 内有序，必须离线预计算
                    # （每步 O(step) 重放 → 每 episode O(step²)，与 graph_streaming 同一个坑）。
                    mem = self._load_graphmem_mem(epis_idx, step_idx)
                    if mem is None:
                        raise FileNotFoundError(
                            f"缺少 GraphMem 预计算记忆：episode_{epis_idx}/graphmem_mem_{step_idx}.npy。"
                            "请先跑 scripts/precompute_graphmem_memory.py --history-config "
                            "perceptual-graphmem-modul.yaml。"
                            "（本类型不做 dataloader 内现算的回退：状态机每步 O(step)，会拖死训练）")
                    data["static_image_emb"] = mem["static_image_emb"]
                    data["static_pos_emb"] = mem["static_pos_emb"]
                    data["static_state_emb"] = self._normalize_state(mem["static_state_emb"])
                    data["static_mask"] = mem["static_mask"]
                elif self.history_config.perceptual_memory.type == "event_graph":   # ← Event-Graph（v1.3）
                    # 与 graphmem 一样必须离线预计算：Controller 逐帧推进，每步 O(step)。
                    # 但这里回来的**不是**聚合好的 512 个 token —— 24 个 latent slot 对
                    # 节点的读取是可训练的，只能在模型 forward 里算。所以这里给的是
                    # 原始节点张量（v1.3 §7），聚合在 EventGraphMemory 里做。
                    data.update(self._load_eventgraph_mem(epis_idx, step_idx))
                elif self.history_config.perceptual_memory.type == "graph_streaming":   # ← 流式图粗化（TCGM v2）
                    mem = self._load_streaming_mem(epis_idx, step_idx)
                    if mem is not None:
                        # 直接读预计算的流式记忆（state 下面统一 normalize）
                        data["static_image_emb"] = mem["static_image_emb"]
                        data["static_pos_emb"] = mem["static_pos_emb"]
                        data["static_state_emb"] = self._normalize_state(mem["static_state_emb"])
                        data["static_mask"] = mem["static_mask"]
                    else:
                        # 回退：没有预计算文件时现算
                        static_img_emb, static_pos_emb, static_state_emb, static_mask = \
                            self.prepare_graph_streaming(epis_idx, step_idx)
                        data["static_image_emb"] = static_img_emb
                        data["static_pos_emb"] = static_pos_emb
                        data["static_state_emb"] = self._normalize_state(static_state_emb)
                        data["static_mask"] = static_mask
                else:
                    (
                        static_img_emb,
                        static_pos_emb,
                        static_state_emb,
                        static_mask, # slow >=64, mid >=16,fast >=4
                    ) = self.prepare_frame_sampling(epis_idx, step_idx)
                    data["static_image_emb"] = static_img_emb
                    data["static_pos_emb"] = static_pos_emb
                    data["static_state_emb"] = self._normalize_state(static_state_emb)
                    data["static_mask"] = static_mask

            elif self.history_config.representation_type == "recurrent":             
                recur_img_emb, recur_pos_emb, recur_state_emb, recur_mask = (
                    self.prepare_token_recurrent(
                        epis_idx, step_idx, exec_start_idx)
                )

                data["recur_image_emb"] = recur_img_emb
                data["recur_pos_emb"] = recur_pos_emb
                data["recur_state_emb"] = self._normalize_state(recur_state_emb)
                data["recur_mask"] = recur_mask  # 1 ~ max_exec_steps

            elif self.history_config.representation_type == "symbolic":
                pass
            else:
                raise ValueError(
                    f"Not supported representation type: {self.history_config.representation_type}"
                )
        
        
        for key in [
            "static_image_emb",
            "static_pos_emb",
            "static_state_emb",
            "static_mask",
            "candidate_image_emb",
            "candidate_pos_emb",
            "candidate_state_emb",
            "candidate_mask",

            "recur_image_emb",
            "recur_pos_emb",
            "recur_state_emb",
            "recur_mask",
            
            "simple_subgoal",
            "grounded_subgoal",
            "prompt",
            # Event-Graph（v1.3）。⚠️ 必须在这里补 None，和 static_*/recur_* 一个道理：
            # RepackTransform 是白名单，item 里少一个键就是 KeyError，而**不带
            # history_config 的调用方**（scripts/compute_norm_stats.py、可视化脚本等）
            # 本来就不会走 event_graph 分支 —— 之前 eg_* 只进了白名单没进这个名单，
            # 导致 compute_norm_stats 在 16 任务数据集上直接 KeyError('eg_node_bstr')。
            # 模型侧 RealRobotInputs 全是 data.get(f, None)，None 天然表示「这一路没有记忆」。
            *EG_FIELDS,
        ]:
            if key not in data:
                data[key] = None        

        return data