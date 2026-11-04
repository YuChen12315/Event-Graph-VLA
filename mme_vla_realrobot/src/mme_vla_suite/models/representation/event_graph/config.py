"""GraphMemoryConfig —— Event-Graph 记忆的唯一配置源。

依据 v1.2 §3.2 与 v1.3 §11 的硬性要求：

    预处理脚本、训练 dataset/replay、在线 mem_buffer/policy server、诊断脚本
    **禁止各自定义默认值**，必须从同一个 GraphMemoryConfig 实例读。

因此本模块是 Event-Graph 全部可调参数的唯一定义处；yaml 只负责覆盖，不负责发明。
任何在本文件之外新出现的 `cfg.get("xxx", <字面量>)` 都是违例。

---------------------------------------------------------------------------
⚠️ 边界判定已从 v1.3 §2.2/§2.3 的「三项证据 + 滑动门槛」换成「安静点切分」
---------------------------------------------------------------------------
2026-09-24 实测（4 个 Counting 任务 × 25 条 episode）推翻了原方案的物理前提：

  · 真值 is_subgoal_boundary 标在**机械臂最安静的那一帧**（关节变化量 0.022，
    全 episode 均值 0.095，落在最安静的 10~19% 分位），前后是漂亮的 V 形。
  · 而原方案的 B 是三项「变化越大分越高」，方向正好相反。
  · 结果：原方案命中率 9.1%，比随机放 7 个点（22%）还差；**单段最多跨 8 个子目标**。
  · 「与先前轨迹差异最大」也救不了 —— 实测 4 个任务全在 0.36~0.53 分位（0.5 = 没信号）。
    谷底那一帧恰恰是最"听话"的一帧：它完全符合「减速然后停住」这个预期。

换成「安静点切分」后：单段最多跨 1~2 个子目标，4 个任务全部成立。
详见 controller.py 模块头。旧字段已全部删除 —— 留着会让人以为它们在起作用。
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

# 影响 event cache **内容** 的字段；任一改动都必须重建 cache。
#
# 注意 pca_asset_version 不在这里：那把「尺子」的校验是独立的一层，由
# EventFeatureStore._verify() 拿 meta.pca_asset_version 与当前加载的 PCA 比对。
# 若把它并进指纹，yaml 就必须硬编码 PCA 的哈希，每次重拟合都得改 yaml —— 而重拟合
# 本来只需要换资产文件。两层各管各的，互不依赖。
_CACHE_RELEVANT_FIELDS = ("event_dim", "event_cache_dtype", "token_per_image")

# v1.3 §8 的表格里写的是 "fp16"，但 numpy 1.26 不认这个字符串（np.dtype("fp16") 直接抛
# TypeError）。既然规范用这个写法，就认这个别名，免得每个填 yaml 的人各踩一次。
_DTYPE_ALIASES = {
    "fp16": "float16", "half": "float16", "f2": "float16",
    "fp32": "float32", "float": "float32", "f4": "float32",
    "bf16": "bfloat16",
}


@dataclass(frozen=True)
class GraphMemoryConfig:
    """Event-Graph 记忆配置（v1.3 §8）。

    不可变：Controller 是确定的，配置在运行期被改会破坏 offline/online 一致性。
    """

    # ── 容量与预算（v1.3 §1 / §7，论文结构，不随工程收敛改变）──────────────
    recent_window: int = 8          # Recent raw frames
    max_nodes: int = 96             # Long-Term Event Graph 容量
    # 空槽填充的目标槽位数（事件 + 填充帧）。None ⇒ 填满 max_nodes（旧行为）。
    # 与 max_nodes 解耦：max_nodes 是**容量**（决定 yaml budget 与序列长度），
    # fill_target 只决定「历史不足时往空槽里塞多少真实帧」。RoboMME 的
    # max_nodes=32 且不写这个字段 ⇒ 仍填到 32，行为逐位不变。
    fill_target: int | None = None
    event_len: int = 8              # 每个 Event 交给 Data Plane 的采样帧数 L
    event_dim: int = 512            # PCA cache / Event / Graph 隐层维度 d_g
    token_per_image: int = 16       # 每帧视觉 token 数（= image_emb_4x4 的 16）

    # ── Control Plane（无参数、确定性）──────────────────────────────────
    event_cache_dtype: str = "float16"   # 仅 Event 链路；rep payload 仍走原始 2048-d

    # 边界怎么切：每 cut_block 帧为一个「格子」，在格子里挑「动静最小」的那一帧下刀。
    #
    # 为什么是这个形状（而不是「打一个异常分、超过门槛就切」）：
    #   · 真值边界在机械臂最安静的一帧上，所以「安静」才是信号，「变化大」是反的。
    #   · 格子制让判据变成**本条 episode 内部的相对比较**，不跟全局比 —— 于是
    #     机械臂整体快或整体慢都不影响结果。实测 StopCube 有 69% 的帧是静止的，
    #     用「低于全局中位数」的判据在那儿只切得出 2.6 段、单段最多跨 8 个子目标；
    #     格子制把它压到单段最多跨 2 个。
    #   · 格子制不需要看未来，所以逐帧在线跑和离线整段回放逐位一致（§10）。
    cut_block: int = 30         # 格子大小 ⇒ 平均每 30 帧切一刀
    cut_max_len: int = 50       # 单段最长帧数。超了强制补刀（兜底：格子里可能没有坑）
    cut_smooth: int = 5         # 「动静」信号的滑动平均窗口。**只看过去**，不看未来
    min_event_len: int = 10     # 两刀最小间隔 = 最短 Event 长度；比它近的候选点作废

    # ── 夹爪计数（「数循环次数」这类任务的循环信号）──────────────────────
    # 判据（见 controller._update_grasp_count）：把「夹爪闭合」的连续帧切成一段，
    # 段内夹爪宽度的**最小值** > grasp_gap，才算这一下真的夹住了东西。
    #
    # ⚠️ 为什么必须用「段内最小值」而不是「闭合 且 宽度 > 阈值」：夹爪闭合的**过程**
    #    会路过夹住时的宽度。实测 episode_0 step 243 刚闭合时读数是 0.0204，随后
    #    一路降到 0.0000（夹空）。瞬时判据会把「闭合动作本身」算成一次抓取 ——
    #    该错法在 BinFill 上 0/100 条对得上（报 390 次 vs 真值 278 次）。
    #
    # 取值依据（真机三态实测）：张开 0.0398 ± 0.0009 / 夹住 0.0183 / 夹空 0.0000。
    #   grasp_open = 0.030 —— 张开与夹住的正中，离张开有 ~10σ 余量。
    #                 验证过 T ∈ [0.025, 0.032] 在 4 任务 × 100 条 episode 上**逐条精确**；
    #                 到 0.035（贴近张开值）才开始把一段劈成两段（BinFill 2 条出错）。
    #   grasp_gap  = 0.009 —— 夹住 0.0183 与夹空 0.0000 之间。
    #
    # 这是**夹爪的物理常数**（换机器人要重标），不是能从数据里拟合的图超参。
    # 它只喂一个新标量特征，**不参与切分** —— 改它不会移动任何一条 event 边界。
    #
    # 真机灵巧手没有「夹爪宽度」这一维，用 grasp_mode 切换判据：
    #   "gripper_width"（默认）—— 仿真平行夹爪，state 最后一维是宽度；
    #   "hand_closure"         —— 真机，取 state[hand_slice] 的均值作闭合度
    #                              （越大越闭合），滞回判闭合段。
    # 真机所有任务都是左手抓取、右手不闭合，所以 hand_slice 指左手 12 维。
    grasp_mode: str = "gripper_width"
    grasp_open: float = 0.030
    grasp_gap: float = 0.009
    hand_slice: tuple[int, int] | None = None
    hand_close_threshold: float = 0.40
    hand_open_threshold: float = 0.30

    # 状态标准化（§2.2「先按现有 norm_stats 标准化」）。
    # 放在这里而不是让调用方自己归一化：Controller 是唯一实现，调用方一律传原始 state。
    # 归一化是必要的 —— 不做的话，行程大的关节会盖过行程小的关节。
    # 值来自 runs/assets/pi05_realrobot/realrobot/norm_stats.json 的 norm_stats.state。
    state_mean: tuple[float, ...] | None = None
    state_std: tuple[float, ...] | None = None

    # ── Merge（v1.3 §4）─────────────────────────────────────────────────
    lambda_b: float = 1.0           # C_i = cos(v̄_i, v̄_{i+1}) − λ_b·b_i（b_i 已归一化）

    # ── Data Plane（v1.3 §7）────────────────────────────────────────────
    semantic_topk: int = 4          # Semantic Edge top-k
    gnn_layers: int = 2             # Relation-aware graph attention 层数

    # ── 冻结 PCA 资产 ──────────────────────────────────────────────────
    # 尺子放在哪。None ⇒ 用 runs/assets/event_graph/pca_{event_dim}.npz。
    # 这里存的是**路径**不是版本号：版本号由资产文件内容算出，写死在 yaml 里没有意义。
    pca_asset_path: str | None = None

    # ══════════════════════════════════════════════════════════════════
    def __post_init__(self) -> None:
        # 别名归一化必须在 validate() 之前：validate 里会拿 np.dtype() 验它。
        # 注意 cache_fingerprint() 用归一化后的值，所以 yaml 写 "fp16" 与 "float16"
        # 得到同一个指纹 —— 这是对的，它们本来就是同一个东西。
        norm = _DTYPE_ALIASES.get(str(self.event_cache_dtype).lower())
        if norm is not None and norm != self.event_cache_dtype:
            object.__setattr__(self, "event_cache_dtype", norm)

        # OmegaConf 的 ListConfig 不是 tuple，而 frozen dataclass 要求字段可哈希、
        # 且这里要被 __hash__/eq 使用（诊断脚本会拿两个 cfg 比较）。统一成 tuple。
        for name in ("state_mean", "state_std"):
            v = getattr(self, name)
            if v is not None and not isinstance(v, tuple):
                object.__setattr__(self, name, tuple(float(x) for x in v))
        if self.hand_slice is not None and not isinstance(self.hand_slice, tuple):
            object.__setattr__(self, "hand_slice", tuple(int(x) for x in self.hand_slice))

        self.validate()

    def validate(self) -> None:
        """结构性约束。违例直接报错 —— 静默容忍会让错误配置一路跑到 eval 才暴露。"""
        if self.event_len < 3:
            # §2.3 的采样规则强制保留 start/rep/end 三帧，L<3 无法同时容纳
            raise ValueError(f"event_len 必须 ≥ 3（start/rep/end 三帧必选），得到 {self.event_len}")
        if self.max_nodes < 2:
            raise ValueError(f"max_nodes 必须 ≥ 2 才能做 adjacent merge，得到 {self.max_nodes}")
        if self.fill_target is not None:
            if self.fill_target < 0:
                raise ValueError(f"fill_target 必须 ≥ 0（0 = 不填充），得到 {self.fill_target}")
            if self.fill_target > self.max_nodes:
                raise ValueError(
                    f"fill_target ({self.fill_target}) 不能大于 max_nodes ({self.max_nodes})："
                    f"长期段只有 {self.max_nodes} 个槽。")
        if self.recent_window < 1:
            raise ValueError(f"recent_window 必须 ≥ 1，得到 {self.recent_window}")

        # budget 拆分：memory tokens = max_nodes×P（长期，一个 Event Node 一组）
        #                            + recent_window×P（近期）
        # ⚠️ 这里不再写死 512 / 24 个槽：长期段的宽度现在由 max_nodes 派生
        #    （见 n_long_groups）。唯一的**外部**真值是 yaml 顶层的 `budget`，
        #    由 from_history_config() 与派生值比对 —— 那是唯一能看见外部真值的地方。

        # ── 切分参数之间的制约 ────────────────────────────────────────
        # 这几个数之间只要不自洽，切出来的段就会系统性地偏，而且不会报错。
        if self.cut_block < 2:
            raise ValueError(f"cut_block 必须 ≥ 2（格子里至少要能比较两帧），得到 {self.cut_block}")
        if self.cut_smooth < 1:
            raise ValueError(f"cut_smooth 必须 ≥ 1，得到 {self.cut_smooth}")
        if self.cut_smooth > self.cut_block:
            # 平滑窗比格子还长 ⇒ 第一个格子里全是「不足窗口」的残缺平均，
            # 格子之间的可比性直接没了。
            raise ValueError(
                f"cut_smooth ({self.cut_smooth}) 不能大于 cut_block ({self.cut_block})")
        if self.min_event_len < 2:
            raise ValueError(f"min_event_len 必须 ≥ 2，得到 {self.min_event_len}")
        if self.cut_max_len < self.min_event_len:
            # 封顶比最小间隔还小时，两条规则会互相打架：切完立刻又要切。
            raise ValueError(
                f"cut_max_len ({self.cut_max_len}) 不能小于 min_event_len ({self.min_event_len})")
        if self.cut_max_len < self.cut_block:
            # 封顶通常会先于格子触发，格子制就没意义了 —— 那是配置错了，不是设计选择。
            raise ValueError(
                f"cut_max_len ({self.cut_max_len}) 不能小于 cut_block ({self.cut_block})，"
                f"否则封顶会一直在格子之前触发")

        # 状态标准化的 mean/std 必须成对且同长 —— 长度不匹配要在这里炸，不能等广播
        for name in ("state_mean", "state_std"):
            v = getattr(self, name)
            if v is not None and len(v) < 1:
                raise ValueError(f"{name} 不能是空序列")
        if (self.state_mean is None) != (self.state_std is None):
            raise ValueError("state_mean/state_std 必须同时提供")
        if self.state_mean is not None and len(self.state_mean) != len(self.state_std):
            raise ValueError(
                f"state_mean 长度 {len(self.state_mean)} != state_std 长度 "
                f"{len(self.state_std)}")

        # 夹爪判据的两个门槛：夹住的门槛必须低于「张开」的门槛，否则「闭合段」永远是
        # 空的、计数恒为 0 —— 而 0 是个合法读数（StopCube 真值就是 0），不会报错。
        if self.grasp_gap <= 0.0:
            raise ValueError(f"grasp_gap 必须 > 0，得到 {self.grasp_gap}")
        if self.grasp_open <= self.grasp_gap:
            raise ValueError(
                f"grasp_open ({self.grasp_open}) 必须大于 grasp_gap ({self.grasp_gap})，"
                f"否则任何一帧都进不了「闭合段」，计数恒为 0")

        # 真机灵巧手模式：切片必须给出且落在 state 内；滞回门槛必须一高一低。
        if self.grasp_mode not in ("gripper_width", "hand_closure"):
            raise ValueError(
                f"grasp_mode 必须是 'gripper_width' 或 'hand_closure'，得到 {self.grasp_mode!r}")
        if self.grasp_mode == "hand_closure":
            if self.hand_slice is None:
                raise ValueError("grasp_mode=hand_closure 必须给 hand_slice，例如 [18, 30]")
            lo, hi = self.hand_slice
            if not (0 <= lo < hi):
                raise ValueError(f"hand_slice 必须是 [lo, hi) 且 lo < hi，得到 {self.hand_slice}")
            if self.state_mean is not None and hi > len(self.state_mean):
                raise ValueError(
                    f"hand_slice 上界 {hi} 超出 state 维度 {len(self.state_mean)}")
            if self.hand_open_threshold >= self.hand_close_threshold:
                raise ValueError(
                    f"hand_open_threshold ({self.hand_open_threshold}) 必须小于 "
                    f"hand_close_threshold ({self.hand_close_threshold})，否则滞回不成立")

        # dtype 必须在 numpy 里真实存在且是浮点 —— 写错的代价是 cache 全废
        import numpy as np
        try:
            dt = np.dtype(self.event_cache_dtype)
        except TypeError as e:
            raise ValueError(f"event_cache_dtype={self.event_cache_dtype!r} 不是合法 numpy dtype") from e
        if not np.issubdtype(dt, np.floating):
            raise ValueError(f"event_cache_dtype 必须是浮点，得到 {self.event_cache_dtype}")

        # （原 `assert budget == 512` 已删除：它算的是自己局部的 24 槽，改 max_nodes
        #   时它照样通过 —— 一条不会失败的断言等于没有，只会给人「已经查过了」的错觉。
        #   yaml 与派生值的比对移到了 from_history_config()。）

    # ── 派生量 ─────────────────────────────────────────────────────────
    @property
    def n_long_groups(self) -> int:
        """长期段分组数：一个 Event Node 一组 P 个 token（v1.4 起不再有 latent slot）。"""
        return self.max_nodes

    @property
    def budget(self) -> int:
        return (self.n_long_groups * self.token_per_image
                + self.recent_window * self.token_per_image)

    def cache_fingerprint(self) -> str:
        """event cache 的语义指纹。

        只覆盖影响 **cache 内容** 的字段。boundary 阈值、merge 权重等改了不需要重建
        cache，所以不进指纹 —— 否则每调一次参就要重跑 3.5 GB 的读写。
        """
        payload = {k: getattr(self, k) for k in _CACHE_RELEVANT_FIELDS}
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def resolve_pca_path(self, root: str = "runs/assets/event_graph") -> str:
        """冻结 PCA 资产的实际路径。"""
        if self.pca_asset_path:
            return self.pca_asset_path
        return os.path.join(root, f"pca_{self.event_dim}.npz")

    # ── 序列化 ─────────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GraphMemoryConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            # 拼错字段名是最常见的静默失败：yaml 里写错了、配置被忽略、跑出来是默认值。
            # 宁可在这里炸掉。
            raise ValueError(
                f"GraphMemoryConfig 不认识这些字段：{sorted(unknown)}。"
                f"合法字段：{sorted(known)}")
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_history_config(cls, history_config: Any) -> "GraphMemoryConfig":
        """从 get_history_config() 载入的 OmegaConf/yaml 构造。

        期望的 yaml 结构（其余顶层字段由 history_pi0 消费，这里只用需要的）：

            representation_type: perceptual
            integration_type: modulation
            budget: 512
            token_per_image: 16
            perceptual_memory:
              type: event_graph
              ... <GraphMemoryConfig 的字段> ...

        顶层与 perceptual_memory 下的字段会合并；顶层优先用于 token_per_image，
        这样与仓库其它 memory 类型的写法保持一致。
        """
        if history_config is None:
            raise ValueError("history_config 为 None，无法构造 GraphMemoryConfig")

        def _get(container: Any, key: str, default: Any = None) -> Any:
            if container is None:
                return default
            if hasattr(container, "get"):
                return container.get(key, default)
            return getattr(container, key, default)

        pm = _get(history_config, "perceptual_memory")
        if pm is None:
            raise ValueError(
                "Event-Graph 需要 history_config.perceptual_memory，但该字段缺失。"
                "检查 yaml 是否用了 perceptual-eventgraph-modul.yaml。")

        merged: dict[str, Any] = dict(pm)
        # 顶层 token_per_image 生效（与 graphmem / frame_sampling 的既有写法一致）
        for key in ("token_per_image", "budget"):
            top = _get(history_config, key)
            if top is not None:
                merged[key] = top

        # 这两个不是 GraphMemoryConfig 的字段：
        #   type   —— 路由键，由 history_pi0 / dataset / policy 的分派逻辑消费
        #   budget —— 派生量（max_nodes×P + recent×P），由 budget property 自己算
        declared_type = merged.pop("type", None)
        if declared_type is not None and declared_type != "event_graph":
            raise ValueError(
                f"perceptual_memory.type={declared_type!r}，但本模块只处理 "
                f"'event_graph'。检查 yaml 是否与 history_config 参数配错。")
        # yaml 里的 budget 不当字段用，但必须与派生值一致：max_nodes 改了、yaml 没改，
        # 长期段宽度就会和帧预算静默脱钩（要等 percep_mem.py 的 assert 在第一次 forward
        # 才炸）。在这里比，是最早能看到外部真值的地方。
        declared_budget = merged.pop("budget", None)
        cfg = cls.from_dict(merged)
        if declared_budget is not None and int(declared_budget) != cfg.budget:
            raise ValueError(
                f"yaml budget={declared_budget} 与派生预算 {cfg.budget} 不一致："
                f"max_nodes {cfg.max_nodes}×{cfg.token_per_image} + "
                f"recent_window {cfg.recent_window}×{cfg.token_per_image}。"
                f"改 max_nodes 时必须同步改 yaml 顶层的 budget。")
        return cfg
