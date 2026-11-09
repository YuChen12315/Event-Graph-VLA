"""按 scripts/train.py 的 init_train_state 初始化 perceptual-eventgraph-expert_v2，并载入 base 或微调权重。

流程与训练入口里 ``model = config.model.create(model_rng)`` 之后的合并一致：
先把 ``history_config`` 写进 ``mme_vla_realrobot``，再建出带 Event-Graph 记忆的
HistoryPi0，再用 ``replace_by_pure_dict`` 写入权重。checkpoint 里没有的参数
（主要是记忆模块）保留这次初始化的值。``freeze_filter`` 命中的参数
（默认是视觉编码器）转成 bfloat16，和训练初始化相同。

记忆结构由 yaml 决定。``get_history_config`` 会到
``src/mme_vla_suite/models/config/realrobot/`` 下找这个文件；传入绝对路径则原样读取。
当前先用 ``yourconfig.yaml`` 占位，换成实际配置时只改这个文件名。

在 ``mme_vla_realrobot`` 目录下运行，这样相对路径能对上。

用法
----
只用 π0.5 base 初始化记忆模型。对应训练的非 resume 分支：``RealRobotWeightLoader``
会把预训练 32 维动作投影补零到真机的 42 维，记忆模块保持随机初始化::

    uv run python deploy/load_eventgraph.py \\
        --history-config yourconfig.yaml

加载某一步微调权重。对应训练的 resume 分支，目录指向 ``<step>`` 或 ``<step>/params``::

    uv run python deploy/load_eventgraph.py \\
        --history-config yourconfig.yaml \\
        --checkpoint-dir runs/ckpts/mme_vla_realrobot/<exp-name>/<step>

在其他脚本里调用（``mme_vla_realrobot`` 需在 ``PYTHONPATH`` 中）::

    from deploy.load_eventgraph import load_eventgraph_model

    loaded = load_eventgraph_model(
        history_config="yourconfig.yaml",
        checkpoint_dir="runs/ckpts/mme_vla_realrobot/<exp-name>/<step>",
        seed=42,
    )
    model = loaded.model
    actions = model.sample_actions(jax.random.key(0), observation)

输入
----
history_config:
    Event-Graph 的 yaml。默认 ``yourconfig.yaml``（占位）。
    相对名在 ``src/mme_vla_suite/models/config/realrobot/`` 下查找，绝对路径直接读取。
    需要 ``representation_type=perceptual``、``perceptual_memory.type=event_graph``，
    且 ``streaming_obs_horizon`` 为 16（此时 ``action_horizon`` 必须是 20）。
config_name:
    训练配置名。默认 ``mme_vla_realrobot``
    （pi05=True，use_history=True，action_dim=42，action_horizon=20）。
checkpoint_dir:
    微调目录，指向 ``<step>`` 或 ``<step>/params``，也可以是 ``gs://`` 路径。
    省略时使用该配置的 ``weight_loader``，即 ``OPENPI_DATA_HOME`` 下的 pi05_base。
seed:
    初始化随机种子，切分方式与 ``scripts/train.py`` 的 ``main`` 相同。
    只影响未被 checkpoint 覆盖的参数。

输出
----
LoadedEventGraph
    model: 已写入权重的 HistoryPi0，处于 eval。
        ``sample_actions(rng, observation)`` 接收带 Event-Graph 字段的
        HistAugObservation，返回形状 ``[batch, action_horizon, action_dim]`` 的动作。
    source: ``"base"`` 或 ``"finetune"``。
    params_path: 实际读取的权重目录。
    history_config: 实际使用的 yaml 名或路径。
    loaded_param_count: 从权重文件写入的参数叶子数。
    total_param_count: 模型参数叶子总数。
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import pathlib

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.weight_loaders as _weight_loaders
import mme_vla_suite.training.config as _config
from mme_vla_suite.models.config.utils import get_history_config
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0

logger = logging.getLogger(__name__)

_PLACEHOLDER_HISTORY_CONFIG = "yourconfig.yaml"


@dataclasses.dataclass
class LoadedEventGraph:
    """``load_eventgraph_model`` 的输出。字段含义见模块文档。"""

    model: HistoryPi0
    source: str
    params_path: str
    history_config: str
    loaded_param_count: int
    total_param_count: int


def _load_weights_and_validate(
    loader: _weight_loaders.WeightLoader, params_shape: at.Params
) -> tuple[at.Params, int, int]:
    """载入权重并按训练入口校验结构。返回 (已载入的子集, 载入叶子数, 总叶子数)。"""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # 去掉仍是 ShapeDtypeStruct 的叶子，只把真正读到的权重交回去合并。
    flat = traverse_util.flatten_dict(loaded_params)
    kept = {k: v for k, v in flat.items() if not isinstance(v, jax.ShapeDtypeStruct)}
    return traverse_util.unflatten_dict(kept), len(kept), len(flat)


def _resolve_finetune_params(checkpoint_dir: str) -> str:
    """解析微调目录，返回 params 路径。"""
    if "://" in checkpoint_dir:
        raw = checkpoint_dir.rstrip("/")
        return raw if raw.endswith("/params") else raw + "/params"

    path = pathlib.Path(checkpoint_dir).expanduser()
    if not path.is_absolute():
        path = pathlib.Path.cwd() / path
    step_dir = path.parent if path.name == "params" else path
    params_dir = step_dir / "params"
    if not params_dir.exists():
        raise FileNotFoundError(f"找不到微调参数目录: {params_dir}。请传入 <step> 或 <step>/params。")
    return str(params_dir)


def _require_eventgraph_config(history_config: str, action_horizon: int):
    """按训练入口检查 yaml：16 帧观测窗口，并且是 Event-Graph 感知记忆。"""
    loaded = get_history_config(history_config)
    if loaded is None:
        raise ValueError(f"history config {history_config} 解析为空。perceptual-eventgraph-expert_v2 必须提供 yaml。")
    if int(loaded.streaming_obs_horizon) != 16:
        raise ValueError(
            f"Unsupported streaming_obs_horizon: {loaded.streaming_obs_horizon}。"
            "perceptual-eventgraph-expert_v2 需要 16。"
        )
    if action_horizon != 20:
        raise ValueError("action_horizon must be 20 when streaming_obs_horizon is 16")
    representation = loaded.representation_type
    memory_type = loaded.perceptual_memory.type if representation == "perceptual" else None
    if representation != "perceptual" or memory_type != "event_graph":
        raise ValueError(
            f"history config {history_config} 是 {representation}/{memory_type}，"
            "本脚本只加载 perceptual-eventgraph-expert_v2（perceptual / event_graph）。"
        )
    return loaded


def load_eventgraph_model(
    history_config: str = _PLACEHOLDER_HISTORY_CONFIG,
    config_name: str = "mme_vla_realrobot",
    checkpoint_dir: str | None = None,
    seed: int = 42,
) -> LoadedEventGraph:
    """初始化 Event-Graph 模型并写入 base 或微调权重。输入输出见模块文档。"""
    config = _config.get_config(config_name)
    if not getattr(config.model, "pi05", False):
        raise ValueError(f"配置 {config_name} 的 model.pi05 需要为 True。")
    if not config.model.use_history:
        raise ValueError(f"配置 {config_name} 的 model.use_history 需要为 True。")
    if not history_config:
        raise ValueError("perceptual-eventgraph-expert_v2 必须提供 history_config yaml。")

    _require_eventgraph_config(history_config, config.model.action_horizon)
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, use_history=True, history_config=history_config),
    )

    source = "base"
    if checkpoint_dir is None:
        loader = config.weight_loader
        params_path = str(getattr(loader, "params_path", config.weight_loader))
    else:
        source = "finetune"
        params_path = _resolve_finetune_params(checkpoint_dir)
        loader = _weight_loaders.CheckpointWeightLoader(params_path)

    # 与 scripts/train.py main() 相同的 key 切分，init 只用后半段。
    rng = jax.random.key(seed)
    _train_rng, init_rng = jax.random.split(rng)

    def build(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> HistoryPi0:
        _rng, model_rng = jax.random.split(rng)
        # 与 train.py init_train_state 内的初始化相同。history_config 已在 create 内载入。
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )
        return nnx.merge(nnx.graphdef(model), params)

    abstract_model = nnx.eval_shape(build, init_rng)
    partial_params, loaded_count, total_count = _load_weights_and_validate(
        loader, nnx.state(abstract_model).to_pure_dict()
    )
    model = build(init_rng, partial_params)
    model.eval()
    jax.block_until_ready(nnx.state(model))

    params = nnx.state(model)
    n_elem = sum(x.size for x in jax.tree_util.tree_leaves(params))
    memory_params = params.filter(config.trainable_filter).filter(
        nnx.All(nnx.Param, nnx_utils.PathRegex(".*mem.*"))
    )
    logger.info("source=%s params=%s history_config=%s", source, params_path, history_config)
    logger.info("载入参数叶子 %d / %d", loaded_count, total_count)
    logger.info("Total Model Size: %s MB", n_elem / 1024 / 1024)
    logger.info(
        "Memory-related Size: %s MB",
        sum(x.size for x in jax.tree_util.tree_leaves(memory_params)) / 1024 / 1024,
    )
    logger.info(
        "pi05=%s use_history=%s action_dim=%s action_horizon=%s",
        config.model.pi05,
        config.model.use_history,
        config.model.action_dim,
        config.model.action_horizon,
    )
    return LoadedEventGraph(
        model=model,
        source=source,
        params_path=params_path,
        history_config=history_config,
        loaded_param_count=loaded_count,
        total_param_count=total_count,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(
        description="初始化 perceptual-eventgraph-expert_v2 并加载 base 或微调权重。详见文件顶部说明。"
    )
    parser.add_argument("--config-name", default="mme_vla_realrobot", help="训练配置名，默认 mme_vla_realrobot")
    parser.add_argument(
        "--history-config",
        default=_PLACEHOLDER_HISTORY_CONFIG,
        help="Event-Graph yaml。默认 yourconfig.yaml，放在 src/mme_vla_suite/models/config/realrobot/ 下",
    )
    parser.add_argument("--checkpoint-dir", default=None, help="微调目录：<step> 或 <step>/params。省略则加载 pi05_base")
    parser.add_argument("--seed", type=int, default=42, help="初始化随机种子")
    args = parser.parse_args()

    loaded = load_eventgraph_model(
        history_config=args.history_config,
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )
    print(
        f"loaded source={loaded.source} params={loaded.params_path} "
        f"history_config={loaded.history_config} "
        f"leaves={loaded.loaded_param_count}/{loaded.total_param_count} "
        f"action_dim={loaded.model.action_dim} action_horizon={loaded.model.action_horizon}"
    )


if __name__ == "__main__":
    main()
