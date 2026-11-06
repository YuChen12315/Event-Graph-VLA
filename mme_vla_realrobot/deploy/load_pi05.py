"""按 scripts/train.py 的 init_train_state 初始化 π0.5，并载入 base 或微调权重。

流程与训练入口里 ``model = config.model.create(model_rng)`` 之后的合并一致：
先按配置建出 HistoryPi0（pi05=True），再用 ``replace_by_pure_dict`` 写入权重。
checkpoint 里没有的参数保留这次初始化的值。``freeze_filter`` 命中的参数
（默认是视觉编码器）转成 bfloat16，和训练初始化相同。

在 ``mme_vla_realrobot`` 目录下运行，这样相对路径能对上。

用法
----
只用 π0.5 base 初始化。对应训练的非 resume 分支：``RealRobotWeightLoader``
会把预训练 32 维动作投影补零到真机的 42 维::

    uv run python deploy/load_pi05.py

加载某一步微调权重。对应训练的 resume 分支，目录指向 ``<step>`` 或 ``<step>/params``::

    uv run python deploy/load_pi05.py \\
        --checkpoint-dir runs/ckpts/pi05_realrobot/<exp-name>/<step>

在其他脚本里调用（``mme_vla_realrobot`` 需在 ``PYTHONPATH`` 中）::

    from deploy.load_pi05 import load_pi05_model

    loaded = load_pi05_model(
        config_name="pi05_realrobot",
        checkpoint_dir="runs/ckpts/pi05_realrobot/<exp-name>/<step>",
        seed=42,
    )
    model = loaded.model
    actions = model.sample_actions(jax.random.key(0), observation)

输入
----
config_name:
    训练配置名。默认 ``pi05_realrobot``
    （pi05=True，action_dim=42，action_horizon=20）。
checkpoint_dir:
    微调目录，指向 ``<step>`` 或 ``<step>/params``，也可以是 ``gs://`` 路径。
    省略时使用该配置的 ``weight_loader``，即 ``OPENPI_DATA_HOME`` 下的 pi05_base。
seed:
    初始化随机种子，切分方式与 ``scripts/train.py`` 的 ``main`` 相同。
    只影响未被 checkpoint 覆盖的参数。

输出
----
LoadedPi05
    model: 已写入权重的 HistoryPi0，处于 eval。
        ``sample_actions(rng, observation)`` 接收模型侧的 HistAugObservation，
        返回形状 ``[batch, action_horizon, action_dim]`` 的动作。
    source: ``"base"`` 或 ``"finetune"``。
    params_path: 实际读取的权重目录。
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
from mme_vla_suite.models.integration.history_pi0 import HistoryPi0

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LoadedPi05:
    """``load_pi05_model`` 的输出。字段含义见模块文档。"""

    model: HistoryPi0
    source: str
    params_path: str
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


def load_pi05_model(
    config_name: str = "pi05_realrobot",
    checkpoint_dir: str | None = None,
    seed: int = 42,
) -> LoadedPi05:
    """初始化 π0.5 并写入 base 或微调权重。输入输出见模块文档。"""
    config = _config.get_config(config_name)
    if not getattr(config.model, "pi05", False):
        raise ValueError(f"配置 {config_name} 的 model.pi05 需要为 True。")
    if config.model.use_history or config.model.history_config is not None:
        raise ValueError(f"配置 {config_name} 带记忆模块，本脚本只加载纯 π0.5。")

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
        # 与 train.py init_train_state 内的初始化相同。
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

    n_elem = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model)))
    logger.info("source=%s params=%s", source, params_path)
    logger.info("载入参数叶子 %d / %d", loaded_count, total_count)
    logger.info("Total Model Size: %s MB", n_elem / 1024 / 1024)
    logger.info(
        "pi05=%s action_dim=%s action_horizon=%s",
        config.model.pi05,
        config.model.action_dim,
        config.model.action_horizon,
    )
    return LoadedPi05(
        model=model,
        source=source,
        params_path=params_path,
        loaded_param_count=loaded_count,
        total_param_count=total_count,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="初始化 π0.5 并加载 base 或微调权重。详见文件顶部说明。")
    parser.add_argument("--config-name", default="pi05_realrobot", help="训练配置名，默认 pi05_realrobot")
    parser.add_argument("--checkpoint-dir", default=None, help="微调目录：<step> 或 <step>/params。省略则加载 pi05_base")
    parser.add_argument("--seed", type=int, default=42, help="初始化随机种子")
    args = parser.parse_args()

    loaded = load_pi05_model(
        config_name=args.config_name,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )
    print(
        f"loaded source={loaded.source} params={loaded.params_path} "
        f"leaves={loaded.loaded_param_count}/{loaded.total_param_count} "
        f"action_dim={loaded.model.action_dim} action_horizon={loaded.model.action_horizon}"
    )


if __name__ == "__main__":
    main()
