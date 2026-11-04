import os
from omegaconf import DictConfig


# 历史配置的搜索目录。state_mean/state_std 与 PCA 资产都跟着数据集走，
# 拿错文件不会报错（切分按别的机器人的量纲走），所以只保留真机这一份。
_CONFIG_DIRS = (
    "src/mme_vla_suite/models/config/realrobot",
)


def get_history_config(history_config: str | DictConfig):
    if history_config in ["None", "none"]:
        return None
    if isinstance(history_config, str):
        import omegaconf

        # 绝对路径原样使用：标定/诊断时会从仓库外传配置进来。
        if os.path.isabs(history_config):
            return omegaconf.OmegaConf.load(history_config)

        tried = []
        for config_dir in _CONFIG_DIRS:
            path = os.path.join(config_dir, history_config)
            tried.append(path)
            if os.path.exists(path):
                return omegaconf.OmegaConf.load(path)
        raise FileNotFoundError(
            f"找不到 history config '{history_config}'。已查找：{tried}。"
            f"（相对路径按仓库根目录解析，请在仓库根目录下运行脚本。）")
    elif isinstance(history_config, DictConfig):
        return history_config
    elif history_config is None:
        return None
    else:
        raise ValueError(f"Invalid history config: {history_config}")
