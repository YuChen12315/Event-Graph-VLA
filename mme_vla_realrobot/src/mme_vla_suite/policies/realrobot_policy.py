import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

from mme_vla_suite.models.integration.history_observation import EG_FIELDS

REALROBOT_ACTION_DIM = 42  # 腰 4 + 左臂 7 + 右臂 7 + 左手 12 + 右手 12


def make_realrobot_example() -> dict:
    """真机策略的最小输入示例（head 单图 + 42 维 state）。"""
    return {
        "observation/state": np.random.rand(REALROBOT_ACTION_DIM),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        assert np.abs(image).max() <= 1.0, "Image is not normalized"
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RealRobotInputs(transforms.DataTransformFn):
    """真机 Astribot 的输入：只有 head 相机（stereo 不用）。

    与仿真数据版输入的区别只有图像：不读 `observation/wrist_image`，`image` 字典
    里只放 `base_0_rgb`。模型侧 `preprocess_observation` 按实际存在的键遍历，单图是
    支持的；`image_mask` 也只需要这一个键。
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
            },
            # perceptual memory
            "static_image_emb": data.get("static_image_emb", None),
            "static_pos_emb": data.get("static_pos_emb", None),
            "static_state_emb": data.get("static_state_emb", None),
            "static_mask": data.get("static_mask", None),
            # graph+DPP learned selection (TCGM): candidate window frames
            "candidate_image_emb": data.get("candidate_image_emb", None),
            "candidate_pos_emb": data.get("candidate_pos_emb", None),
            "candidate_state_emb": data.get("candidate_state_emb", None),
            "candidate_mask": data.get("candidate_mask", None),
            # recurrent memory
            "recur_image_emb": data.get("recur_image_emb", None),
            "recur_pos_emb": data.get("recur_pos_emb", None),
            "recur_state_emb": data.get("recur_state_emb", None),
            "recur_mask": data.get("recur_mask", None),
            # symbolic memory
            "simple_subgoal": data.get("simple_subgoal", None),
            "grounded_subgoal": data.get("grounded_subgoal", None),
            "progress": (
                np.asarray(data["progress"], dtype=np.float32)
                if data.get("progress") is not None
                else None
            ),
            **{f: data.get(f, None) for f in EG_FIELDS},
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RealRobotOutputs(transforms.DataTransformFn):
    """推理输出：42 维绝对关节目标。"""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :REALROBOT_ACTION_DIM])}
