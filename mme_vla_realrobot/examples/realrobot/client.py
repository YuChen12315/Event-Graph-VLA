"""Astribot 真机推理客户端示例（π0.5 / perceptual-eventgraph-expert_v2）。

与 RoboMME 评测脚本使用同一套 websocket 协议：

    client.reset()                  # 新 episode，清空服务端 memory
    client.add_buffer({...})        # 把「上次推理之后」的所有帧送入 memory
    client.infer({...})["actions"]  # 返回 (20, 42) 的关节目标

本文件只负责协议与本地缓冲管理，不依赖 Astribot SDK。真机部署时实现
`RobotIO` 的四个方法（读 head 图、读 42 维 state、下发 42 维 action、
判断 episode 是否结束），然后调用 `run_episode()` 即可。

42 维顺序（与 build_realrobot_dataset.py / RealRobotDataConfig 完全一致）：

    state[0:4]    腰部 4
    state[4:11]   左臂 7
    state[11:18]  右臂 7
    state[18:30]  左手 12
    state[30:42]  右手 12

action 的顺序与 state 相同。服务端输出前已经把前 18 维从 delta 还原成
绝对关节目标（AbsoluteActions），后 24 维本来就是绝对位置，因此客户端
可以把 actions 直接交给控制器，不需要再做反归一化。

图像只需要 head 相机（RGB，任意分辨率）；服务端 memory 和模型输入都会
自行 resize_with_pad 到 224×224。左右拼接的 stereo 不参与训练和推理。
"""

from __future__ import annotations

import argparse
import collections
import time
from typing import Protocol

import numpy as np

from openpi_client import websocket_client_policy as _websocket_client_policy

STATE_DIM = 42
ACTION_DIM = 42
DEFAULT_EXEC_HORIZON = 16  # 20 步 chunk 中实际执行的步数，与 RoboMME 评测一致


class RobotIO(Protocol):
    """真机侧需要实现的接口（把下面的方法接到你的控制器/SDK 即可）。"""

    def get_head_image(self) -> np.ndarray:
        """返回当前 head 相机图像，H×W×3，uint8，RGB。"""
        ...

    def get_state(self) -> np.ndarray:
        """返回当前 42 维 state（float32），顺序见文件头注释。"""
        ...

    def send_action(self, action: np.ndarray) -> None:
        """下发单步 42 维绝对关节目标（float32）。"""
        ...

    def episode_done(self) -> bool:
        """本 episode 是否结束（例如计数完成、人工停止或超时）。"""
        ...


def build_state(
    waist: np.ndarray,
    left_arm: np.ndarray,
    right_arm: np.ndarray,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
) -> np.ndarray:
    """按训练时的顺序拼出 42 维 state。各段长度必须分别为 4/7/7/12/12。"""
    parts = [
        np.asarray(waist, dtype=np.float32).reshape(-1),
        np.asarray(left_arm, dtype=np.float32).reshape(-1),
        np.asarray(right_arm, dtype=np.float32).reshape(-1),
        np.asarray(left_hand, dtype=np.float32).reshape(-1),
        np.asarray(right_hand, dtype=np.float32).reshape(-1),
    ]
    expected = (4, 7, 7, 12, 12)
    for name, part, size in zip(
        ("waist", "left_arm", "right_arm", "left_hand", "right_hand"), parts, expected, strict=True
    ):
        if part.size != size:
            raise ValueError(f"{name} 应为 {size} 维，实际 {part.size} 维")
    return np.concatenate(parts).astype(np.float32)


def split_action(action: np.ndarray) -> dict[str, np.ndarray]:
    """把 42 维 action 切成具名关节组，方便下发或记录。"""
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size != ACTION_DIM:
        raise ValueError(f"action 应为 {ACTION_DIM} 维，实际 {action.size} 维")
    return {
        "waist": action[0:4],
        "left_arm": action[4:11],
        "right_arm": action[11:18],
        "left_hand": action[18:30],
        "right_hand": action[30:42],
    }


class RealRobotPolicyClient:
    """带 memory 缓冲管理的 websocket 客户端。

    用法（每个 episode）：

        with RealRobotPolicyClient(host, port) as client:
            run_episode(client, robot, prompt)

    注意 `act()` 会清空本地缓冲；执行 action 期间产生的新帧由
    `run_episode()` 逐帧调用 `add_observation()` 累积，下一次 `act()` 时
    一次性送入服务端 memory。
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        self._policy = _websocket_client_policy.MMEVLAWebsocketClientPolicy(host, port)
        self._images: list[np.ndarray] = []
        self._states: list[np.ndarray] = []

    @property
    def server_metadata(self) -> dict:
        return self._policy.get_server_metadata()

    def reset(self) -> None:
        """开始新 episode：清空服务端 memory 与本地缓冲。"""
        resp = self._policy.reset()
        if not resp.get("reset_finished", False):
            raise RuntimeError(f"服务端 reset 未完成: {resp}")
        self._images.clear()
        self._states.clear()

    def add_observation(self, image: np.ndarray, state: np.ndarray) -> None:
        """缓存一帧观测，等待下一次 act() 一起送入 memory。"""
        image = np.asarray(image)
        if image.dtype != np.uint8:
            raise ValueError(f"image 需要 uint8，实际 {image.dtype}")
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"image 需要 H×W×3，实际 {image.shape}")
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.size != STATE_DIM:
            raise ValueError(f"state 需要 {STATE_DIM} 维，实际 {state.size} 维")
        self._images.append(image)
        self._states.append(state)

    def act(
        self,
        image: np.ndarray,
        state: np.ndarray,
        prompt: str,
        exec_horizon: int = DEFAULT_EXEC_HORIZON,
    ) -> np.ndarray:
        """送入新观测并推理，返回 (exec_horizon, 42) 的绝对关节目标。"""
        self.add_observation(image, state)

        images = np.stack(self._images, axis=0).astype(np.uint8)[:, None]  # (T, 1, H, W, 3)
        states = np.stack(self._states, axis=0).astype(np.float32)         # (T, 42)
        resp = self._policy.add_buffer(
            {
                "images": images,
                "state": states,
                "add_buffer": True,
                "exec_start_idx": 0,  # 真机没有 demo 前缀，全部是执行阶段
            }
        )
        if not resp.get("add_buffer_finished", False):
            raise RuntimeError(f"服务端 add_buffer 未完成: {resp}")

        resp = self._policy.infer(
            {
                "observation/image": np.asarray(image, dtype=np.uint8),
                "observation/state": np.asarray(state, dtype=np.float32),
                "prompt": prompt,
            }
        )

        # 服务端已经持有这些帧的 memory，本地缓冲可以清空。
        self._images.clear()
        self._states.clear()

        actions = np.asarray(resp["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise RuntimeError(f"服务端返回 actions 形状异常: {actions.shape}")
        return actions[:exec_horizon]

    def close(self) -> None:
        self._policy.close()

    def __enter__(self) -> "RealRobotPolicyClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def run_episode(
    client: RealRobotPolicyClient,
    robot: RobotIO,
    prompt: str,
    exec_horizon: int = DEFAULT_EXEC_HORIZON,
    control_hz: float = 30.0,
) -> None:
    """跑一个 episode：推理 → 逐帧执行 → 累积新观测 → 再推理。

    每个控制周期读一帧并缓存；action chunk 用完后重新推理。推理耗时较长时
    会跳过等待（下一拍立刻继续），避免控制循环堆积延迟。
    """
    client.reset()
    action_plan: collections.deque[np.ndarray] = collections.deque()
    period = 1.0 / control_hz
    next_tick = time.monotonic()

    while not robot.episode_done():
        image = robot.get_head_image()
        state = robot.get_state()

        if action_plan:
            client.add_observation(image, state)
        else:
            action_plan.extend(client.act(image, state, prompt, exec_horizon))

        robot.send_action(action_plan.popleft())

        next_tick += period
        delay = next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            next_tick = time.monotonic()


def main() -> None:
    parser = argparse.ArgumentParser(description="真机 websocket 客户端连通性检查")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    with RealRobotPolicyClient(args.host, args.port) as client:
        print("连接成功，服务端 metadata:", client.server_metadata)
        print("实现 RobotIO 后，调用 run_episode(client, robot, prompt) 开始控制。")


if __name__ == "__main__":
    main()
