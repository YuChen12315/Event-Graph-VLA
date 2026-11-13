"""WebSocket policy server for deployment.

The server owns model loading and inference only.  Camera acquisition, robot
control, receding-horizon execution, recording and keyboard handling remain on
the robot-side client.

Wire format
-----------
Handshake (UTF-8 JSON text, identical to the DECO policy server):
    client -> {"obs": [...], "act": [...]}
    server -> {"ok": true, "contract": {"state_channels": ...,
               "action_channels": ..., "image_channels": ...,
               "chunk_size": ...,
               "action_dim": ...}}

Inference request (binary):
    pack_frame(
        {"task_idx": 1,
         "buffers": [jpeg_bytes, state_bytes, tac1_bytes, tac2_bytes]},
        [stereo_jpeg, state, left_tactile, right_tactile],
    )

``stereo_jpeg`` is the head-camera BGR image as produced by OpenCV.  This server
runs perceptual-eventgraph-expert_v2: the frame is encoded once with SigLIP and
that token is reused both as Event-Graph memory and as the π0.5 prefix.
``state`` is the 42-D Astribot joint vector.  ``--prompt`` is the language
instruction tokenized into that observation.

Response (binary):
    pack_frame({"shape": [action_horizon, action_dim], "buffers": [n]},
               [action_bytes])

Example:
    #开启 JAX 编译缓存（避免每次重启重复编译峰值）
    JAX_ENABLE_COMPILATION_CACHE=1 \
    JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax-compile \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run deploy/policy_server_mme.py

    python deploy/policy_server_mme.py \
        --model-config config/v2/DACO_ft.yaml --host 127.0.0.1 --port 8765
"""

import argparse
import asyncio
import json
import os
import struct
import sys
from datetime import datetime
from functools import wraps
from time import perf_counter
from typing import Iterable

import cv2
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import websockets
import yaml
from pathlib import Path

import openpi.transforms as _transforms
from openpi.shared.nnx_utils import module_jit
from openpi.training import checkpoints as _checkpoints

_CODEBASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_ROOT = os.path.join(_CODEBASE_ROOT, "src")
for _path in (_SRC_ROOT, _CODEBASE_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from utils.camera_preview import CameraPreview
from load_eventgraph import load_eventgraph_model
import mme_vla_suite.training.config as _train_config
from mme_vla_suite.models.integration.history_observation import HistAugObservation


MAX_HEADER_BYTES = 64 * 1024

def timeit(func):
    """Print elapsed time and append it to CSV for later analysis."""
    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = perf_counter() - start
                print(f"[耗时] {func.__qualname__}: {elapsed:.6f} 秒", flush=True)
        return async_wrapper

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = perf_counter() - start
            print(f"[耗时] {func.__qualname__}: {elapsed:.6f} 秒", flush=True)

    return wrapper

def pack_frame(header: dict, buffers: Iterable[bytes]) -> bytes:
    buffers = list(buffers)
    header = dict(header)
    header["buffers"] = [len(buf) for buf in buffers]
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<I", len(header_bytes)) + header_bytes + b"".join(buffers)


def unpack_frame(data: bytes):
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("inference request must be a binary WebSocket frame")
    data = bytes(data)
    if len(data) < 4:
        raise ValueError("frame is shorter than the 4-byte header length")

    (header_len,) = struct.unpack_from("<I", data)
    if header_len <= 0 or header_len > MAX_HEADER_BYTES:
        raise ValueError(f"invalid JSON header length: {header_len}")
    header_end = 4 + header_len
    if header_end > len(data):
        raise ValueError("truncated JSON header")

    header = json.loads(data[4:header_end].decode("utf-8"))
    sizes = header.get("buffers")
    if not isinstance(sizes, list) or any(not isinstance(n, int) or n < 0 for n in sizes):
        raise ValueError("header.buffers must be a list of non-negative byte lengths")

    buffers = []
    pos = header_end
    for size in sizes:
        end = pos + size
        if end > len(data):
            raise ValueError("truncated binary buffer")
        buffers.append(data[pos:end])
        pos = end
    if pos != len(data):
        raise ValueError(f"frame has {len(data) - pos} trailing bytes")
    return header, buffers

def strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _load_norm_stats(checkpoint: str | None, data_config):
    """优先读微调目录里的 assets，否则用训练配置的 runs/assets。"""
    asset_id = data_config.asset_id
    if checkpoint and asset_id and "://" not in str(checkpoint):
        step = Path(checkpoint).expanduser()
        if not step.is_absolute():
            step = Path.cwd() / step
        if step.name == "params":
            step = step.parent
        assets = step / "assets"
        if assets.exists():
            try:
                stats = _checkpoints.load_norm_stats(assets, asset_id)
            except FileNotFoundError:
                stats = None
            if stats is not None:
                return stats
    if data_config.norm_stats is None:
        raise FileNotFoundError(
            "找不到 norm stats。请把 --checkpoint 指到含 assets/ 的 <step>，"
            "或准备 runs/assets/mme_vla_realrobot/realrobot/norm_stats.json。"
        )
    return data_config.norm_stats


def _inference_device():
    """推理设备。没有 GPU 时 π0.5 的 10 步采样不可能到官方的百毫秒级。"""
    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        gpus = []
    if gpus:
        return gpus[0]
    return jax.devices()[0]


def _storage_summary(model) -> tuple[int, int, dict[str, int]]:
    numpy_n = 0
    device_n = 0
    dtypes: dict[str, int] = {}
    for leaf in jax.tree.leaves(nnx.state(model)):
        if isinstance(leaf, np.ndarray):
            numpy_n += 1
        elif isinstance(leaf, jax.Array):
            device_n += 1
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None:
            key = str(dtype)
            dtypes[key] = dtypes.get(key, 0) + 1
    return numpy_n, device_n, dtypes


def _place_inference_params(model, device):
    """把权重放到推理设备，浮点参数统一成 bfloat16。

    ``load_eventgraph_model`` 沿用训练初始化：权重是 numpy，并且只把 ``.*img.*`` 转成
    bfloat16。``module_jit`` 每次调用都会把这份 state 传进 ``jax.jit``。numpy
    数组每次都会整模从 CPU 拷到 GPU（几千个小数组，float32 时约十 GB），所以
    稳态也会落在 1 秒左右。官方 ``create_trained_policy`` 在 restore 时就用
    ``dtype=bfloat16`` 的 device array，计算走 Tensor Core。
    """
    model.eval()
    graphdef, state = nnx.split(model)

    def _leaf(variable):
        value = variable.value
        dtype = getattr(value, "dtype", None)
        if dtype is None:
            return variable
        if jnp.issubdtype(dtype, jnp.floating):
            value = jnp.asarray(value, dtype=jnp.bfloat16, device=device)
        else:
            value = jnp.asarray(value, device=device)
        return variable.replace(value)

    state = state.map(lambda _path, variable: _leaf(variable))
    model = nnx.merge(graphdef, state)
    model.eval()
    jax.block_until_ready(nnx.state(model))
    return model


def _batch_input(value):
    """Host 数组加 batch 维。关闭 x64 时 JAX 会把 float64 收成 float32，这里显式指定，避免统计量把 state 留在 float64。"""
    if value is None or not hasattr(value, "dtype"):
        return value
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32, copy=False)
    return jnp.asarray(array)[np.newaxis, ...]


_HISTORY_CONFIG = "perceptual-eventgraph-expert_v2_realrobot.yaml"


class InferenceServer:
    def __init__(
        self,
        checkpoint: str = None,
        prompt: str = "",
        seed: int = 42,
        history_config: str = _HISTORY_CONFIG,
    ):
        if not prompt or not prompt.strip():
            raise ValueError("π0.5 需要语言指令，请通过 --prompt 传入。")
        self.prompt = prompt.strip()

        loaded = load_eventgraph_model(
            history_config=history_config,
            config_name="mme_vla_realrobot",
            checkpoint_dir=checkpoint,
            seed=seed,
        )
        device = _inference_device()
        before_numpy, before_device, before_dtypes = _storage_summary(loaded.model)
        self.model = _place_inference_params(loaded.model, device)
        _, _, after_dtypes = _storage_summary(self.model)
        self._sample_actions = module_jit(self.model.sample_actions)
        self._vision_encode = module_jit(self.model.vision_encode)
        self._rng = jax.random.key(seed)
        self._seed = seed
        print(
            f"[policy_server] params host_leaves={before_numpy} device_leaves={before_device} "
            f"dtypes {before_dtypes} -> {after_dtypes} on {device}",
            flush=True,
        )
        if device.platform == "cpu":
            print(
                "[policy_server] 当前 JAX 没有 GPU，π0.5 会在 CPU 上跑，每次采样仍会远慢于官方 GPU 推理。",
                flush=True,
            )

        train_config = _train_config.get_config("mme_vla_realrobot")
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        norm_stats = _load_norm_stats(checkpoint, data_config)
        # 与 MME_VLA_Policy.infer 相同：RealRobotInputs → 归一化 → resize / tokenize prompt / pad。
        self._input_transform = _transforms.compose([
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ])
        self._output_transform = _transforms.compose([
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ])

        # 契约通道。bridge 按此顺序拼接 state buffer、按此顺序切片 action 向量，
        # 顺序必须与 scripts/build_realrobot_dataset.py 的 42 维约定一致:
        #   state  = astribot.obs.joint_pos(18) + left_hand.obs.joint_pos(12) + right_hand.obs.joint_pos(12)
        #   action = astribot.act.joint_pos(18) + left_hand.act.joint_pos(12) + right_hand.act.joint_pos(12)
        yaml_data = {
            "state_channels": [
                "astribot.obs.joint_pos",
                "left_hand.obs.joint_pos",
                "right_hand.obs.joint_pos",
            ],
            "action_channels": [
                "astribot.act.joint_pos",
                "left_hand.act.joint_pos",
                "right_hand.act.joint_pos",
            ],
            "image_channels": ["head_camera.obs.rgb"],
        }

        self.state_channels = yaml_data["state_channels"]
        self.action_channels = yaml_data["action_channels"]
        self.image_channels = yaml_data["image_channels"]
        self.chunk_size = 20
        self.action_dim = 42  # 模型输出维度 (body 18 + 左手 12 + 右手 12)
        self.obs_dim = 42
        # bridge 契约维度: astribot.act.joint_pos(23) + 左手(12) + 右手(12) = 47。
        # 模型 42 维在 body 18 之后补 5 个 0 (底盘 3 + 头部 2, 采集中锁死) 下发。
        self.wire_action_dim = self.action_dim + 5

        # CUDA model execution is deliberately serialized.  asyncio may serve
        # multiple sockets, but concurrent calls into one model can cause large
        # latency/memory spikes and offers no batching benefit here.
        self.inference_lock = asyncio.Lock()
        self._state_norm = norm_stats["state"]
        self._use_quantiles = bool(data_config.use_quantile_norm)
        self._open_buffer()
        print(
            f"[policy_server] loaded eventgraph; "
            f"chunk={self.chunk_size}, obs_dim={self.obs_dim}, action_dim={self.action_dim}, "
            f"history_config={history_config!r}, prompt={self.prompt!r}"
        )
        self.preview = None  # optional CameraPreview instance, set by main() if --preview is used
        self._warmup()

    def _normalize_state(self, state):
        stats = self._state_norm
        if self._use_quantiles:
            return (state - stats.q01) / (stats.q99 - stats.q01 + 1e-6) * 2.0 - 1.0
        return (state - stats.mean) / (stats.std + 1e-6)

    def _open_buffer(self) -> None:
        """新建一条空的 Event-Graph 在线缓冲。预热帧不留在正式会话里。"""
        from mme_vla_suite.models.representation.event_graph.config import GraphMemoryConfig
        from mme_vla_suite.models.representation.event_graph.frame_summary import (
            FrozenFrameSummary,
        )
        from mme_vla_suite.models.representation.event_graph.online import (
            EventGraphMemoryBuffer,
        )

        history = self.model.history_config
        cfg = GraphMemoryConfig.from_history_config(history)
        if cfg.token_per_image != 16:
            raise ValueError(
                f"event_graph 在线推理要求 token_per_image=16，得到 {cfg.token_per_image}。"
            )
        pca = FrozenFrameSummary.load(cfg.resolve_pca_path())
        feature = history.memory_feature
        self.mem_buffer = EventGraphMemoryBuffer(
            cfg,
            pca,
            dims=(
                int(feature.img.input_dim),
                int(feature.pos.input_dim),
                int(feature.state.input_dim),
            ),
            normalize_state=self._normalize_state,
            num_views=history.num_views,
            img_emb_dim=int(feature.img.input_dim),
            pos_emb_dim=int(feature.pos.input_dim),
            state_emb_dim=int(feature.state.input_dim),
            vision_enc_fn=self._vision_encode,
        )
        self._step = 0

    def _encode_prefix(self, rgb: np.ndarray):
        """把已经 resize 过的头图编一次 SigLIP。返回 (1, 1, tokens, dim)。"""
        array = np.asarray(rgb)
        if array.dtype == np.uint8:
            array = array.astype(np.float32) / 255.0 * 2.0 - 1.0
        else:
            array = np.asarray(array, dtype=np.float32)
        return self._vision_encode(jnp.asarray(array)[None, None])

    def _push_memory(self, rgb: np.ndarray, raw_state: np.ndarray, emb) -> None:
        pixels = np.asarray(rgb)
        if pixels.dtype != np.uint8:
            pixels = np.clip(np.rint((pixels.astype(np.float32) + 1.0) * 127.5), 0, 255).astype(np.uint8)
        self.mem_buffer.add_buffer(
            pixels[None, None],
            np.asarray(raw_state, dtype=np.float32).reshape(1, -1),
            [self._step],
            precomputed_emb=emb,
        )
        self._step += 1

    def _sample(self, head_rgb: np.ndarray, raw_state: np.ndarray):
        transformed = self._input_transform({
            "observation/image": head_rgb,
            "observation/state": raw_state,
            "prompt": self.prompt,
        })
        state_out = np.asarray(transformed["state"])
        rgb = transformed["image"]["base_0_rgb"]
        emb = self._encode_prefix(rgb)
        self._push_memory(rgb, raw_state, emb)
        transformed.update(self.mem_buffer.memory())
        observation = self._to_observation(transformed)
        self._rng, sample_rng = jax.random.split(self._rng)
        actions = self._sample_actions(
            sample_rng,
            observation,
            prefix_image_tokens={"base_0_rgb": emb[:, 0]},
        )
        return actions, state_out

    def _to_observation(self, transformed: dict) -> HistAugObservation:
        return HistAugObservation.from_dict(jax.tree.map(_batch_input, transformed))

    def _warmup(self) -> None:
        """启动时编译并再跑一次。第一条机器人请求不应承担整网 XLA 编译。"""
        print("[policy_server] compiling sample_actions...", flush=True)
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        state = np.zeros((self.obs_dim,), dtype=np.float32)
        times = []
        for _ in range(2):
            start = perf_counter()
            actions, _ = self._sample(image, state)
            jax.block_until_ready(actions)
            times.append(perf_counter() - start)
        self._open_buffer()
        self._rng = jax.random.key(self._seed)
        print(
            f"[policy_server] warmup compile {times[0]:.3f}s, steady {times[1]:.3f}s",
            flush=True,
        )

    @property
    def contract(self):
        return {
            "state_channels": self.state_channels,
            "action_channels": self.action_channels,
            "image_channels": self.image_channels,
            "chunk_size": self.chunk_size,
            "action_dim": self.wire_action_dim,  # 下发维度 (42 + 5 补位)
        }

    def check_hello(self, hello: dict):
        """Use the same channel availability check as the DECO server."""
        obs_available = set(hello.get("obs", []))
        act_available = set(hello.get("act", []))
        missing = [
            channel
            for channel in self.state_channels + self.image_channels
            if channel not in obs_available
        ]
        missing += [
            channel for channel in self.action_channels if channel not in act_available
        ]
        if missing:
            return f"节点通道不满足 policy 所需 (缺: {missing})"
        return None

    @timeit
    def predict(self, header: dict, buffers: list[bytes]) -> np.ndarray:
        # bridge 每帧发送: image_channels 顺序的 JPEG + 一个拼接 state buffer。
        # 本配置无 tactile, 恒为 [jpeg, state] 两个 buffer。
        expected_buffers = len(self.image_channels) + 1
        if len(buffers) != expected_buffers:
            raise ValueError(
                f"expected {expected_buffers} buffers [images..., state], got {len(buffers)}"
            )
        jpeg_buffer, state_buffer = buffers

        head_bgr = cv2.imdecode(np.frombuffer(jpeg_buffer, np.uint8), cv2.IMREAD_COLOR)
        if head_bgr is None:
            raise ValueError("OpenCV could not decode stereo JPEG")
        if self.preview is not None:
            self.preview.update(head_bgr)
        height, width = head_bgr.shape[:2]
        if height <= 0 or width <= 0 or width % 2:
            raise ValueError(f"invalid side-by-side stereo image shape: {head_bgr.shape}")

        state = np.frombuffer(state_buffer, dtype=np.float64).astype(np.float32)
        # bridge 按契约拼接: astribot 23 + 左手 12 + 右手 12 = 47 维。
        # 训练数据丢弃 astribot 的 18:23 (底盘 3 + 头部 2, 采集中锁死),
        # 裁为 42 = 身体 18 + 左手 12 + 右手 12。若 bridge 已裁好则原样使用。
        if state.size == 47:
            state = np.concatenate([state[:18], state[23:]])
        if state.size != self.obs_dim:
            raise ValueError(f"expected {self.obs_dim} state values, got {state.size}")

        # 纯 π0.5 的 head 图
        head = cv2.cvtColor(head_bgr, cv2.COLOR_BGR2RGB)
        if self.preview is not None:
            self.preview.update(head_bgr)

        # 当前帧只编码一次：SigLIP token 同时写入记忆，并作为前缀，不再进 sample_actions 里重算。
        actions, state_out = self._sample(head, state)
        actions = np.asarray(jax.device_get(actions))[0]
        action = np.asarray(
            self._output_transform({"state": state_out, "actions": actions})["actions"],
            dtype="<f4",
        )
        expected = (self.chunk_size, self.action_dim)
        if action.shape != expected:
            raise RuntimeError(f"model returned action shape {action.shape}, expected {expected}")
        wire_pad = np.asarray([0.0, 0.8, 0.0, 0.0, 0.0], dtype="<f4")
        assert wire_pad.size == self.wire_action_dim - self.action_dim
        action = np.concatenate(
            [
                action[:, :18],
                np.tile(wire_pad, (action.shape[0], 1)),
                action[:, 18:],
            ],
            axis=1,
        )

        return action


async def handle_connection(websocket, core: InferenceServer):
    peer = websocket.remote_address
    try:
        raw_hello = await websocket.recv()
        if not isinstance(raw_hello, str):
            raise ValueError("first message must be a JSON text handshake")
        hello = json.loads(raw_hello)
        error = core.check_hello(hello)
        if error is not None:
            await websocket.send(json.dumps({"ok": False, "error": error}))
            await websocket.close()
            print(f"[policy_server] handshake rejected {peer}: {error}")
            return

        await websocket.send(json.dumps({"ok": True, "contract": core.contract}))
        print(f"[policy_server] client connected: {peer}")

        async for message in websocket:
            try:
                header, buffers = unpack_frame(message)
                async with core.inference_lock:
                    action = await asyncio.to_thread(core.predict, header, buffers)
                await websocket.send(
                    pack_frame(
                        {"shape": list(action.shape)},
                        [action.tobytes(order="C")],
                    )
                )
            except Exception as exc:
                await websocket.send(json.dumps({"ok": False, "error": str(exc)}))
    except Exception as exc:
        try:
            await websocket.send(json.dumps({"ok": False, "error": str(exc)}))
        except websockets.ConnectionClosed:
            pass
    finally:
        print(f"[policy_server] client disconnected: {peer}")


async def main(args):
    core = InferenceServer(
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        history_config=args.history_config,
        seed=args.seed,
    )
    preview_task = None
    if args.preview:
        core.preview = CameraPreview(label="deco_task7_stage1")
        preview_task = asyncio.create_task(core.preview.run())
    try:
        async with websockets.serve(
            lambda websocket: handle_connection(websocket, core),
            args.host,
            args.port,
            max_size=args.max_message_mb * 1024 * 1024,
            compression=None,
            ping_interval=20,
            ping_timeout=20,
            max_queue=2,
        ):
            print(
                f"[policy_server] listening on ws://{args.host}:{args.port} "
            )
            await asyncio.Future()
    finally:
        if preview_task is not None:
            preview_task.cancel()
            await asyncio.gather(preview_task, return_exceptions=True)
        if hasattr(core, "_record_executor"):
            await asyncio.to_thread(core._record_executor.shutdown, wait=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DECO WebSocket policy server")
    parser.add_argument("--model-config", default="")
    parser.add_argument("--checkpoint", type=str, default="/home/zy/zhihao/Event-Graph-VLA/mme_vla_realrobot/runs/ckpts/pi05_realrobot/pi05_yu/39999", help="微调目录：<step> 或 <step>/params")
    parser.add_argument("--history-config", default=_HISTORY_CONFIG, help="Event-Graph yaml，默认真机 perceptual-eventgraph-expert_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-message-mb", type=int, default=20)
    parser.add_argument("--prompt", type=str, default=str("The right hand touches the red area on the trash bin to open the lid, and the left hand puts two red paper balls into the trash bin."), help="Uni task module name, e.g.")
    parser.add_argument("--preview", action="store_true", help="show received stereo frames in a local desktop window")
    asyncio.run(main(parser.parse_args()))
