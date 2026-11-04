# mme_vla_realrobot

Astribot 双臂灵巧手（42 维）真机数据的训练 / 推理代码。仓库里有两条**互相独立**的路线，
可以分别交给不同的人：

| 路线 | 训练配置 | 说明 |
|---|---|---|
| **π0.5 baseline** | `pi05_realrobot` | 纯 π0.5，不接记忆 |
| **perceptual-eventgraph-expert_v2** | `mme_vla_realrobot` | π0.5 + Event-Graph 长时记忆 |

所有命令都在本目录根下执行。

---

# 公共准备（两条路线都要做）

## 1. 环境

要求 Python 3.11 / 3.12，使用 [uv](https://docs.astral.sh/uv/)：

```bash
uv sync
export OPENPI_DATA_HOME=/path/to/openpi_data   # 默认 ../openpi_data
```

`OPENPI_DATA_HOME` 下需要有 π0.5 权重与 tokenizer 资产：

```
openpi_data/
├── openpi-assets/checkpoints/pi05_base/params
├── big_vision/paligemma_tokenizer.model
└── pi05_vision_encoder/siglip_params.pkl
```

推荐三个目录同级：

```
parent/
├── mme_vla_realrobot/   # 本项目
├── Real_Memory_Data/    # 原始数据
└── openpi_data/         # 权重 + tokenizer
```

π0.5 原生动作投影是 32 维，本数据是 42 维；`RealRobotWeightLoader` 会把
`action_in_proj` / `action_out_proj` 零填充到 42，前 32 维沿用预训练权重。
**两条路线都用这个 loader。**

## 2. 原始数据与指令

原始数据是 LeRobot v3.0 目录（`A` … `G`，每个文件夹一个 dataset）。
在原始数据根目录放 `prompts.json`，按文件夹给指令：

```json
{ "A": "The right hand touches the red area on the trash bin to open the lid, and the left hand puts two red paper balls into the trash bin." }
```

缺失时回退为文件夹名；模板见 `examples/realrobot/prompts.example.json`。
**只使用 head 相机**，stereo 不参与。

## 3. 数据准备

两条路线共用同一份转换结果。一键流水线：

```bash
bash scripts/run_realrobot_pipeline.sh                 # 默认 A
FOLDERS="A B" bash scripts/run_realrobot_pipeline.sh   # 多个文件夹
EXTRA="--cpu" bash scripts/run_realrobot_pipeline.sh   # GPU 被占时用 CPU
```

各步骤与两条路线的关系：

| 步骤 | 输出 | π0.5 | Event-Graph |
|---|---|---|---|
| `build_realrobot_dataset.py` | `data/realrobot_preprocessed`（pkl + token_emb + meta） | ✓ | ✓ |
| `scan_step_index.py` | `meta/exec_start_map.json` | | ✓ |
| `build_event_cache.py` | event cache + `runs/assets/event_graph/pca_512_realrobot.npz` | | ✓ |
| `precompute_eventgraph_memory.py` | `features/episode_*/eventgraph/` | | ✓ |
| `compute_norm_stats.py` | `runs/assets/pi05_realrobot/realrobot/norm_stats.json` | ✓ | ✓ |
| 复制 norm stats | `runs/assets/mme_vla_realrobot/realrobot/` | | ✓ |

只做 π0.5 且数据还没转换时，可以只跑转换 + norm stats，省掉 Event-Graph 步骤：

```bash
uv run python scripts/build_realrobot_dataset.py --dataset-path data/realrobot_preprocessed --yes
uv run python scripts/compute_norm_stats.py --config-name pi05_realrobot \
  --repo-id realrobot --dataset-path data/realrobot_preprocessed
```

42 维顺序（state 与 action 相同）：

```
[0:4]   腰 4
[4:11]  左臂 7
[11:18] 右臂 7
[18:30] 左手 12
[30:42] 右手 12
```

训练时前 18 维转 delta，后 24 维保持绝对位置（`make_bool_mask(18, -24)`）。

## 4. 真机客户端接口（两条路线共用）

[examples/realrobot/client.py](examples/realrobot/client.py) 提供 websocket 客户端、
memory 缓冲和完整的控制循环。只需要实现 `RobotIO` 的四个方法：

```python
class MyRobot:  # RobotIO
    def get_head_image(self): ...       # 当前 head RGB，H×W×3，uint8
    def get_state(self): ...            # 42 维 state
    def send_action(self, action): ...  # 单步 42 维绝对关节目标
    def episode_done(self): ...
```

42 维拼接与拆分用 `build_state()` / `split_action()`。跑一个 episode：

```python
with RealRobotPolicyClient("127.0.0.1", 8000) as client:
    run_episode(client, MyRobot(), prompt="count the objects on the table")
```

数据流（30 Hz）：

```
robot.get_head_image() / robot.get_state()
        │
        ├─ client.add_observation()   # 执行 action 期间累积帧
        └─ client.act()               # action chunk 用完时：
              ├─ add_buffer           # 累积帧 → 服务端 memory
              └─ infer                # 当前帧 → (20,42) actions
        │
robot.send_action(action[i])          # 每拍下发一步
```

每个 episode 先 `client.reset()`；`act()` 返回的 action 已由服务端还原成绝对
关节目标。默认执行 chunk 的前 16 步、30 Hz。务必在 `with` / `finally` 中关闭
连接。

---

# 第一部分：π0.5 baseline（负责人 A）

负责 `pi05_realrobot` 的训练和推理，不需要 Event-Graph 特征。

## 1.1 训练

前置数据：`data/realrobot_preprocessed` 与
`runs/assets/pi05_realrobot/realrobot/norm_stats.json`。

```bash
bash scripts/train_realrobot.sh pi05
```

可用环境变量覆盖 `DATA`、`GPUS`、`BATCH`、`FSDP`：

```bash
DATA=data/realrobot_preprocessed GPUS=0,1,2,3,4,5,6,7 BATCH=64 FSDP=8 \
  bash scripts/train_realrobot.sh pi05
```

默认 `batch_size=64`、`num_train_steps=80000`、`fsdp_devices=8`、
`action_horizon=20`、`action_dim=42`。输出：

```
runs/ckpts/pi05_realrobot/<exp-name>/<step>/
```

`<exp-name>` 默认就是 `pi05_realrobot`。

⚠️ 训练命令带 `--overwrite`。`train.py` 会先跑约 11 步的 tentative run 并建出
checkpoint 目录，第二次进 `main()` 时若目录已存在且没有 `--overwrite/--resume`
会 `FileExistsError`。要接着已有 checkpoint 训练，把 `--overwrite` 换成
`--resume --resum-ckpt-id=<step>`（优化器状态不恢复）。

## 1.2 推理

```bash
export OPENPI_DATA_HOME=/path/to/openpi_data

CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port=8000 --seed=7 \
  policy:checkpoint \
  --policy.config=pi05_realrobot \
  --policy.dir=runs/ckpts/pi05_realrobot/<exp-name>/<step>
```

- `--policy.dir` 必须指到 `<exp-name>/<step>`；norm stats 从 checkpoint 的
  `assets/` 读取。
- 推理机只需要：代码 + `openpi_data` + checkpoint。**不需要** `data/` 和
  `runs/assets/`。
- JAX 默认占满所有 GPU，务必设置 `CUDA_VISIBLE_DEVICES`。
- 客户端直接 `run_episode` 即可；π0.5 没有 memory，`add_buffer` 是 no-op，
  不需要额外处理。

---

# 第二部分：perceptual-eventgraph-expert_v2（负责人 B）

负责 `mme_vla_realrobot`。需要公共准备里的 Event-Graph 特征。

## 2.1 训练

前置数据（缺一不可）：

- `data/realrobot_preprocessed`（pkl + token_emb）
- `meta/exec_start_map.json`
- `features/episode_*/event_*` 与 `features/episode_*/eventgraph/`
- `runs/assets/event_graph/pca_512_realrobot.npz`
- `runs/assets/mme_vla_realrobot/realrobot/norm_stats.json`

如果只有原始数据，直接跑完整流水线：

```bash
bash scripts/run_realrobot_pipeline.sh
```

如果 π0.5 那边已经转换过数据，只补 Event-Graph 这几步：

```bash
uv run python scripts/scan_step_index.py --dataset-path data/realrobot_preprocessed
uv run python scripts/build_event_cache.py --dataset-path data/realrobot_preprocessed \
  --history-config perceptual-eventgraph-expert_v2_realrobot.yaml
uv run python scripts/precompute_eventgraph_memory.py \
  --dataset-path data/realrobot_preprocessed \
  --history-config perceptual-eventgraph-expert_v2_realrobot.yaml
mkdir -p runs/assets/mme_vla_realrobot/realrobot
cp -r runs/assets/pi05_realrobot/realrobot/. runs/assets/mme_vla_realrobot/realrobot/
```

训练：

```bash
bash scripts/train_realrobot.sh eg
```

可覆盖 `DATA` / `GPUS` / `BATCH` / `FSDP`，例如：

```bash
DATA=data/realrobot_preprocessed GPUS=0,1,2,3,4,5,6,7 BATCH=64 FSDP=8 \
  bash scripts/train_realrobot.sh eg
```

默认超参与 π0.5 相同。输出：

```
runs/ckpts/mme_vla_realrobot/<exp-name>/<step>/
```

`<exp-name>` 默认是 `perceptual-eventgraph-expert_v2_realrobot`。脚本会自动传
`--model.history_config=perceptual-eventgraph-expert_v2_realrobot.yaml`，训练开始
时把配置名写进 `<exp-name>/history_config.txt`，推理时自动读回。

⚠️ `--overwrite` / `--resume` 的行为与第一部分完全相同。

## 2.2 推理

⚠️ 与 π0.5 不同，EG 推理机必须额外带上：

- `runs/assets/event_graph/pca_512_realrobot.npz`
  （`EventGraphMemoryBuffer` 启动时要加载；路径相对项目根目录）
- `src/mme_vla_suite/models/config/realrobot/perceptual-eventgraph-expert_v2_realrobot.yaml`
  （随代码一起拷，不要删；`history_config.txt` 只存文件名，运行时按项目根目录解析）

然后启动服务：

```bash
export OPENPI_DATA_HOME=/path/to/openpi_data

CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port=8000 --seed=7 \
  policy:checkpoint \
  --policy.config=mme_vla_realrobot \
  --policy.dir=runs/ckpts/mme_vla_realrobot/<exp-name>/<step>
```

客户端 memory 注意事项：

- 每个 episode 先 `client.reset()`，服务端会清空 Event-Graph。
- 执行 action 期间用 `add_observation()` 累积帧；下一次 `act()` 会通过
  `add_buffer` 把这一批帧送进 memory，不要跳过。
- 一个服务端连接同时只跑一个 episode。
- `MemoryBuffer` 位置编码表默认 4096 帧（30 Hz 约 136 s）；更长的 episode
  需要调大 `max_steps`。
- 抓取信号使用 `grasp_mode: hand_closure` + `hand_slice: [18, 30]`（左手 12 维
  均值）；所有任务都是左手抓取，右手不闭合。

---

# 通用注意事项

- **指令**：没放 `prompts.json` 时所有 prompt 都是文件夹名，模型学不到任务区分。
- **单图**：数据转换、客户端和服务端都只处理 head 相机；不要传
  `observation/wrist_image`。
- **Event-Graph 容量**（仅第二部分）：改 `max_nodes` 必须同步改顶层 `budget`
  （`budget = max_nodes*16 + recent_window*16`）。预计算后可用 `steps.npz`
  的 `vis_ptr` 差值量最大并发节点数，接近 `max_nodes` 就调大。
- **时间戳**：训练和推理都按 parquet 行 / 视频帧索引对齐，不要按
  `timestamp_us` 重采样。
- **磁盘**：pkl + features 约每 100 条 22 GB，700 条约 150 GB。
