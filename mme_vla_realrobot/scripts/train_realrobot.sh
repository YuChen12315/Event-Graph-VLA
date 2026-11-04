#!/usr/bin/env bash
# 真机数据训练入口。
#   bash scripts/train_realrobot.sh pi05   # π0.5 baseline
#   bash scripts/train_realrobot.sh eg     # perceptual-eventgraph-expert_v2
#
# 前置：先跑 scripts/run_realrobot_pipeline.sh（转换 + 特征 + norm stats）。
# 注意：两处训练都带 --overwrite。train.py 会先跑一段 tentative run，它会建出
#       runs/ckpts/<config>/<exp-name>/；第二次进 main() 时若目录已存在且没给
#       --overwrite/--resume 会直接 FileExistsError。要接着已有 checkpoint 训练，
#       把 --overwrite 换成 --resume --resum-ckpt-id=<step>（优化器状态不恢复）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 默认布局：项目根、Real_Memory_Data/、openpi_data/ 三者同级。
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$ROOT/../openpi_data}"
MODE="${1:-}"
DATA="${DATA:-$ROOT/data/realrobot_preprocessed}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
BATCH="${BATCH:-64}"
FSDP="${FSDP:-8}"

if [ -z "$MODE" ]; then
  echo "用法: $0 [pi05|eg]" >&2
  exit 1
fi

if [ ! -d "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params" ]; then
  echo "错误：找不到 π0.5 权重 $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params" >&2
  echo "      把 openpi_data 放到 $ROOT/..，或 export OPENPI_DATA_HOME=/path/to/openpi_data" >&2
  exit 1
fi
if [ ! -d "$DATA" ]; then
  echo "错误：找不到预处理数据 $DATA" >&2
  echo "      先跑 bash scripts/run_realrobot_pipeline.sh，或 export DATA=/path/to/realrobot_preprocessed" >&2
  exit 1
fi

if [ "$MODE" = "pi05" ]; then
  CUDA_VISIBLE_DEVICES="$GPUS" XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    uv run scripts/train.py pi05_realrobot \
      --exp-name=pi05_realrobot \
      --batch-size="$BATCH" \
      --num-workers=16 \
      --fsdp-devices="$FSDP" \
      --dataset-path="$DATA" \
      --no-wandb-enabled \
      --overwrite
elif [ "$MODE" = "eg" ]; then
  CUDA_VISIBLE_DEVICES="$GPUS" XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
    uv run scripts/train.py mme_vla_realrobot \
      --exp-name=perceptual-eventgraph-expert_v2_realrobot \
      --batch-size="$BATCH" \
      --num-workers=16 \
      --fsdp-devices="$FSDP" \
      --dataset-path="$DATA" \
      --model.use_history \
      --model.history_config=perceptual-eventgraph-expert_v2_realrobot.yaml \
      --no-wandb-enabled \
      --overwrite
else
  echo "用法: $0 [pi05|eg]" >&2
  exit 1
fi
