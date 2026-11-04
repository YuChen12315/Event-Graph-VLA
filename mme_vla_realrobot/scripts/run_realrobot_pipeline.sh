#!/usr/bin/env bash
# Astribot 真机数据全流程：原始 LeRobot -> pkl/features -> Event-Graph -> norm stats
#
# 用法：
#   bash scripts/run_realrobot_pipeline.sh                 # A（当前已有的文件夹）
#   FOLDERS="A B" bash scripts/run_realrobot_pipeline.sh    # 多个文件夹
#   EXTRA="--cpu" bash scripts/run_realrobot_pipeline.sh    # GPU 被占时 CPU 编码（慢）
#   BUILD_PAYLOAD=1 bash scripts/run_realrobot_pipeline.sh  # 额外生成 eg_payload 加速缓存
#
# 每一步都可以单独重跑；已转换的 episode 会自动跳过。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 默认布局：项目根、Real_Memory_Data/、openpi_data/ 三者同级。
# 换机器时任意一项放到别处，用同名环境变量覆盖即可。
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$ROOT/../openpi_data}"

RAW="${RAW:-$ROOT/../Real_Memory_Data}"
DATA="${DATA:-$ROOT/data/realrobot_preprocessed}"
FOLDERS="${FOLDERS:-A}"
EXTRA="${EXTRA:-}"
HIST="perceptual-eventgraph-expert_v2_realrobot.yaml"
PCA="$ROOT/runs/assets/event_graph/pca_512_realrobot.npz"

if [ ! -d "$RAW" ]; then
  echo "错误：找不到原始数据目录 $RAW" >&2
  echo "      把 Real_Memory_Data 放到 $ROOT/..，或 export RAW=/path/to/Real_Memory_Data" >&2
  exit 1
fi
if [ ! -d "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params" ]; then
  echo "错误：找不到 π0.5 权重 $OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params" >&2
  echo "      把 openpi_data 放到 $ROOT/..，或 export OPENPI_DATA_HOME=/path/to/openpi_data" >&2
  exit 1
fi

if [ -x .venv/bin/python ]; then
  PY=(.venv/bin/python)
else
  PY=(uv run python)
fi

echo "==> 1/6 转换原始数据"
"${PY[@]}" scripts/build_realrobot_dataset.py \
  --raw-root "$RAW" --dataset-path "$DATA" --folders $FOLDERS --yes $EXTRA

echo "==> 2/6 扫描 exec_start_idx"
"${PY[@]}" scripts/scan_step_index.py --dataset-path "$DATA"

echo "==> 3/6 构建 Event cache + 拟合 PCA（$PCA）"
"${PY[@]}" scripts/build_event_cache.py \
  --dataset-path "$DATA" --history-config "$HIST" --pca-path "$PCA"

echo "==> 4/6 预计算 Event-Graph 记忆"
"${PY[@]}" scripts/precompute_eventgraph_memory.py \
  --dataset-path "$DATA" --history-config "$HIST"

if [ "${BUILD_PAYLOAD:-0}" = "1" ]; then
  echo "==> 4b 可选：生成 eg_payload 加速缓存"
  "${PY[@]}" scripts/build_eg_payload_cache.py --dataset-path "$DATA"
fi

echo "==> 5/6 计算 norm stats（π0.5 baseline）"
"${PY[@]}" scripts/compute_norm_stats.py \
  --config-name pi05_realrobot --repo-id realrobot --dataset-path "$DATA"

echo "==> 6/6 给 Event-Graph 训练复制同一份 norm stats"
mkdir -p runs/assets/mme_vla_realrobot/realrobot
cp -r runs/assets/pi05_realrobot/realrobot/. runs/assets/mme_vla_realrobot/realrobot/

echo
echo "完成。可以开始训练："
echo "  bash scripts/train_realrobot.sh pi05"
echo "  bash scripts/train_realrobot.sh eg"
