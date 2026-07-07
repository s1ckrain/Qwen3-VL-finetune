#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export NAVAGENT_QWENVL_DATA_DIR="${NAVAGENT_QWENVL_DATA_DIR:-/root/data1/SFT/qwenvl}"
MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT:-/root/data1/models/lora}"
FORCE_REBUILD_DATA="${FORCE_REBUILD_DATA:-0}"

if [[ "${FORCE_REBUILD_DATA}" == "1" ]] || [[ ! -f "${NAVAGENT_QWENVL_DATA_DIR}/navagent_observing.jsonl" ]]; then
  "${SCRIPT_DIR}/prepare_navagent_data.sh"
fi

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}"
GPU_IDS="${GPU_IDS:-0,1}"

if [[ -n "${GPU_IDS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

DEFAULT_NPROC_PER_NODE="$(python - <<'PY'
import os
gpu_ids = os.environ.get("GPU_IDS", "").strip()
if gpu_ids:
    parts = [p.strip() for p in gpu_ids.split(",") if p.strip()]
    print(len(parts))
else:
    print()
PY
)"

if [[ -n "${DEFAULT_NPROC_PER_NODE}" ]]; then
  NPROC_PER_NODE="${NPROC_PER_NODE:-${DEFAULT_NPROC_PER_NODE}}"
else
  NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}"
fi

USE_DEEPSPEED="${USE_DEEPSPEED:-1}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./scripts/zero3.json}"

MODEL_PATH="${MODEL_PATH:-/root/data/models/Qwen3-VL-8B-Instruct}"
DATASETS="${DATASETS:-navagent_observing,navagent_estimating,navagent_scheduler,navagent_planning}"

# Edit these four defaults directly if you prefer fixed in-file weights.
WEIGHT_OBSERVING="${WEIGHT_OBSERVING:-1}"
WEIGHT_ESTIMATING="${WEIGHT_ESTIMATING:-1}"
WEIGHT_SCHEDULER="${WEIGHT_SCHEDULER:-1}"
WEIGHT_PLANNING="${WEIGHT_PLANNING:-1}"
DEFAULT_DATASET_RESAMPLE_WEIGHTS="observing=${WEIGHT_OBSERVING},estimating=${WEIGHT_ESTIMATING},scheduler=${WEIGHT_SCHEDULER},planning=${WEIGHT_PLANNING}"
DATASET_RESAMPLE_WEIGHTS="${DATASET_RESAMPLE_WEIGHTS:-${DEFAULT_DATASET_RESAMPLE_WEIGHTS}}"

OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_OUTPUT_ROOT}/navagent-lora}"
RUN_NAME="${RUN_NAME:-navagent-lora}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-8192}"
MAX_PIXELS="${MAX_PIXELS:-451584}"
MIN_PIXELS="${MIN_PIXELS:-12544}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
REPORT_TO="${REPORT_TO:-none}"
USE_BF16="${USE_BF16:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
PLOT_LOSS_AFTER_TRAIN="${PLOT_LOSS_AFTER_TRAIN:-1}"

mkdir -p "${OUTPUT_DIR}"

ARGS=(
  --model_name_or_path "${MODEL_PATH}"
  --dataset_use "${DATASETS}"
  --dataset_resample_weights "${DATASET_RESAMPLE_WEIGHTS}"
  --data_flatten False
  --data_packing False
  --tune_mm_vision False
  --tune_mm_mlp True
  --tune_mm_llm True
  --lora_enable True
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --output_dir "${OUTPUT_DIR}"
  --run_name "${RUN_NAME}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --weight_decay 0
  --warmup_ratio 0.03
  --lr_scheduler_type cosine
  --logging_steps "${LOGGING_STEPS}"
  --save_strategy steps
  --save_steps "${SAVE_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --eval_strategy no
  --model_max_length "${MODEL_MAX_LENGTH}"
  --max_pixels "${MAX_PIXELS}"
  --min_pixels "${MIN_PIXELS}"
  --dataloader_num_workers 4
  --report_to "${REPORT_TO}"
)

if [[ "${USE_BF16}" == "1" ]]; then
  ARGS+=(--bf16)
fi

if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  ARGS+=(--gradient_checkpointing True)
else
  ARGS+=(--gradient_checkpointing False)
fi

if [[ "${USE_DEEPSPEED}" == "1" ]]; then
  ARGS+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

echo "Running NavAgent LoRA SFT"
torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  qwenvl/train/train_qwen.py \
  "${ARGS[@]}"

if [[ "${PLOT_LOSS_AFTER_TRAIN}" == "1" ]] && [[ -f "${OUTPUT_DIR}/skill_loss_history.jsonl" ]]; then
  python "${REPO_ROOT}/tools/plot_skill_loss.py" \
    --input "${OUTPUT_DIR}/skill_loss_history.jsonl" \
    --output "${OUTPUT_DIR}/loss_curves.png" \
    --title "NavAgent LoRA Loss Curves"
fi
