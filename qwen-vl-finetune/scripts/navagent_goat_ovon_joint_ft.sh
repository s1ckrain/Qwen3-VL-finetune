#!/bin/bash
# NavAgent GOAT + OVON 联合全参数 SFT（独立训练入口）

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- 模型、数据与输出路径 ---------------------------------------------------
# 可在启动命令前覆盖 MODEL_PATH，以选择 2B/4B 或已有 SFT 权重。
: "${MODEL_PATH:=/root/data/models/Qwen3-VL-2B-Instruct}"
: "${NAVAGENT_GOAT_QWENVL_DATA_DIR:=/root/data1/SFT-goat/qwenvl}"
: "${NAVAGENT_OVON_QWENVL_DATA_DIR:=/root/data1/SFT-ovon/qwenvl}"
: "${MODEL_OUTPUT_ROOT:=/root/data/train/SFT-goat-ovon-2B-1.5M}"
: "${OUTPUT_DIR:=${MODEL_OUTPUT_ROOT}}"
export NAVAGENT_GOAT_QWENVL_DATA_DIR
export NAVAGENT_OVON_QWENVL_DATA_DIR

# ---- 数据组合 ---------------------------------------------------------------
# 默认自然联合六个数据源；不传 dataset_resample_weights，因此每个被选中的
# JSONL 条目在每个 epoch 中进入训练集一次，不做 skill/domain 上采样。
: "${DATA_PERCENT:=100}"
: "${DATASETS:=}"
: "${SKILL_LOSS_WEIGHTS:=observing=1,estimating=1,planning=1}"

# ---- 训练参数 ---------------------------------------------------------------
: "${NUM_TRAIN_EPOCHS:=2}"
: "${MAX_STEPS:=-1}"
: "${LEARNING_RATE:=1e-5}"
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=4}"
: "${GRADIENT_ACCUMULATION_STEPS:=2}"
: "${MODEL_MAX_LENGTH:=16384}"
: "${MAX_PIXELS:=307200}"
: "${MIN_PIXELS:=12544}"
: "${WARMUP_RATIO:=0.10}"
: "${WEIGHT_DECAY:=0}"
: "${LR_SCHEDULER_TYPE:=cosine}"
: "${SEED:=42}"

# ---- 模型可训练模块 ---------------------------------------------------------
: "${TUNE_MM_VISION:=True}"
: "${TUNE_MM_MLP:=True}"
: "${TUNE_MM_LLM:=True}"
: "${USE_BF16:=1}"
: "${GRADIENT_CHECKPOINTING:=1}"
: "${USE_DEEPSPEED:=1}"
: "${DEEPSPEED_CONFIG:=./scripts/zero3.json}"

# ---- 日志 / checkpoint ------------------------------------------------------
: "${LOGGING_STEPS:=20}"
: "${SAVE_STRATEGY:=steps}"
: "${SAVE_STEPS:=2000}"
: "${SAVE_TOTAL_LIMIT:=2}"
# 空值：新训练；latest：从 OUTPUT_DIR 最新 checkpoint 恢复；也可给具体路径。
: "${RESUME_FROM_CHECKPOINT:=}"
: "${REPORT_TO:=none}"
: "${RUN_NAME:=navagent-sft-goat-ovon-2B-1.5M-p${DATA_PERCENT}}"
: "${PLOT_LOSS_AFTER_TRAIN:=1}"
: "${DATALOADER_NUM_WORKERS:=8}"
: "${DATALOADER_PREFETCH_FACTOR:=4}"

# ---- GOAT / OVON 分域周期评测 ---------------------------------------------
# 每域每 skill 取 96 条；两个域合计的生成量约等于原单域 192 条配置。
: "${EVAL_STEPS:=2000}"
: "${EVAL_NUM_SAMPLES_PER_SKILL:=96}"
: "${EVAL_MAX_NEW_TOKENS:=256}"
: "${EVAL_SKILLS:=estimating,planning}"
: "${EVAL_VAL_DIR:=goat=${NAVAGENT_GOAT_QWENVL_DATA_DIR},ovon=${NAVAGENT_OVON_QWENVL_DATA_DIR}}"
: "${EVAL_ON_START:=1}"
: "${EVAL_TRAIN_PROBE_SIZE:=96}"
: "${EVAL_PIXEL_TOLERANCE:=64}"

# ---- 分布式 -----------------------------------------------------------------
: "${GPU_IDS:=0,1}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=$(shuf -i 20001-29999 -n 1)}"

# ============================================================================
# 以下为启动逻辑
# ============================================================================

if ! [[ "${DATA_PERCENT}" =~ ^[0-9]+$ ]] \
   || (( DATA_PERCENT < 1 || DATA_PERCENT > 100 )); then
  echo "DATA_PERCENT 必须是 [1, 100] 之间的整数, 当前: ${DATA_PERCENT}" >&2
  exit 1
fi

# 六个训练文件必须存在；启用周期评测时，两域的 estimating/planning val
# 也必须存在。
for _root in \
  "${NAVAGENT_GOAT_QWENVL_DATA_DIR}" \
  "${NAVAGENT_OVON_QWENVL_DATA_DIR}"; do
  for _skill in observing estimating planning; do
    _train_file="${_root}/navagent_${_skill}_train.jsonl"
    if [[ ! -s "${_train_file}" ]]; then
      echo "ERROR: missing or empty joint-training dataset: ${_train_file}" >&2
      exit 1
    fi
  done
  if (( EVAL_STEPS > 0 )); then
    for _skill in estimating planning; do
      _val_file="${_root}/navagent_${_skill}_val.jsonl"
      if [[ ! -s "${_val_file}" ]]; then
        echo "ERROR: missing or empty joint-eval dataset: ${_val_file}" >&2
        exit 1
      fi
    done
  fi
done

if [[ "${USE_DEEPSPEED}" == "1" && ! -f "${DEEPSPEED_CONFIG}" ]]; then
  echo "ERROR: DeepSpeed config not found: ${DEEPSPEED_CONFIG}" >&2
  exit 1
fi

# GPU 设置
if [[ -n "${GPU_IDS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  IFS=',' read -ra _gpu_arr <<< "${GPU_IDS}"
  : "${NPROC_PER_NODE:=${#_gpu_arr[@]}}"
else
  : "${NPROC_PER_NODE:=$(nvidia-smi --list-gpus | wc -l)}"
fi

# 构造六个独立数据集名称。DATA_PERCENT<100 时，每个数据源独立下采样；
# 默认 100%，即全部样本自然联合。
if [[ -z "${DATASETS}" ]]; then
  _joint_datasets=(
    navagent_goat_observing_train
    navagent_goat_estimating_train
    navagent_goat_planning_train
    navagent_ovon_observing_train
    navagent_ovon_estimating_train
    navagent_ovon_planning_train
  )
  _dataset_parts=()
  for _dataset in "${_joint_datasets[@]}"; do
    if (( DATA_PERCENT >= 100 )); then
      _dataset_parts+=("${_dataset}")
    else
      _dataset_parts+=("${_dataset}%${DATA_PERCENT}")
    fi
  done
  DATASETS="$(IFS=,; echo "${_dataset_parts[*]}")"
fi

mkdir -p "${OUTPUT_DIR}"

ARGS=(
  --model_name_or_path "${MODEL_PATH}"
  --dataset_use "${DATASETS}"
  --skill_loss_weights "${SKILL_LOSS_WEIGHTS}"
  --data_flatten False
  --data_packing False
  --lora_enable False
  --tune_mm_vision "${TUNE_MM_VISION}"
  --tune_mm_mlp "${TUNE_MM_MLP}"
  --tune_mm_llm "${TUNE_MM_LLM}"
  --output_dir "${OUTPUT_DIR}"
  --run_name "${RUN_NAME}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_ratio "${WARMUP_RATIO}"
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}"
  --logging_steps "${LOGGING_STEPS}"
  --save_strategy "${SAVE_STRATEGY}"
  --model_max_length "${MODEL_MAX_LENGTH}"
  --max_pixels "${MAX_PIXELS}"
  --min_pixels "${MIN_PIXELS}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}"
  --dataloader_pin_memory True
  --report_to "${REPORT_TO}"
  --seed "${SEED}"
)

[[ "${USE_BF16}" == "1" ]] && ARGS+=(--bf16)
[[ "${GRADIENT_CHECKPOINTING}" == "1" ]] \
  && ARGS+=(--gradient_checkpointing True) \
  || ARGS+=(--gradient_checkpointing False)
[[ "${USE_DEEPSPEED}" == "1" ]] && ARGS+=(--deepspeed "${DEEPSPEED_CONFIG}")
(( MAX_STEPS > 0 )) && ARGS+=(--max_steps "${MAX_STEPS}")

if [[ "${SAVE_STRATEGY}" != "no" ]]; then
  [[ "${SAVE_STRATEGY}" == "steps" ]] && ARGS+=(--save_steps "${SAVE_STEPS}")
  ARGS+=(--save_total_limit "${SAVE_TOTAL_LIMIT}")
fi

# checkpoint 恢复
_resume_path=""
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  if [[ "${RESUME_FROM_CHECKPOINT}" == "latest" ]]; then
    _latest=$(ls -1d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null \
              | awk -F'checkpoint-' '{print $2"\t"$0}' \
              | sort -k1 -n | tail -n1 | cut -f2 || true)
    if [[ -n "${_latest}" ]]; then
      _resume_path="${_latest}"
      echo ">>> RESUME_FROM_CHECKPOINT=latest -> ${_resume_path}"
    else
      echo ">>> RESUME_FROM_CHECKPOINT=latest 但 ${OUTPUT_DIR} 无 checkpoint-*, 从头训"
    fi
  elif [[ -d "${RESUME_FROM_CHECKPOINT}" ]]; then
    _resume_path="${RESUME_FROM_CHECKPOINT}"
  else
    echo "ERROR: RESUME_FROM_CHECKPOINT 路径不存在: ${RESUME_FROM_CHECKPOINT}" >&2
    exit 1
  fi
  [[ -n "${_resume_path}" ]] && ARGS+=(--resume_from_checkpoint "${_resume_path}")
fi

# 多域周期评测
if (( EVAL_STEPS > 0 )); then
  ARGS+=(
    --eval_strategy steps
    --eval_steps "${EVAL_STEPS}"
    --eval_val_dir "${EVAL_VAL_DIR}"
    --eval_skills "${EVAL_SKILLS}"
    --eval_num_samples_per_skill "${EVAL_NUM_SAMPLES_PER_SKILL}"
    --eval_max_new_tokens "${EVAL_MAX_NEW_TOKENS}"
    --eval_train_probe_size "${EVAL_TRAIN_PROBE_SIZE}"
    --eval_planning_pixel_tolerance "${EVAL_PIXEL_TOLERANCE}"
  )
  [[ "${EVAL_ON_START}" == "1" ]] && ARGS+=(--eval_on_start)
else
  ARGS+=(--eval_strategy no)
fi

cat <<BANNER
================================================================================
  NavAgent GOAT + OVON Joint Full FT
--------------------------------------------------------------------------------
  model          : ${MODEL_PATH}
  GOAT data      : ${NAVAGENT_GOAT_QWENVL_DATA_DIR}
  OVON data      : ${NAVAGENT_OVON_QWENVL_DATA_DIR}
  datasets       : ${DATASETS}
  sampling       : natural union (no dataset resampling)
  loss weights   : ${SKILL_LOSS_WEIGHTS}
  data percent   : ${DATA_PERCENT}%
--------------------------------------------------------------------------------
  gpus           : ${GPU_IDS}   (nproc_per_node=${NPROC_PER_NODE})
  global_bs      : $(( PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NPROC_PER_NODE ))
  epochs         : ${NUM_TRAIN_EPOCHS}    max_steps=${MAX_STEPS}
  lr             : ${LEARNING_RATE}  warmup=${WARMUP_RATIO}  sched=${LR_SCHEDULER_TYPE}
  max_len        : ${MODEL_MAX_LENGTH}    px=[${MIN_PIXELS}, ${MAX_PIXELS}]
  tune vis/mlp/llm : ${TUNE_MM_VISION}/${TUNE_MM_MLP}/${TUNE_MM_LLM}
  bf16=${USE_BF16}  grad_ckpt=${GRADIENT_CHECKPOINTING}  deepspeed=${USE_DEEPSPEED} (${DEEPSPEED_CONFIG})
  save_strategy  : ${SAVE_STRATEGY}   save_steps=${SAVE_STEPS}   keep_last=${SAVE_TOTAL_LIMIT}
  resume_from    : ${_resume_path:-(from scratch)}
--------------------------------------------------------------------------------
  eval_steps     : ${EVAL_STEPS}   skills=${EVAL_SKILLS}
  eval_samples   : ${EVAL_NUM_SAMPLES_PER_SKILL}/domain/skill
  train_probe    : ${EVAL_TRAIN_PROBE_SIZE}/domain/skill
  eval_val_dirs  : ${EVAL_VAL_DIR}
--------------------------------------------------------------------------------
  output_dir     : ${OUTPUT_DIR}
  run_name       : ${RUN_NAME}
================================================================================
BANNER

torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  qwenvl/train/train_qwen.py \
  "${ARGS[@]}"

if [[ "${PLOT_LOSS_AFTER_TRAIN}" == "1" ]] \
   && [[ -f "${OUTPUT_DIR}/skill_loss_history.jsonl" ]]; then
  python "${REPO_ROOT}/tools/plot_skill_loss.py" \
    --input "${OUTPUT_DIR}/skill_loss_history.jsonl" \
    --output "${OUTPUT_DIR}/loss_curves.png" \
    --title "NavAgent GOAT + OVON Joint FT Loss Curves"
fi
