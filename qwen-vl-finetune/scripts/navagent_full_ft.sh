#!/bin/bash
# NavAgent 全参数 SFT 启动脚本 (Qwen3-VL)
# 所有参数支持环境变量覆盖, 例如:
#   NUM_TRAIN_EPOCHS=3 LEARNING_RATE=2e-6 bash scripts/navagent_full_ft.sh
set -euo pipefail

# ---- OOM 防御: 减少 CUDA caching allocator 碎片 ---------------------------
# expandable_segments=True: 让 allocator 动态扩缩 segment, 避免长跑 + eval 碎片
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- 路径 -------------------------------------------------------------------
# 数据目录需提前由 scripts/prepare_navagent_data.sh 准备好, 本脚本不会再生成
: "${MODEL_PATH:=/root/data/models/Qwen3-VL-2B-Instruct}"
: "${NAVAGENT_QWENVL_DATA_DIR:=/root/data1/SFT-ovon/qwenvl}"
: "${MODEL_OUTPUT_ROOT:=/root/data/train/SFT-ovon-2B-1M}"
: "${OUTPUT_DIR:=${MODEL_OUTPUT_ROOT}}"
export NAVAGENT_QWENVL_DATA_DIR

# ---- 数据组合 ---------------------------------------------------------------
# DATA_PERCENT: 取每个 skill 前 N% (1..100)
# LOSS_WEIGHT_*: 各 skill loss 权重 (会归一化, 不影响有效 LR)
# RESAMPLE_WEIGHT_*: 各 skill 数据重复倍数, 设为 0 等于关闭该 skill
# DATASETS: 设非空可完全覆盖数据集字符串; 留空则按 RESAMPLE_WEIGHT 自动拼
#
# 当前默认: 4 skill 联合训练, 全 1:1:1:1
# 单 skill 训练: 把其他 3 个 RESAMPLE_WEIGHT_* 设 0
: "${DATA_PERCENT:=100}"
: "${LOSS_WEIGHT_OBSERVING:=1}"
: "${LOSS_WEIGHT_ESTIMATING:=1}"
: "${LOSS_WEIGHT_SCHEDULER:=0}"
: "${LOSS_WEIGHT_PLANNING:=1}"
: "${RESAMPLE_WEIGHT_OBSERVING:=1}"
: "${RESAMPLE_WEIGHT_ESTIMATING:=1}"
: "${RESAMPLE_WEIGHT_SCHEDULER:=0}"
: "${RESAMPLE_WEIGHT_PLANNING:=1}"
: "${DATASETS:=}"

# ---- 训练参数 ---------------------------------------------------------------
: "${NUM_TRAIN_EPOCHS:=2}"
: "${MAX_STEPS:=-1}"
: "${LEARNING_RATE:=1e-5}" # 4B改成1e-5
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=4}" # 4B改成4
: "${GRADIENT_ACCUMULATION_STEPS:=2}" # 4卡用2，8卡用1
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
: "${DEEPSPEED_CONFIG:=./scripts/zero3.json}"  # zero2+bf16 腐蚀权重产生 NaN(已确认), 用 zero3

# ---- 日志 / checkpoint ------------------------------------------------------
# SAVE_STRATEGY: steps (默认, 按 SAVE_STEPS 周期存中间 ckpt, 防 OOM 丢全程进度)
#                no (仅最终模型) / epoch (每 epoch 末尾存)
# 注: train_qwen.py 在 trainer.train() 结束后会主动 save_model 到 OUTPUT_DIR 根目录,
#     所以最终模型一定会有, 中间 ckpt 用于 OOM 后 --resume_from_checkpoint 续训.
# SAVE_TOTAL_LIMIT=3 表示最多保留最近 3 个 ckpt, 防止盘爆.
: "${LOGGING_STEPS:=20}"
: "${SAVE_STRATEGY:=steps}"
: "${SAVE_STEPS:=2000}"
: "${SAVE_TOTAL_LIMIT:=2}"
# RESUME_FROM_CHECKPOINT: 留空则从头训; 设 "latest" 则自动找 OUTPUT_DIR 下最新 checkpoint-*;
#                         设具体路径 (如 /path/to/checkpoint-12000) 则从该 ckpt 续训.
: "${RESUME_FROM_CHECKPOINT:=}"
: "${REPORT_TO:=none}"
: "${RUN_NAME:=navagent-sft-ovon-2B-1M-p${DATA_PERCENT}}"
: "${PLOT_LOSS_AFTER_TRAIN:=1}"
: "${DATALOADER_NUM_WORKERS:=8}"
: "${DATALOADER_PREFETCH_FACTOR:=4}"

# ---- 周期 skill eval --------------------------------------------------------
# 仅 estimating + planning 在 prepare_navagent_data.sh 里被切了 val/train_probe
#   estimating: 比较 status (Normal/Lost/Completed)
#   planning  : 比较 selected_direction + target_pixel (L2 < EVAL_PIXEL_TOLERANCE 算对)
# observing / scheduler 没有 val 切分, 不参与周期 eval (训这两个 skill 时设 EVAL_STEPS=0).
: "${EVAL_STEPS:=2000}"
: "${EVAL_NUM_SAMPLES_PER_SKILL:=192}"
: "${EVAL_MAX_NEW_TOKENS:=256}"
: "${EVAL_SKILLS:=estimating,planning}"
: "${EVAL_VAL_DIR:=${NAVAGENT_QWENVL_DATA_DIR}}"
: "${EVAL_ON_START:=1}"
: "${EVAL_TRAIN_PROBE_SIZE:=192}"
: "${EVAL_PIXEL_TOLERANCE:=64}"

# ---- 分布式 -----------------------------------------------------------------
# 注: 同机多 run 并行时 GPU_IDS 必须互斥 (CUDA_VISIBLE_DEVICES 隔离), MASTER_PORT 默认
#     在 [20001, 29999] 随机抽 1 个, 碰撞概率极低; 如同时跑多个想完全确定也可显式指定.
: "${GPU_IDS:=0,1,2,3}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${MASTER_PORT:=$(shuf -i 20001-29999 -n 1)}"

# ===========================================================================
# 以下是脚本内部逻辑, 一般不需要改
# ===========================================================================

# 输入校验
if ! [[ "${DATA_PERCENT}" =~ ^[0-9]+$ ]] || (( DATA_PERCENT < 1 || DATA_PERCENT > 100 )); then
  echo "DATA_PERCENT 必须是 [1, 100] 之间的整数, 当前: ${DATA_PERCENT}" >&2
  exit 1
fi

# 数据存在性检查
# USE_TRAIN_SPLIT=1: 读 navagent_<skill>_train.jsonl (已剔除 val/probe 的版本),
# 保证 val/train_probe 那 192 条不会泄露进训练. 默认开启.
# USE_TRAIN_SPLIT=0: 直接读 navagent_<skill>.jsonl 全量, val 会被训, eval 数虚高.
: "${USE_TRAIN_SPLIT:=1}"
if [[ ! -f "${NAVAGENT_QWENVL_DATA_DIR}/navagent_observing.jsonl" ]]; then
  echo "ERROR: 数据目录缺少 jsonl 文件: ${NAVAGENT_QWENVL_DATA_DIR}" >&2
  echo "       请先运行 scripts/prepare_navagent_data.sh" >&2
  exit 1
fi
if [[ "${USE_TRAIN_SPLIT}" == "1" \
      && ! -f "${NAVAGENT_QWENVL_DATA_DIR}/navagent_observing_train.jsonl" ]]; then
  echo "ERROR: USE_TRAIN_SPLIT=1 但缺少 navagent_observing_train.jsonl" >&2
  echo "       请用 VAL_SPLIT_ENABLE=1 重跑 prepare_navagent_data.sh" >&2
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

# 拼数据集字符串 (USE_TRAIN_SPLIT=1 表示用 split_navagent_val.py 切出的 _train.jsonl)
_skill_suffix=""
if [[ "${USE_TRAIN_SPLIT}" == "1" ]]; then
  _skill_suffix="_train"
fi

# Scheduler 已退化为纯 Python 规则 (NavAgent/skills/scheduler.py), 不进训练
# 混合; sft-label.sh 默认 SKILLS 也不再产出 navagent_scheduler.jsonl.
_skills=(observing estimating planning)
if [[ -z "${DATASETS}" ]]; then
  _parts=()
  for s in "${_skills[@]}"; do
    if (( DATA_PERCENT >= 100 )); then
      _parts+=("navagent_${s}${_skill_suffix}")
    else
      _parts+=("navagent_${s}${_skill_suffix}%${DATA_PERCENT}")
    fi
  done
  DATASETS="$(IFS=,; echo "${_parts[*]}")"
fi

# Build per-skill weight strings DYNAMICALLY from _skills so removing a
# skill from the array (e.g. scheduler) doesn't leave a dangling weight
# entry referencing a non-existent dataset.
if [[ -z "${DATASET_RESAMPLE_WEIGHTS:-}" ]]; then
  _w=()
  for s in "${_skills[@]}"; do
    _ucase=$(echo "$s" | tr '[:lower:]' '[:upper:]')
    _var="RESAMPLE_WEIGHT_${_ucase}"
    _w+=("${s}=${!_var}")
  done
  DATASET_RESAMPLE_WEIGHTS="$(IFS=,; echo "${_w[*]}")"
fi
if [[ -z "${SKILL_LOSS_WEIGHTS:-}" ]]; then
  _w=()
  for s in "${_skills[@]}"; do
    _ucase=$(echo "$s" | tr '[:lower:]' '[:upper:]')
    _var="LOSS_WEIGHT_${_ucase}"
    _w+=("${s}=${!_var}")
  done
  SKILL_LOSS_WEIGHTS="$(IFS=,; echo "${_w[*]}")"
fi

mkdir -p "${OUTPUT_DIR}"

ARGS=(
  --model_name_or_path "${MODEL_PATH}"
  --dataset_use "${DATASETS}"
  --dataset_resample_weights "${DATASET_RESAMPLE_WEIGHTS}"
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

# 仅在 save_strategy 不为 no 时才传 save_steps / save_total_limit
if [[ "${SAVE_STRATEGY}" != "no" ]]; then
  [[ "${SAVE_STRATEGY}" == "steps" ]] && ARGS+=(--save_steps "${SAVE_STEPS}")
  ARGS+=(--save_total_limit "${SAVE_TOTAL_LIMIT}")
fi

# 续训: 支持 latest 自动找 OUTPUT_DIR 下最新 checkpoint-N
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
  else
    if [[ -d "${RESUME_FROM_CHECKPOINT}" ]]; then
      _resume_path="${RESUME_FROM_CHECKPOINT}"
    else
      echo "ERROR: RESUME_FROM_CHECKPOINT 路径不存在: ${RESUME_FROM_CHECKPOINT}" >&2
      exit 1
    fi
  fi
  [[ -n "${_resume_path}" ]] && ARGS+=(--resume_from_checkpoint "${_resume_path}")
fi

# 周期 eval
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

# 启动 banner
cat <<BANNER
================================================================================
  NavAgent SFT
--------------------------------------------------------------------------------
  model        : ${MODEL_PATH}
  data_dir     : ${NAVAGENT_QWENVL_DATA_DIR}
  datasets     : ${DATASETS}
  resample_w   : ${DATASET_RESAMPLE_WEIGHTS}
  loss_w       : ${SKILL_LOSS_WEIGHTS}
  data_percent : ${DATA_PERCENT}%
--------------------------------------------------------------------------------
  gpus         : ${GPU_IDS}   (nproc_per_node=${NPROC_PER_NODE})
  global_bs    : $(( PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NPROC_PER_NODE ))
  epochs       : ${NUM_TRAIN_EPOCHS}    max_steps=${MAX_STEPS}
  lr           : ${LEARNING_RATE}  warmup=${WARMUP_RATIO}  sched=${LR_SCHEDULER_TYPE}
  max_len      : ${MODEL_MAX_LENGTH}    px=[${MIN_PIXELS}, ${MAX_PIXELS}]
  tune vis/mlp/llm : ${TUNE_MM_VISION}/${TUNE_MM_MLP}/${TUNE_MM_LLM}
  bf16=${USE_BF16}  grad_ckpt=${GRADIENT_CHECKPOINTING}  deepspeed=${USE_DEEPSPEED} (${DEEPSPEED_CONFIG})
  save_strategy: ${SAVE_STRATEGY}   save_steps=${SAVE_STEPS}   keep_last=${SAVE_TOTAL_LIMIT}
  resume_from  : ${_resume_path:-(from scratch)}
--------------------------------------------------------------------------------
  eval_steps   : ${EVAL_STEPS}   skills=${EVAL_SKILLS}
  eval_samples : ${EVAL_NUM_SAMPLES_PER_SKILL}/skill   max_new_tokens=${EVAL_MAX_NEW_TOKENS}
  train_probe  : ${EVAL_TRAIN_PROBE_SIZE}/skill
  pixel_tol    : ${EVAL_PIXEL_TOLERANCE} px
  eval_val_dir : ${EVAL_VAL_DIR}
--------------------------------------------------------------------------------
  output_dir   : ${OUTPUT_DIR}
  run_name     : ${RUN_NAME}
================================================================================
BANNER

# 启动训练
torchrun \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  qwenvl/train/train_qwen.py \
  "${ARGS[@]}"

# 训练后画 loss 曲线
if [[ "${PLOT_LOSS_AFTER_TRAIN}" == "1" ]] \
   && [[ -f "${OUTPUT_DIR}/skill_loss_history.jsonl" ]]; then
  python "${REPO_ROOT}/tools/plot_skill_loss.py" \
    --input "${OUTPUT_DIR}/skill_loss_history.jsonl" \
    --output "${OUTPUT_DIR}/loss_curves.png" \
    --title "NavAgent Full FT Loss Curves"
fi
