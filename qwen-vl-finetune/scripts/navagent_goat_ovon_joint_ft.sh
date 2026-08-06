#!/bin/bash
# Joint GOAT + OVON full-parameter SFT for NavAgent.
#
# The two corpora remain in their original directories.  This wrapper registers
# all six train JSONL files with the existing Qwen3-VL loader, uses every row
# exactly once per epoch, and evaluates GOAT / OVON separately.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Data roots -------------------------------------------------------------
: "${NAVAGENT_GOAT_QWENVL_DATA_DIR:=/root/data1/SFT-goat/qwenvl}"
: "${NAVAGENT_OVON_QWENVL_DATA_DIR:=/root/data1/SFT-ovon/qwenvl}"
export NAVAGENT_GOAT_QWENVL_DATA_DIR
export NAVAGENT_OVON_QWENVL_DATA_DIR

# navagent_full_ft.sh retains a legacy single-root preflight.  Point that check
# at GOAT; this wrapper performs the authoritative six-file preflight below.
export NAVAGENT_QWENVL_DATA_DIR="${NAVAGENT_GOAT_QWENVL_DATA_DIR}"

# ---- Joint dataset list -----------------------------------------------------
: "${DATA_PERCENT:=100}"
if ! [[ "${DATA_PERCENT}" =~ ^[0-9]+$ ]] \
   || (( DATA_PERCENT < 1 || DATA_PERCENT > 100 )); then
  echo "DATA_PERCENT 必须是 [1, 100] 之间的整数, 当前: ${DATA_PERCENT}" >&2
  exit 1
fi

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
export DATASETS="${DATASETS:-$(IFS=,; echo "${_dataset_parts[*]}")}"

# Natural union: do not balance domains or skills and do not duplicate rows.
export DATASET_RESAMPLE_MODE="${DATASET_RESAMPLE_MODE:-natural}"
export SKILL_LOSS_WEIGHTS="${SKILL_LOSS_WEIGHTS:-observing=1,estimating=1,planning=1}"

# ---- Multi-domain periodic evaluation --------------------------------------
# 96 samples per domain keeps the total generation load close to the previous
# single-domain default of 192 samples per skill.
export EVAL_VAL_DIR="${EVAL_VAL_DIR:-goat=${NAVAGENT_GOAT_QWENVL_DATA_DIR},ovon=${NAVAGENT_OVON_QWENVL_DATA_DIR}}"
export EVAL_NUM_SAMPLES_PER_SKILL="${EVAL_NUM_SAMPLES_PER_SKILL:-96}"
export EVAL_TRAIN_PROBE_SIZE="${EVAL_TRAIN_PROBE_SIZE:-96}"

# ---- Independent output -----------------------------------------------------
export MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT:-/root/data/train/SFT-goat-ovon-2B-1.5M}"
export RUN_NAME="${RUN_NAME:-navagent-sft-goat-ovon-2B-1.5M-p${DATA_PERCENT}}"
export USE_TRAIN_SPLIT=1

# ---- Preflight --------------------------------------------------------------
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
  for _skill in estimating planning; do
    _val_file="${_root}/navagent_${_skill}_val.jsonl"
    if [[ ! -s "${_val_file}" ]]; then
      echo "ERROR: missing or empty joint-eval dataset: ${_val_file}" >&2
      exit 1
    fi
  done
done

cat <<BANNER
================================================================================
  NavAgent GOAT + OVON Joint SFT
--------------------------------------------------------------------------------
  GOAT data      : ${NAVAGENT_GOAT_QWENVL_DATA_DIR}
  OVON data      : ${NAVAGENT_OVON_QWENVL_DATA_DIR}
  sampling       : ${DATASET_RESAMPLE_MODE} (each selected row once per epoch)
  data percent   : ${DATA_PERCENT}%
  eval domains   : ${EVAL_VAL_DIR}
  output root    : ${MODEL_OUTPUT_ROOT}
================================================================================
BANNER

exec bash "${SCRIPT_DIR}/navagent_full_ft.sh" "$@"
