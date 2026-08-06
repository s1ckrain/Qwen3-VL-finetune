#!/bin/bash
# NavAgent DAgger (Stage-2) continue-finetuning launcher.
#
# This is a thin wrapper over scripts/navagent_full_ft.sh. Stage-2 starts from
# a Stage-1 SFT checkpoint as model weights, trains on full DAgger
# student-rollout data only, and writes to a separate output directory. It does
# not mix SFT train replay and does not use disagreement-only data by default.
#
# "continue" here means weight initialization from the SFT checkpoint with a
# fresh optimizer/scheduler in a new output dir. Crash resume still uses
# RESUME_FROM_CHECKPOINT=latest inside that output dir.
#
# Override anything from the environment, e.g.:
#   SFT_CKPT=/path/to/sft GPU_IDS=0,1,2,3 bash scripts/navagent_dagger_ft.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Data / checkpoint dirs ------------------------------------------------
export NAVAGENT_DAGGER_DATA_DIR="${NAVAGENT_DAGGER_DATA_DIR:-/root/data1/SFT-v4/dagger_qwenvl}"
# navagent_full_ft.sh still performs a generic navagent_observing.jsonl check
# against NAVAGENT_QWENVL_DATA_DIR, so point it at the DAgger qwenvl dir.
export NAVAGENT_QWENVL_DATA_DIR="${NAVAGENT_QWENVL_DATA_DIR:-${NAVAGENT_DAGGER_DATA_DIR}}"

SFT_CKPT="${SFT_CKPT:-${MODEL_PATH:-/root/data1/models/SFT-v4-1m}}"
export MODEL_PATH="${SFT_CKPT}"

# ---- Dataset mix: full DAgger only -----------------------------------------
export DATASETS="${DATASETS:-navagent_observing_dagger,navagent_estimating_dagger,navagent_planning_dagger}"
# Full DAgger means every exported row is used once unless the caller
# explicitly overrides the resampling mode/weights for an ablation.
export DATASET_RESAMPLE_MODE="${DATASET_RESAMPLE_MODE:-natural}"
export SKILL_LOSS_WEIGHTS="${SKILL_LOSS_WEIGHTS:-observing=1,estimating=1,planning=1}"
export USE_TRAIN_SPLIT="${USE_TRAIN_SPLIT:-0}"

# DAgger has no held-out val split by default; closed-loop GOAT/OVON eval is
# the real validation. Re-enable EVAL_STEPS manually only if val files exist.
export EVAL_STEPS="${EVAL_STEPS:-0}"

# ---- Output / resume -------------------------------------------------------
export MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT:-/root/data1/models/SFT-v4-dagger-continue}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-latest}"
export RUN_NAME="${RUN_NAME:-navagent-dagger-continue}"

# ---- Pre-flight ------------------------------------------------------------
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: SFT checkpoint not found: ${MODEL_PATH}" >&2
  echo "       Set SFT_CKPT=/path/to/stage1-sft or MODEL_PATH=/path/to/stage1-sft." >&2
  exit 1
fi

for _f in navagent_observing.jsonl navagent_estimating.jsonl navagent_planning.jsonl; do
  if [[ ! -f "${NAVAGENT_DAGGER_DATA_DIR}/${_f}" ]]; then
    echo "ERROR: missing dagger dataset: ${NAVAGENT_DAGGER_DATA_DIR}/${_f}" >&2
    echo "       Run scripts/prepare_dagger_data.sh first." >&2
    exit 1
  fi
done

echo ">>> NavAgent DAgger continue-finetuning"
echo ">>> SFT ckpt : ${MODEL_PATH}"
echo ">>> DAgger   : ${NAVAGENT_DAGGER_DATA_DIR}"
echo ">>> datasets : ${DATASETS}"
echo ">>> output   : ${MODEL_OUTPUT_ROOT}"

exec bash "${SCRIPT_DIR}/navagent_full_ft.sh" "$@"
