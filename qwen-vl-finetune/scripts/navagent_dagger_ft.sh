#!/bin/bash
# NavAgent DAgger (Stage-2) training launcher.
#
# Thin wrapper over scripts/navagent_full_ft.sh — that SFT launcher is NOT
# modified. We only set DAgger-specific overrides (all of which navagent_full_ft.sh
# already respects via `: "${VAR:=...}"`) and then exec it:
#
#   * mix = the active (450k) SFT `_train` datasets  +  the Stage-2 dagger datasets
#   * resample: SFT skills kept at the SFT-v4 baseline (all weights -> 155189, i.e.
#     balanced up to the largest skill, exactly like the pure-SFT all-1 run, so this
#     stays ALIGNED with SFT); dagger kept at its real row count = NOT upsampled.
#     dagger ends up ~19% of the mix. loss weights all 1x.
#   * train FROM BASE Qwen3-VL-4B into a SEPARATE output dir
#     (SFT-v4-dagger), NOT continue-finetuning the SFT-v4 checkpoint
#   * everything else (hyperparams, deepspeed zero3, eval) inherited verbatim
#
# Override anything from the environment, e.g.:
#   GPU_IDS=0,1,2,3 NAVAGENT_DAGGER_DATA_DIR=/path bash scripts/navagent_dagger_ft.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Data dirs -------------------------------------------------------------
# 1M SFT qwenvl (provides observing/estimating/planning *_train*) ...
export NAVAGENT_QWENVL_DATA_DIR="${NAVAGENT_QWENVL_DATA_DIR:-/root/data1/SFT-v4/qwenvl}"
# ... and the Stage-2 disagreement-filtered dagger qwenvl (separate dir).
export NAVAGENT_DAGGER_DATA_DIR="${NAVAGENT_DAGGER_DATA_DIR:-/root/data1/SFT-v4/dagger_qwenvl}"

# ---- Dataset mix (explicit -> disables the dynamic builders in full_ft) -----
# SFT _train (3 skills, active 450k variant) + dagger (est/plan, disagreement-filtered).
export DATASETS="${DATASETS:-navagent_observing_train,navagent_estimating_train,navagent_planning_train,navagent_estimating_dagger,navagent_planning_dagger}"
# Resample weight here = each dataset's TARGET row count. full_ft's resampler uses
# target_i = base_count * w_i with base_count = max(len_j/w_j); with the values below
# no len/weight exceeds 1 so base_count==1 and target==weight literally. The ratio of
# final sample counts == the ratio of weights, so weights MUST be on one scale (all
# row counts) — never mix "1" with raw row counts (that blows dagger up ~10000x).
#   - SFT three skills -> 155189 (= the largest skill): balanced exactly like the
#     pure-SFT all-1 run, so the SFT half stays ALIGNED with the SFT-v4 baseline.
#   - dagger two skills -> their real row counts: kept as-is, NOT upsampled.
# => dagger ~19.3% of the 577k-sample mix.
# !!! Hard-codes CURRENT row counts; regenerate data / swap datasets => you MUST update
#     these (SFT three -> the new largest skill's count; dagger -> the new sizes). !!!
export DATASET_RESAMPLE_WEIGHTS="${DATASET_RESAMPLE_WEIGHTS:-observing=155189,estimating=155189,planning=155189,estimating_dagger=10584,planning_dagger=100912}"
export SKILL_LOSS_WEIGHTS="${SKILL_LOSS_WEIGHTS:-observing=1,estimating=1,planning=1,estimating_dagger=1,planning_dagger=1}"

# ---- From BASE, separate output dir ----------------------------------------
export MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT:-/root/data1/models/SFT-v4-dagger-450k}"
# "latest" + a fresh (empty) output dir => navagent_full_ft.sh trains from base
# (its RESUME default uses `:=`, so an empty string would be replaced by the
# SFT-v4 checkpoint — we must use "latest" instead). On a crash-resume it will
# correctly pick up the newest checkpoint in MODEL_OUTPUT_ROOT.
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-latest}"
export RUN_NAME="${RUN_NAME:-navagent-dagger-r1}"

# ---- Pre-flight: dagger data must exist ------------------------------------
for _f in navagent_estimating.jsonl navagent_planning.jsonl; do
  if [[ ! -f "${NAVAGENT_DAGGER_DATA_DIR}/${_f}" ]]; then
    echo "ERROR: missing dagger dataset: ${NAVAGENT_DAGGER_DATA_DIR}/${_f}" >&2
    echo "       Run scripts/prepare_dagger_data.sh first." >&2
    exit 1
  fi
done

echo ">>> NavAgent DAgger training (from base; output -> ${MODEL_OUTPUT_ROOT})"
echo ">>> SFT dir : ${NAVAGENT_QWENVL_DATA_DIR}"
echo ">>> DAgger  : ${NAVAGENT_DAGGER_DATA_DIR}"
echo ">>> datasets: ${DATASETS}"

exec bash "${SCRIPT_DIR}/navagent_full_ft.sh" "$@"
