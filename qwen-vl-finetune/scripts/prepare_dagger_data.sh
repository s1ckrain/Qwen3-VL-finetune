#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ============================================================
# DAgger (Stage-2) data prep. Sibling of prepare_navagent_data.sh —
# that SFT script is NOT touched. Differences:
#   - reads the DAgger raw/labeled dirs, writes a SEPARATE dagger_qwenvl dir
#   - calls dagger_data_gen/run_export_qwenvl.py (the fork) with
#     --disagreement-only so only student<->oracle disagreement steps survive
#   - exports est + plan only; no val split (DAgger has no val; eval = GOAT 667)
# ============================================================
NAVAGENT_LABELED_DIR="${NAVAGENT_LABELED_DIR:-/root/data1/SFT-v4/dagger_labeled}"
NAVAGENT_RAW_DIR="${NAVAGENT_RAW_DIR:-/root/data1/SFT-v4/dagger_raw}"
NAVAGENT_QWENVL_DATA_DIR="${NAVAGENT_QWENVL_DATA_DIR:-/root/data1/SFT-v4/dagger_qwenvl}"
VERIFY_FILES="${VERIFY_FILES:-0}"
INCLUDE_PLANNING_INCONSISTENCY="${INCLUDE_PLANNING_INCONSISTENCY:-0}"

# DAgger step-level student<->oracle disagreement filter (the whole point of
# Stage-2): keep only steps where the student was wrong. Set DISAGREEMENT_ONLY=0
# to export every collected step instead.
DISAGREEMENT_ONLY="${DISAGREEMENT_ONLY:-1}"
PIXEL_TOL="${PIXEL_TOL:-64}"        # must match skill-eval EVAL_PIXEL_TOLERANCE
DAGGER_SKILLS=("estimating" "planning")

# 跳过导出: 若已有 qwenvl 数据可直接复用, 不重跑 run_export_qwenvl.py
SKIP_EXPORT="${SKIP_EXPORT:-0}"
SOURCE_QWENVL_DIR="${SOURCE_QWENVL_DIR:-}"

# 训练 / 验证集切分 — DAgger 默认不切 val (验证用 GOAT 667 子任务)
VAL_SPLIT_ENABLE="${VAL_SPLIT_ENABLE:-0}"
VAL_TARGET_PER_SKILL="${VAL_TARGET_PER_SKILL:-200}"
VAL_SPLIT_SEED="${VAL_SPLIT_SEED:-42}"
VAL_SPLIT_FORCE="${VAL_SPLIT_FORCE:-0}"

# estimating 类别均衡切分 (val + train_probe 各 96 Normal + 96 Completed)
# 再按 goal_type (object/description/image) 1:1:1 子分层
#   -> val/probe: Normal × {32+32+32} + Completed × {32+32+32} = 192 each
# + 去 CoT (data_gen 已 oracle-only 直出, 此 strip 是幂等兜底)
ESTIMATING_BALANCE_VAL_TRAIN_PROBE="${ESTIMATING_BALANCE_VAL_TRAIN_PROBE:-1}"
ESTIMATING_BALANCE_PER_CLASS="${ESTIMATING_BALANCE_PER_CLASS:-96}"
ESTIMATING_SUB_STRATIFY_GOAL_TYPE="${ESTIMATING_SUB_STRATIFY_GOAL_TYPE:-1}"
ESTIMATING_NO_COT="${ESTIMATING_NO_COT:-1}"

# planning 类别均衡切分 (val + train_probe 各 48 × 4 方向 = 192)
# 再按 goal_type (object/description/image) 1:1:1 子分层
#   -> val/probe: 4 directions × 3 goal_types × 16 = 192 each
# + 去 CoT 和 target_description (推理 prompt 已同步对齐, ckpt 可直接接入在线 PlanningSkill)
PLANNING_BALANCE_VAL_TRAIN_PROBE="${PLANNING_BALANCE_VAL_TRAIN_PROBE:-1}"
PLANNING_BALANCE_PER_CLASS="${PLANNING_BALANCE_PER_CLASS:-48}"
PLANNING_SUB_STRATIFY_GOAL_TYPE="${PLANNING_SUB_STRATIFY_GOAL_TYPE:-1}"
PLANNING_NO_COT="${PLANNING_NO_COT:-1}"

if [[ "${SKIP_EXPORT}" == "1" ]]; then
  if [[ ! -d "${NAVAGENT_QWENVL_DATA_DIR}" ]]; then
    if [[ -n "${SOURCE_QWENVL_DIR}" ]]; then
      if [[ ! -d "${SOURCE_QWENVL_DIR}" ]]; then
        echo "ERROR: SOURCE_QWENVL_DIR 不存在: ${SOURCE_QWENVL_DIR}" >&2
        exit 1
      fi
      echo "SKIP_EXPORT=1: 从 ${SOURCE_QWENVL_DIR} 复制到 ${NAVAGENT_QWENVL_DATA_DIR}"
      mkdir -p "${NAVAGENT_QWENVL_DATA_DIR}"
      # 只复制原始的 navagent_<skill>.jsonl, 已有的 _train/_val/_train_probe
      # split 和 manifest 会被本脚本后续重新生成, 不要带过来.
      shopt -s nullglob
      for _src in "${SOURCE_QWENVL_DIR}"/navagent_*.jsonl; do
        _bn="$(basename "${_src}")"
        case "${_bn}" in
          navagent_*_train.jsonl|navagent_*_val.jsonl|navagent_*_train_probe.jsonl)
            continue ;;
        esac
        cp -p "${_src}" "${NAVAGENT_QWENVL_DATA_DIR}/${_bn}"
      done
      # 也带上其他非 jsonl 资源 (除了 manifest), 一般是 images / metadata
      for _src in "${SOURCE_QWENVL_DIR}"/*; do
        _bn="$(basename "${_src}")"
        case "${_bn}" in
          navagent_val_manifest.json|navagent_*.jsonl)
            continue ;;
        esac
        cp -rp "${_src}" "${NAVAGENT_QWENVL_DATA_DIR}/${_bn}"
      done
      shopt -u nullglob
    else
      echo "ERROR: SKIP_EXPORT=1 但目标目录不存在且没设 SOURCE_QWENVL_DIR" >&2
      echo "       请设 SOURCE_QWENVL_DIR=<已有的 qwenvl 目录>" >&2
      exit 1
    fi
  else
    echo "SKIP_EXPORT=1: 复用现有 ${NAVAGENT_QWENVL_DATA_DIR}"
  fi
else
  # DAgger export: per-skill (est/plan) via the FORKED exporter, with the
  # step-level student<->oracle disagreement filter. data_gen/run_export_qwenvl.py
  # (the SFT exporter) is intentionally NOT used here.
  for skill in "${DAGGER_SKILLS[@]}"; do
    CMD=(
      python "/root/NavAgent/dagger_data_gen/run_export_qwenvl.py"
      --labeled-dir "${NAVAGENT_LABELED_DIR}"
      --raw-dir "${NAVAGENT_RAW_DIR}"
      --output-dir "${NAVAGENT_QWENVL_DATA_DIR}"
      --skill "${skill}"
    )
    if [[ "${DISAGREEMENT_ONLY}" == "1" ]]; then
      CMD+=(--disagreement-only --pixel-tol "${PIXEL_TOL}")
    fi
    if [[ "${VERIFY_FILES}" == "1" ]]; then
      CMD+=(--verify-files)
    fi
    if [[ "${INCLUDE_PLANNING_INCONSISTENCY}" == "1" ]]; then
      CMD+=(--include-planning-inconsistency)
    fi
    echo "导出 DAgger Qwen-VL 数据 (${skill}, disagreement_only=${DISAGREEMENT_ONLY}, pixel_tol=${PIXEL_TOL}) -> ${NAVAGENT_QWENVL_DATA_DIR}"
    "${CMD[@]}"
  done
fi

if [[ "${VAL_SPLIT_ENABLE}" == "1" ]]; then
  SPLIT_CMD=(
    python "${REPO_ROOT}/tools/split_navagent_val.py"
    --input-dir "${NAVAGENT_QWENVL_DATA_DIR}"
    --output-dir "${NAVAGENT_QWENVL_DATA_DIR}"
    --target-per-skill "${VAL_TARGET_PER_SKILL}"
    --seed "${VAL_SPLIT_SEED}"
  )
  if [[ "${VAL_SPLIT_FORCE}" == "1" ]]; then
    SPLIT_CMD+=(--force)
  fi
  _strat_msg=""
  if [[ "${ESTIMATING_BALANCE_VAL_TRAIN_PROBE}" == "1" ]]; then
    SPLIT_CMD+=(--stratify-balanced-skill estimating
                --balance-per-class "estimating=${ESTIMATING_BALANCE_PER_CLASS}")
    _strat_msg+="estimating=${ESTIMATING_BALANCE_PER_CLASS}x2 "
    if [[ "${ESTIMATING_SUB_STRATIFY_GOAL_TYPE}" == "1" ]]; then
      SPLIT_CMD+=(--sub-stratify-skill "estimating=goal_type")
      _strat_msg+="(estimating goal_type 1:1:1) "
    fi
  fi
  if [[ "${PLANNING_BALANCE_VAL_TRAIN_PROBE}" == "1" ]]; then
    SPLIT_CMD+=(--stratify-balanced-skill planning
                --balance-per-class "planning=${PLANNING_BALANCE_PER_CLASS}")
    _strat_msg+="planning=${PLANNING_BALANCE_PER_CLASS}x4 "
    if [[ "${PLANNING_SUB_STRATIFY_GOAL_TYPE}" == "1" ]]; then
      SPLIT_CMD+=(--sub-stratify-skill "planning=goal_type")
      _strat_msg+="(planning goal_type 1:1:1) "
    fi
  fi
  if [[ -n "${_strat_msg}" ]]; then
    echo "切分 train/val (stratified: ${_strat_msg}; 其他 skill: episode 级, target=${VAL_TARGET_PER_SKILL})"
  else
    echo "切分 train/val (episode 级, target=${VAL_TARGET_PER_SKILL} 每个主 skill)"
  fi
  "${SPLIT_CMD[@]}"
fi

# CoT 兜底剥离: 新版 data_gen 走 oracle-only 直出已是 minimal schema, 这两步是
# 幂等 no-op; 主要兼容用 SKIP_EXPORT=1 + SOURCE_QWENVL_DIR 复用旧 (带 CoT) 数据
# 的场景, 把旧数据原地剥成 minimal schema 后再训.
if [[ "${ESTIMATING_NO_COT}" == "1" ]]; then
  echo "去掉 estimating 的 CoT (仅保留 status / deviated_node_id / suggested_rollback_target)"
  python "/root/NavAgent/tools/strip_estimating_cot.py" \
    --input-dir "${NAVAGENT_QWENVL_DATA_DIR}"
fi

if [[ "${PLANNING_NO_COT}" == "1" ]]; then
  echo "去掉 planning 的 CoT 和 free-form 字段 (仅保留 selected_direction / target_pixel)"
  python "/root/NavAgent/tools/strip_planning_cot.py" \
    --input-dir "${NAVAGENT_QWENVL_DATA_DIR}"
fi
