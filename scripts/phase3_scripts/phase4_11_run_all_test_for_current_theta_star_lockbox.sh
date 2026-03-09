#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Phase4_11: Run ALL frozen Test sets (lockbox) for the CURRENT θ*
# - No sweep (lockbox): metrics aggregation runs with --no_sweep
# - Fixed datasets: {test_core, test_conflict, test_noisy, test_oom}
# - Safety: verify dataset SHA-256 against the Phase1 frozen manifest
# ------------------------------------------------------------
# Usage:
#   ./phase4_11_run_all_test_for_current_theta_star_lockbox_v1_2.sh <IDX> <METHOD> <RUN_TAG> [--force] [--dataset_dir <dir>] [--skip_sha_check]
#
# Example:
#   ./phase4_11_run_all_test_for_current_theta_star_lockbox_v1_2.sh 201 llm_reasoner_theta_star 2026-01-28 --force
# ------------------------------------------------------------

IDX="${1:-}"
METHOD="${2:-}"
RUN_TAG="${3:-}"

if [[ -z "${IDX}" || -z "${METHOD}" || -z "${RUN_TAG}" ]]; then
  echo "[ERROR] Missing required args."
  echo "Usage: $0 <IDX> <METHOD> <RUN_TAG> [--force] [--dataset_dir <dir>] [--skip_sha_check]"
  exit 1
fi

FORCE=0
DATASET_DIR=""
SKIP_SHA_CHECK=0

shift 3
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --dataset_dir) DATASET_DIR="$2"; shift 2 ;;
    --skip_sha_check) SKIP_SHA_CHECK=1; shift ;;
    *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PHASE3_RUN="$REPO_ROOT/scripts/phase3_scripts/phase3_01_run_eval_ros.py"
PHASE3_AGG="$REPO_ROOT/scripts/phase3_scripts/phase3_02_aggregate_metrics.py"

TEST_ROOT="$REPO_ROOT/config/test"
TEST_LOG_DIR="$TEST_ROOT/test_log"
TEST_METRICS_DIR="$TEST_ROOT/test_metrics"

mkdir -p "$TEST_LOG_DIR" "$TEST_METRICS_DIR"

# Prefer explicit dataset_dir. Otherwise, try canonical locations.
if [[ -z "$DATASET_DIR" ]]; then
  if [[ -d "$TEST_ROOT/test_dataset" ]]; then
    DATASET_DIR="$TEST_ROOT/test_dataset"
  elif [[ -d "$TEST_ROOT/test_queries" ]]; then
    DATASET_DIR="$TEST_ROOT/test_queries"
  else
    DATASET_DIR="$TEST_ROOT"
  fi
fi

# Frozen test set names (without .json)
DATASETS=("test_core" "test_conflict" "test_noisy" "test_oom")

# Phase1 frozen-manifest SHA-256 (for lockbox verification)
# (These should match phase1_frozen_manifest.json.)
declare -A EXPECTED_SHA256=(
  ["test_core.json"]="242c023876a63cf3ed178dd3ca6ef4d6ad3ffa388505c5542b76b55988ee9869"
  ["test_conflict.json"]="b4a2c9d05caa95e34780218fc23a60aa93fc46fc7eb35e0b8dcd36e24bd92abb"
  ["test_noisy.json"]="3f9208a525bc488f9f6cf47dd36cdc152112e161724bd25c10569beaa7e279af"
  ["test_oom.json"]="b2a03ecffa41c56245c43af9d6da6d74f7e7d0821f9d2a55c19d8dd3d6afdc34"
)

sha256_file() {
  local f="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$f" | awk '{print $1}'
  else
    echo ""
  fi
}

# Sanity checks
[[ -f "$PHASE3_RUN" ]] || { echo "[ERROR] Not found: $PHASE3_RUN"; exit 1; }
[[ -f "$PHASE3_AGG" ]] || { echo "[ERROR] Not found: $PHASE3_AGG"; exit 1; }
[[ -d "$DATASET_DIR" ]] || { echo "[ERROR] Dataset dir not found: $DATASET_DIR"; exit 1; }

echo "[INFO] IDX=$IDX  METHOD=$METHOD  RUN_TAG=$RUN_TAG"
echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] DATASET_DIR=$DATASET_DIR"
echo "[INFO] FORCE=$FORCE  SKIP_SHA_CHECK=$SKIP_SHA_CHECK"
echo

for DS in "${DATASETS[@]}"; do
  DATASET_JSON="$DATASET_DIR/${DS}.json"
  [[ -f "$DATASET_JSON" ]] || { echo "[ERROR] Missing dataset: $DATASET_JSON"; exit 1; }

  if [[ "$SKIP_SHA_CHECK" -eq 0 ]]; then
    bn="$(basename "$DATASET_JSON")"
    expected="${EXPECTED_SHA256[$bn]:-}"
    if [[ -z "$expected" ]]; then
      echo "[ERROR] Expected SHA not registered for $bn (script bug)."
      exit 1
    fi
    got="$(sha256_file "$DATASET_JSON")"
    if [[ -z "$got" ]]; then
      echo "[WARN] sha256sum/shasum not available; cannot verify SHA. (use --skip_sha_check to silence)"
    elif [[ "$got" != "$expected" ]]; then
      echo "[ERROR] SHA-256 mismatch for $bn"
      echo "  expected: $expected"
      echo "  got     : $got"
      exit 1
    else
      echo "[OK] SHA-256 verified: $bn"
    fi
  fi
done

echo

for DS in "${DATASETS[@]}"; do
  echo "============================================================"
  echo "[RUN] Dataset: $DS  (theta IDX=$IDX)"
  echo "============================================================"

  DATASET_JSON="$DATASET_DIR/${DS}.json"

  OUT_JSONL="$TEST_LOG_DIR/${DS}_theta${IDX}_run.jsonl"
  OUT_DIR="$TEST_METRICS_DIR/theta${IDX}/${DS}"
  PREFIX="${DS}_theta${IDX}_"   # must start with "test_" to satisfy lockbox conventions downstream

  mkdir -p "$OUT_DIR"

  if [[ "$FORCE" -eq 1 ]]; then
    echo "[INFO] --force: removing existing outputs for $DS (IDX=$IDX)"
    rm -f "$OUT_JSONL"
    rm -rf "$OUT_DIR"
    mkdir -p "$OUT_DIR"
  fi

  echo "[STEP1] Running eval -> JSONL"
  python3 "$PHASE3_RUN" \
    --dataset "$DATASET_JSON" \
    --out "$OUT_JSONL" \
    --run_id "$RUN_TAG" \
    --method "$METHOD"

  echo "[STEP2] Aggregating metrics (lockbox: --no_sweep)"
  python3 "$PHASE3_AGG" \
    --in_jsonl "$OUT_JSONL" \
    --out_dir "$OUT_DIR" \
    --prefix "$PREFIX" \
    --no_sweep

  echo "[DONE] $DS"
  echo
done

echo "[ALL DONE] Test lockbox run complete. Outputs:"
echo "  JSONL   : $TEST_LOG_DIR/<dataset>_theta${IDX}_run.jsonl"
echo "  METRICS : $TEST_METRICS_DIR/theta${IDX}/<dataset>/"
