#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <IDX> <METHOD> [RUN_TAG] [--no_sweep] [--force]"
  echo ""
  echo "  IDX        : theta index used in filenames, e.g., 3 -> theta3"
  echo "  METHOD     : method string recorded in JSONL, e.g., llm_reasoner_theta3"
  echo "  RUN_TAG    : optional string used in run_id; default is current time YYYY-MM-DD-HHMM"
  echo "  --no_sweep : pass --no_sweep to phase3_02_aggregate_metrics.py"
  echo "  --force    : delete existing JSONL + metrics for this IDX before running (clean re-run)"
  exit 2
fi

IDX="$1"
METHOD="$2"
shift 2

# Optional RUN_TAG (only if next arg is not an option)
RUN_TAG="$(date +%Y-%m-%d-%H%M)"
if [[ $# -gt 0 && "${1:0:1}" != "-" ]]; then
  RUN_TAG="$1"
  shift
fi

NO_SWEEP=0
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no_sweep) NO_SWEEP=1 ;;
    --force)    FORCE=1 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
  shift
done

# Resolve paths based on this script's location (NOT based on current working directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PHASE3_RUN="$SCRIPT_DIR/phase3_01_run_eval_ros.py"
PHASE3_AGG="$SCRIPT_DIR/phase3_02_aggregate_metrics.py"

DEV_ROOT="$REPO_ROOT/config/dev"
DATASET_DIR="$DEV_ROOT/dev_dataset"
LOG_DIR="$DEV_ROOT/dev_log"
METRICS_ROOT="$DEV_ROOT/dev_metrics"

# Sanity checks
if [[ ! -f "$PHASE3_RUN" ]]; then
  echo "ERROR: Not found: $PHASE3_RUN" >&2
  exit 1
fi
if [[ ! -f "$PHASE3_AGG" ]]; then
  echo "ERROR: Not found: $PHASE3_AGG" >&2
  exit 1
fi
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "ERROR: Not found dataset dir: $DATASET_DIR" >&2
  echo "Expected: $REPO_ROOT/config/dev/dev_dataset" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$METRICS_ROOT"

echo "[INFO] SCRIPT_DIR   = $SCRIPT_DIR"
echo "[INFO] REPO_ROOT    = $REPO_ROOT"
echo "[INFO] DEV_ROOT     = $DEV_ROOT"
echo "[INFO] DATASET_DIR  = $DATASET_DIR"
echo "[INFO] LOG_DIR      = $LOG_DIR"
echo "[INFO] METRICS_ROOT = $METRICS_ROOT"
echo "[INFO] IDX=$IDX METHOD=$METHOD RUN_TAG=$RUN_TAG NO_SWEEP=$NO_SWEEP FORCE=$FORCE"
echo ""

DATASETS=(dev_core dev_conflict dev_noisy dev_oom)

for DS in "${DATASETS[@]}"; do
  DATASET_JSON="$DATASET_DIR/${DS}.json"
  OUT_JSONL="$LOG_DIR/${DS}_theta${IDX}_run.jsonl"
  RUN_ID="${RUN_TAG}_theta${IDX}_${DS}"

  OUT_DIR="$METRICS_ROOT/theta${IDX}/${DS}/"
  PREFIX="${DS}_theta${IDX}_"

  if [[ ! -f "$DATASET_JSON" ]]; then
    echo "ERROR: Dataset not found: $DATASET_JSON" >&2
    exit 1
  fi

  if [[ $FORCE -eq 1 ]]; then
    echo "[FORCE] Removing existing outputs for $DS theta${IDX}"
    rm -f "$OUT_JSONL"
    rm -rf "$OUT_DIR"
  fi

  echo "========== [RUN] ${DS} (theta${IDX}) =========="
  python3 "$PHASE3_RUN" \
    --dataset "$DATASET_JSON" \
    --out "$OUT_JSONL" \
    --run_id "$RUN_ID" \
    --method "$METHOD"

  echo "---------- [AGG] ${DS} (theta${IDX}) ----------"
  mkdir -p "$OUT_DIR"
  if [[ $NO_SWEEP -eq 1 ]]; then
    python3 "$PHASE3_AGG" \
      --in_jsonl "$OUT_JSONL" \
      --out_dir "$OUT_DIR" \
      --prefix "$PREFIX" \
      --no_sweep
  else
    python3 "$PHASE3_AGG" \
      --in_jsonl "$OUT_JSONL" \
      --out_dir "$OUT_DIR" \
      --prefix "$PREFIX"
  fi

  echo ""
done

echo "[DONE] Completed Dev run+aggregate for theta${IDX} across: ${DATASETS[*]}"
