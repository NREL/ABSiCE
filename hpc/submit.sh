#!/usr/bin/env bash
# Submit an ABSiCE sensitivity or calibration workflow to Slurm via TORC.
#
# Usage:
#   TORC_ACCOUNT=myproject ./hpc/submit.sh [workflow] [extra torc args...]
#
# Supported workflows:
#   RTN scenarios:
#     recycling              — Scale recycling costs across ratios
#     transport              — Scale transport costs across ratios
#     att_calibration        — Calibrate att_mean (RTN-based, uses cost rates)
#   Calibration (no RTN, constant costs):
#     calibration_sweep      — Sequential sweep of att_mean values (1 job, 16 CPUs)
#     calibration_sweep_parallel — Parallel sweep of att_mean values (10 jobs)
#
# Required environment variable:
#   TORC_ACCOUNT  — Slurm account name (e.g. "myproject")
#
# Optional environment variables:
#   TORC_PARTITION — Slurm partition (passed as --partition; overrides YAML placeholder)
#   TORC_NODES     — Number of Slurm nodes to request (default: from YAML)
#
# Examples:
#   TORC_ACCOUNT=solar ./hpc/submit.sh recycling
#   TORC_ACCOUNT=solar TORC_PARTITION=short ./hpc/submit.sh transport
#   TORC_ACCOUNT=solar ./hpc/submit.sh att_calibration
#   TORC_ACCOUNT=solar ./hpc/submit.sh calibration_sweep
#   TORC_ACCOUNT=solar ./hpc/submit.sh calibration_sweep_parallel
#   TORC_ACCOUNT=solar ./hpc/submit.sh recycling --poll-interval 30
#
# Each run is stamped with a timestamp; torc logs go to torc_output/<run_name>/

set -euo pipefail

WORKFLOW="${1:-recycling}"
shift || true   # remaining args forwarded to torc

# Validate workflow argument
if [[ "$WORKFLOW" != "recycling" && "$WORKFLOW" != "transport" && "$WORKFLOW" != "att_calibration" && "$WORKFLOW" != "calibration_sweep" && "$WORKFLOW" != "calibration_sweep_parallel" ]]; then
    echo "Error: workflow must be one of: recycling, transport, att_calibration, calibration_sweep, calibration_sweep_parallel" >&2
    echo "Got: '$WORKFLOW'" >&2
    exit 1
fi

# Require TORC_ACCOUNT
if [[ -z "${TORC_ACCOUNT:-}" ]]; then
    echo "Error: TORC_ACCOUNT environment variable is not set." >&2
    echo "  Usage: TORC_ACCOUNT=myproject ./hpc/submit.sh [recycling|transport]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

# Map workflow to YAML file
case "$WORKFLOW" in
    att_calibration)
        YAML_FILE="$SCRIPT_DIR/torc_att_calibration.yaml"
        ;;
    calibration_sweep)
        YAML_FILE="$SCRIPT_DIR/torc_calibration_sweep.yaml"
        ;;
    calibration_sweep_parallel)
        YAML_FILE="$SCRIPT_DIR/torc_calibration_sweep_parallel.yaml"
        ;;
    *)
        # recycling, transport — add _sensitivity suffix
        YAML_FILE="$SCRIPT_DIR/torc_${WORKFLOW}_sensitivity.yaml"
        ;;
esac

if [[ ! -f "$YAML_FILE" ]]; then
    echo "Error: workflow file not found: $YAML_FILE" >&2
    exit 1
fi

# Timestamp-stamped run name and torc output directory
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# Build descriptive run name based on workflow type
case "$WORKFLOW" in
    calibration_sweep|calibration_sweep_parallel)
        RUN_NAME="absice_${WORKFLOW}_${TIMESTAMP}"
        ;;
    *)
        RUN_NAME="absice_${WORKFLOW}_sensitivity_${TIMESTAMP}"
        ;;
esac

OUTPUT_DIR="torc_output/${RUN_NAME}"

# Build optional extra flags for torc slurm generate
GENERATE_EXTRA=()
if [[ -n "${TORC_PARTITION:-}" ]]; then
    GENERATE_EXTRA+=(--partition "$TORC_PARTITION")
fi
if [[ -n "${TORC_NODES:-}" ]]; then
    GENERATE_EXTRA+=(--nodes "$TORC_NODES")
fi

echo "Submitting: $YAML_FILE"
echo "  Run name : $RUN_NAME"
echo "  Output   : $OUTPUT_DIR"
echo "  Account  : $TORC_ACCOUNT"
echo "  Partition: ${TORC_PARTITION:-<from YAML>}"
echo "  Workspace: $WORKSPACE_DIR"
echo ""

# Run from workspace root so relative paths in the YAML resolve correctly.
cd "$WORKSPACE_DIR"

# Capture torc submit stdout to extract workflow ID for post-submission verification.
_TMPOUT=$(mktemp)
trap 'rm -f "$_TMPOUT"' EXIT

# If the spec already defines slurm_schedulers, skip 'torc slurm generate'
# (which would conflict) and submit directly.
if grep -q '^slurm_schedulers:' "$YAML_FILE"; then
    echo "Note: slurm_schedulers already defined in spec — skipping 'torc slurm generate'."
    sed "s|^name:.*|name: ${RUN_NAME}|" "$YAML_FILE" \
      | torc submit - --output-dir "$OUTPUT_DIR" "$@" \
      | tee "$_TMPOUT"
else
    # Generate Slurm-annotated spec and pipe directly to submit.
    sed "s|^name:.*|name: ${RUN_NAME}|" "$YAML_FILE" \
      | torc slurm generate \
            --account "$TORC_ACCOUNT" \
            "${GENERATE_EXTRA[@]}" \
            - \
      | torc submit - --output-dir "$OUTPUT_DIR" "$@" \
      | tee "$_TMPOUT"
fi

# ── Verify job count ──────────────────────────────────────────────────────────
WORKFLOW_ID=$(grep -oP '(?<=workflow )\d+' "$_TMPOUT" | tail -1)
if [[ -n "$WORKFLOW_ID" ]]; then
    ACTUAL_JOBS=$(torc -f json jobs list "$WORKFLOW_ID" 2>/dev/null \
                  | jq '.items | length' 2>/dev/null || true)
    echo ""
    echo "Verification: workflow $WORKFLOW_ID — ${ACTUAL_JOBS:-?} job(s) created."
fi
