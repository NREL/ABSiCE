#!/usr/bin/env bash
# Submit an ABSiCE sensitivity workflow to Slurm via TORC.
#
# Usage:
#   TORC_ACCOUNT=myproject ./hpc/submit.sh [recycling|transport|att_calibration] [extra torc args...]
#
# Required environment variable:
#   TORC_ACCOUNT  — Slurm account name (e.g. "myproject")
#
# Examples:
#   TORC_ACCOUNT=pvabm ./hpc/submit.sh recycling
#   TORC_ACCOUNT=pvabm ./hpc/submit.sh att_calibration
#   TORC_ACCOUNT=pvabm ./hpc/submit.sh recycling --poll-interval 30
#
# NOTE: all three torc_*.yaml specs already define `slurm_schedulers:`
# themselves (account/partition/walltime/mail are pinned in the YAML), so
# this script always submits directly — it does NOT call
# `torc slurm generate`, which does not accept `--partition`/`--nodes`
# flags in the installed TORC version (v0.40) and would fail if invoked.
# TORC_ACCOUNT is still required as a submission-time sanity check.
#
# Each run is stamped with a timestamp; torc logs go to torc_output/<run_name>/

set -euo pipefail

WORKFLOW="${1:-recycling}"
shift || true   # remaining args forwarded to torc

# Validate workflow argument. transport is accepted for CLI parity/routing
# only — hpc/torc_transport_sensitivity.yaml is out of scope this session
# and has not been validated against the new run_single_scenario.py CLI.
if [[ "$WORKFLOW" != "recycling" && "$WORKFLOW" != "transport" && "$WORKFLOW" != "att_calibration" ]]; then
    echo "Error: workflow must be 'recycling', 'transport', or 'att_calibration', got '$WORKFLOW'" >&2
    exit 1
fi

# Require TORC_ACCOUNT
if [[ -z "${TORC_ACCOUNT:-}" ]]; then
    echo "Error: TORC_ACCOUNT environment variable is not set." >&2
    echo "  Usage: TORC_ACCOUNT=myproject ./hpc/submit.sh [recycling|att_calibration]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
# att_calibration maps to its own YAML; other workflows use the _sensitivity suffix.
if [[ "$WORKFLOW" == "att_calibration" ]]; then
    YAML_FILE="$SCRIPT_DIR/torc_att_calibration.yaml"
else
    YAML_FILE="$SCRIPT_DIR/torc_${WORKFLOW}_sensitivity.yaml"
fi

if [[ ! -f "$YAML_FILE" ]]; then
    echo "Error: workflow file not found: $YAML_FILE" >&2
    exit 1
fi

if ! grep -q '^slurm_schedulers:' "$YAML_FILE"; then
    echo "Error: $YAML_FILE does not define slurm_schedulers: — this script" >&2
    echo "  only supports specs with a manually-defined scheduler block." >&2
    exit 1
fi

# Timestamp-stamped run name and torc output directory
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="absice_${WORKFLOW}_sensitivity_${TIMESTAMP}"
OUTPUT_DIR="torc_output/${RUN_NAME}"

echo "Submitting: $YAML_FILE"
echo "  Run name : $RUN_NAME"
echo "  Output   : $OUTPUT_DIR"
echo "  Account  : $TORC_ACCOUNT"
echo "  Workspace: $WORKSPACE_DIR"
echo ""

# Run from workspace root so relative paths in the YAML resolve correctly.
cd "$WORKSPACE_DIR"

# Capture torc submit stdout to extract workflow ID for post-submission verification.
_TMPOUT=$(mktemp)
trap 'rm -f "$_TMPOUT"' EXIT

sed "s|^name:.*|name: ${RUN_NAME}|" "$YAML_FILE" \
  | torc submit - --output-dir "$OUTPUT_DIR" "$@" \
  | tee "$_TMPOUT"

# ── Verify job count ──────────────────────────────────────────────────────────
WORKFLOW_ID=$(grep -oP '(?<=workflow )\d+' "$_TMPOUT" | tail -1)
if [[ -n "$WORKFLOW_ID" ]]; then
    ACTUAL_JOBS=$(torc -f json jobs list "$WORKFLOW_ID" 2>/dev/null \
                  | jq '.items | length' 2>/dev/null || true)
    echo ""
    echo "Verification: workflow $WORKFLOW_ID — ${ACTUAL_JOBS:-?} job(s) created."
fi
