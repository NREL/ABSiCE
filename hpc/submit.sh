#!/usr/bin/env bash
# Submit an ABSiCE sensitivity workflow to Slurm via TORC.
#
# Usage:
#   TORC_ACCOUNT=myproject ./hpc/submit.sh [recycling|transport] [extra torc args...]
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
#   TORC_ACCOUNT=solar ./hpc/submit.sh recycling --poll-interval 30
#
# Each run is stamped with a timestamp; torc logs go to torc_output/<run_name>/

set -euo pipefail

WORKFLOW="${1:-recycling}"
shift || true   # remaining args forwarded to torc

# Validate workflow argument
if [[ "$WORKFLOW" != "recycling" && "$WORKFLOW" != "transport" ]]; then
    echo "Error: workflow must be 'recycling' or 'transport', got '$WORKFLOW'" >&2
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
YAML_FILE="$SCRIPT_DIR/torc_${WORKFLOW}_sensitivity.yaml"

if [[ ! -f "$YAML_FILE" ]]; then
    echo "Error: workflow file not found: $YAML_FILE" >&2
    exit 1
fi

# Timestamp-stamped run name and torc output directory
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="absice_${WORKFLOW}_sensitivity_${TIMESTAMP}"
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

# If the spec already defines slurm_schedulers, skip 'torc slurm generate'
# (which would conflict) and submit directly.
if grep -q '^slurm_schedulers:' "$YAML_FILE"; then
    echo "Note: slurm_schedulers already defined in spec — skipping 'torc slurm generate'."
    sed "s|^name:.*|name: ${RUN_NAME}|" "$YAML_FILE" \
      | torc submit - --output-dir "$OUTPUT_DIR" "$@"
else
    # Generate Slurm-annotated spec and pipe directly to submit.
    sed "s|^name:.*|name: ${RUN_NAME}|" "$YAML_FILE" \
      | torc slurm generate \
            --account "$TORC_ACCOUNT" \
            "${GENERATE_EXTRA[@]}" \
            - \
      | torc submit - --output-dir "$OUTPUT_DIR" "$@"
fi
