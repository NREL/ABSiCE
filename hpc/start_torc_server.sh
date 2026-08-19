#!/bin/bash
# Start the torc server and export the API URL.
#
# Usage:
#   ./start_torc_server.sh                      # uses current hostname, OS-assigned port
#   ./start_torc_server.sh --host kl1           # use a specific node
#   ./start_torc_server.sh --port 52619         # use a specific port
#   ./start_torc_server.sh --host kl1 --port 52619

set -euo pipefail

# Default host to the current login node's fully-qualified hostname
DEFAULT_HOST="$(hostname -f)"
HOST="$DEFAULT_HOST"
PORT_ARG="0"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)
            # Accept short name (kl1) or FQDN
            if [[ "$2" != *.* ]]; then
                HOST="${2}.hsn.cm.kestrel.hpc.nlr.gov"
            else
                HOST="$2"
            fi
            shift 2
            ;;
        --port)
            PORT_ARG="$2"
            shift 2
            ;;
        *)
            # Legacy positional argument: port only
            PORT_ARG="$1"
            shift
            ;;
    esac
done

DB="torc.db"
LOG_FILE="torc_server.log"

# Step 1: Add torc binary to PATH.
# Either point this at your local torc server install, or use a shared
# cluster path, e.g. export PATH="<TORC_BINARY_DIR>:$PATH"
export PATH="<TORC_BINARY_DIR>:$PATH"

# Step 2: Start the server in the background
echo "Starting torc-server on port ${PORT_ARG} (0 = OS-assigned random port)..."
torc-server run \
    --database "$DB" \
    --host "$HOST" \
    --port "$PORT_ARG" \
    --completion-check-interval-secs 5 \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "torc-server PID: $SERVER_PID"

# Step 3: Determine the actual port.
# If a specific port was given, use it directly.
# Otherwise, wait for the server to log the OS-assigned port.
if [[ "$PORT_ARG" != "0" ]]; then
    PORT="$PORT_ARG"
else
    echo "Waiting for server to report its assigned port..."
    TIMEOUT=30
    ELAPSED=0
    PORT=""
    while [[ $ELAPSED -lt $TIMEOUT ]]; do
        if [[ -f "$LOG_FILE" ]]; then
            PORT=$(grep -oP "(?<=${HOST}:)\d+(?=/torc-service)" "$LOG_FILE" | head -1 || true)
            if [[ -n "$PORT" ]]; then
                break
            fi
        fi
        sleep 1
        ELAPSED=$((ELAPSED + 1))
    done

    if [[ -z "$PORT" ]]; then
        echo "ERROR: Could not detect the server port from $LOG_FILE after ${TIMEOUT}s." >&2
        echo "Check $LOG_FILE for startup errors." >&2
        exit 1
    fi
fi

echo "Server started on port: $PORT"

# Step 4: Export the API URL
export TORC_API_URL="http://${HOST}:${PORT}/torc-service/v1"
echo "TORC_API_URL=${TORC_API_URL}"
echo ""
echo "To use this URL in your current shell, run:"
echo "  export TORC_API_URL=${TORC_API_URL}"
