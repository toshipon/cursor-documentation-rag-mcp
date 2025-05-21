#!/bin/bash
set -e

# Setup data directories if they don't exist
mkdir -p /app/data
mkdir -p /app/models
mkdir -p /app/vector_store

# Download PLaMo model if it doesn't exist
if [ ! -f "/app/models/plamo-embedding-1b/config.json" ]; then
    echo "Downloading PLaMo-Embedding-1B model..."
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='pfnet/plamo-embedding-1b', local_dir='/app/models/plamo-embedding-1b')"
    echo "Model downloaded successfully"
fi

# Command mapping
function start-mcp-server() {
    echo "Starting MCP Server with $WORKERS workers..."
    exec gunicorn -k uvicorn.workers.UvicornWorker \
        --workers $WORKERS \
        --bind 0.0.0.0:8000 \
        --timeout 120 \
        --access-logfile - \
        --error-logfile - \
        "mcp.server:app"
}

function start-file-watcher() {
    echo "Starting File Watcher..."
    exec python scripts/start_file_watcher.py
}

function scheduled-vectorization() {
    if [ "$ENABLE_SCHEDULED_VECTORIZATION" = "true" ]; then
        echo "Starting Scheduled Vectorization with interval: $SCHEDULE_INTERVAL seconds..."
        exec python scripts/scheduled_vectorization.py --interval $SCHEDULE_INTERVAL
    else
        echo "Scheduled Vectorization is disabled. Exiting..."
        exit 0
    fi
}

function vectorize-docs() {
    echo "Vectorizing Documents..."
    exec python scripts/vectorize_docs.py "$@"
}

function help() {
    echo "Available commands:"
    echo "  start-mcp-server - Start the MCP server"
    echo "  start-file-watcher - Start the file watcher service"
    echo "  scheduled-vectorization - Start the scheduled vectorization service"
    echo "  vectorize-docs [args] - Run document vectorization with optional arguments"
    echo "  help - Display this help message"
}

# Execute command based on first argument or default to help
if [[ $1 == "start-mcp-server" ]]; then
    start-mcp-server
elif [[ $1 == "start-file-watcher" ]]; then
    start-file-watcher
elif [[ $1 == "scheduled-vectorization" ]]; then
    scheduled-vectorization
elif [[ $1 == "vectorize-docs" ]]; then
    shift
    vectorize-docs "$@"
elif [[ $1 == "help" ]]; then
    help
elif [[ -z $1 ]]; then
    help
else
    # If command is not recognized, try to execute it directly
    exec "$@"
fi