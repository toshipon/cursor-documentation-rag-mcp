#!/bin/bash
set -e

# Setup data directories if they don't exist
mkdir -p /app/data
mkdir -p /app/models
mkdir -p /app/vector_store

# Function to check if SQLite vector extension is working
check_sqlite_vec() {
    echo "Checking SQLite vector extension..."
    python -c "
import sqlite3
try:
    conn = sqlite3.connect(':memory:')
    conn.enable_load_extension(True)
    conn.load_extension('$SQLITE_VEC_LIB_PATH')
    conn.execute('CREATE VIRTUAL TABLE vss_test USING vec(document(3))')
    conn.execute(\"INSERT INTO vss_test(rowid, document) VALUES (1, '[1.0, 0.0, 0.0]')\")
    print('SQLite vector extension loaded successfully!')
    exit(0)
except Exception as e:
    print(f'Error loading SQLite vector extension: {e}')
    exit(1)
"
    return $?
}

# Setup SQLite vector extension
if [ -f "$SQLITE_VEC_LIB_PATH" ]; then
    if check_sqlite_vec; then
        echo "Vector search extension loaded and working properly."
        export FALLBACK_TO_BASIC_SEARCH=false
    else
        echo "Vector search extension found but not working. Using fallback implementation."
        export FALLBACK_TO_BASIC_SEARCH=true
    fi
else
    echo "Vector search extension not found at $SQLITE_VEC_LIB_PATH. Using fallback implementation."
    export FALLBACK_TO_BASIC_SEARCH=true
fi

# Download PLaMo model if it doesn't exist
if [ ! -f "/app/models/plamo-embedding-1b/pytorch_model.bin" ] && [ ! -f "/app/models/plamo-embedding-1b/model.safetensors" ]; then
    echo "Downloading PLaMo-Embedding-1B model..."
    python /app/scripts/download_model.py --model-path /app/models/plamo-embedding-1b
    
    # Check if the download was successful
    if [ ! -f "/app/models/plamo-embedding-1b/pytorch_model.bin" ] && [ ! -f "/app/models/plamo-embedding-1b/model.safetensors" ]; then
        echo "ERROR: Failed to download model weights. Please check network connectivity and try again."
        exit 1
    fi
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
    exec python scripts/start_file_watcher.py "$@"
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