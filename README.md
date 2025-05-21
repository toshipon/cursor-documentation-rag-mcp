# Cursor Documentation RAG MCP System

## Overview
A system that vectorizes and searches through various document types (Markdown, PDF, source code) using PLaMo-Embedding-1B. It provides vector search capabilities via an MCP (Memory Context Protocol) server to allow Cursor to search through documentation efficiently.

## Key Features
- Document vectorization using PLaMo-Embedding-1B
- Vector storage with SQLite-VSS
- FastAPI-based MCP server with performance optimizations
- Support for Markdown, PDF, and code documents
- File change monitoring for automatic updates
- Scheduled document scanning
- Batch query processing with caching
- Docker deployment with resource management

## Setup

### Using Docker (Recommended)

```bash
# Start all services
docker-compose up -d

# Or start individual services
docker-compose up -d mcp-server
docker-compose up -d file-watcher
docker-compose up -d scheduled-vectorization
```

### Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download the PLaMo-Embedding-1B model
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='pfnet/plamo-embedding-1b', local_dir='models/plamo-embedding-1b')"

# Vectorize documents
python scripts/vectorize_docs.py

# Start the MCP server
python scripts/start_mcp_server.py
```

## System Components

### Core Components
- **Embedding Engine**: Converts text to vector embeddings using PLaMo-Embedding-1B
- **Text Splitters**: Handles different document types with chunking strategies
- **Document Processors**: Specialized processors for Markdown, code, and PDF files
- **Vector Database**: SQLite-VSS for vector similarity search
- **MCP Server**: FastAPI server with query endpoints

### Automation Components
- **File Watcher**: Monitors file changes and triggers vectorization
- **Vectorization Worker**: Processes file change events
- **Scheduler**: Periodic document scanning

## Performance Optimizations
- **Caching**: Both query results and embeddings are cached
- **Batch Processing**: Support for batch query operations
- **Resource Management**: Container memory limits
- **Worker Scaling**: Configurable worker processes for the API server
- **Advanced Filtering**: Metadata-based filtering capabilities

## Configuration
The system can be configured via environment variables or the `.env` file:

- `DATA_DIR`: Directory containing documents to vectorize
- `MODEL_DIR`: Directory containing the embedding model
- `VECTOR_DB_PATH`: Path to the SQLite vector database
- `EMBEDDING_MODEL_PATH`: Path to the PLaMo model
- `WORKERS`: Number of worker processes for the API server
- `BATCH_SIZE`: Batch size for embedding generation
- `MAX_BATCH_SIZE`: Maximum batch size for query processing
- `SCHEDULE_INTERVAL`: Interval for scheduled vectorization (seconds)

## API Endpoints

- `POST /query`: Search for similar documents
- `POST /batch_query`: Batch search for multiple queries
- `GET /stats`: Get database statistics
- `GET /health`: Health check endpoint

## Directory Structure

- `vectorize/`: Embedding and text splitting logic
- `db/`: Vector database implementation
- `mcp/`: MCP server implementation
- `scripts/`: Utility scripts
- `workers/`: File watcher and worker implementation
- `docker/`: Docker configuration files
- `models/`: Embedding models

## License
MIT