# Docker Setup for Cursor Document RAG MCP System

This directory contains Docker configuration files for deploying the Cursor Documentation RAG MCP system.

## Components

- `Dockerfile`: Container definition for the system
- `entrypoint.sh`: Script to handle various commands within the container
- `docker-compose.yml`: Multi-container configuration for running all system components

## Available Services

The docker-compose configuration includes three services:

1. **mcp-server**: Runs the MCP server to handle vector search requests
2. **file-watcher**: Monitors document directories for changes and triggers vectorization
3. **scheduled-vectorization**: Runs periodic scanning of documents

## Data Volumes

The setup uses the following volume mounts:

- `./data`: Contains all documents to be processed
- `./models`: Contains embedding models (PLaMo-Embedding-1B)
- `./vector_store`: Contains the SQLite vector database

## Environment Variables

You can customize the following environment variables:

- `DATA_DIR`: Directory containing documents to vectorize
- `MODEL_DIR`: Directory containing embedding models
- `VECTOR_DB_PATH`: Path to the SQLite vector database
- `EMBEDDING_MODEL_PATH`: Path to the PLaMo-Embedding-1B model
- `LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING, ERROR)

## Usage

### Building and Starting All Services

```bash
docker-compose up -d
```

### Starting Individual Services

```bash
docker-compose up -d mcp-server
docker-compose up -d file-watcher
docker-compose up -d scheduled-vectorization
```

### Manually Vectorizing Documents

```bash
docker-compose run --rm mcp-server vectorize-docs
```

### Viewing Logs

```bash
docker-compose logs -f mcp-server
docker-compose logs -f file-watcher
docker-compose logs -f scheduled-vectorization
```

### Stopping Services

```bash
docker-compose down
```
