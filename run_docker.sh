#!/bin/bash
# Sample script to run and test the Docker setup

echo "Building Docker images..."
docker-compose build

echo "Starting MCP server..."
docker-compose up -d mcp-server

# Wait for server to start
echo "Waiting for server to start..."
sleep 10

# Check if the server is running
echo "Checking server health..."
curl -s http://localhost:8000/health | grep -q "healthy"
if [ $? -eq 0 ]; then
    echo "✅ MCP server is running correctly"
else
    echo "❌ MCP server failed to start"
    docker-compose logs mcp-server
    exit 1
fi

# Test a simple query
echo "Testing a sample query..."
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How to use this system?", "top_k": 3}' | python -m json.tool

# Start other services
echo "Starting file watcher and scheduled vectorization..."
docker-compose up -d

# Print status
echo "Docker services status:"
docker-compose ps

echo "Setup complete! The system is now running."
echo
echo "To view logs:"
echo "  docker-compose logs -f"
echo
echo "To stop all services:"
echo "  docker-compose down"
