FROM python:3.10-slim

# Install dependencies for building packages
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  build-essential \
  git \
  curl \
  libgomp1 libatlas-base-dev liblapack-dev \
  sqlite3 \
  sqlite3-dev \
  python3-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# Build and install sqlite-vec from source for the proper architecture
RUN cd /tmp && \
  git clone https://github.com/asg017/sqlite-vec.git && \
  cd sqlite-vec && \
  make loadable && \
  mkdir -p /usr/local/lib/sqlite-vec/ && \
  cp dist/debug/vec0.so /usr/local/lib/sqlite-vec/ && \
  chmod +x /usr/local/lib/sqlite-vec/vec0.so

WORKDIR /app

RUN pip install --upgrade pip

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional performance-related packages
RUN pip install --no-cache-dir \
  uvloop \
  httptools \
  gunicorn \
  cachetools \
  huggingface-hub==0.19.4 \
  transformers==4.36.1

# Copy application code
COPY . .

# Create necessary directories with appropriate permissions
RUN mkdir -p /app/data /app/models /app/vector_store && \
  chmod -R 755 /app/data /app/models /app/vector_store

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Set Python optimizations
ENV PYTHONOPTIMIZE=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

# Set environment variables for vector extension
ENV SQLITE_VEC_LIB_PATH=/usr/local/lib/sqlite-vec/vec0.so
ENV FALLBACK_TO_BASIC_SEARCH=true

# Expose ports for MCP server
EXPOSE 8000

# Set entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Default command to run when starting the container
CMD ["start-mcp-server"]