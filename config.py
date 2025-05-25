import os
import logging
import logging.config

# Base directory of the project (where config.py is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Core Paths ---
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "models"))
LOG_DIR = os.environ.get("LOG_DIR", os.path.join(BASE_DIR, "logs"))
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- File Watcher Settings ---
_default_watched_dir = os.path.join(BASE_DIR, "sample_documents")
if not os.path.exists(_default_watched_dir):
    os.makedirs(_default_watched_dir, exist_ok=True)
WATCHED_DIRS_STR = os.environ.get("WATCHED_DIRS", _default_watched_dir)
WATCHED_DIRS = [d.strip() for d in WATCHED_DIRS_STR.split(',')]

SUPPORTED_EXTENSIONS = {
    '.pdf', '.md', '.markdown', '.txt', '.py', # Simplified for brevity
}

# --- Vector Database Settings ---
VECTOR_DB_PATH = os.environ.get("VECTOR_DB_PATH", os.path.join(DATA_DIR, "test_memory_bank.db"))
os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
VECTOR_DB_DIMENSION = int(os.environ.get("VECTOR_DB_DIMENSION", "384")) # Dummy dimension
VECTOR_DB_VSS_ENABLED = os.environ.get("VECTOR_DB_VSS_ENABLED", "False").lower() == 'true' # Disable VSS for simpler test run initially

# --- Embedding Model Settings ---
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", os.path.join(MODEL_DIR, "dummy_model")) # Path for dummy if needed
USE_DUMMY_EMBEDDER = os.environ.get("USE_DUMMY_EMBEDDER", "True").lower() == 'true' # Default to True for testing
EMBEDDING_MODEL_TRUST_REMOTE_CODE = os.environ.get("EMBEDDING_MODEL_TRUST_REMOTE_CODE", "True").lower() == 'true'


# --- MCP Server Settings ---
MCP_SERVER_HOST = os.environ.get("MCP_SERVER_HOST", "127.0.0.1")
MCP_SERVER_PORT = int(os.environ.get("MCP_SERVER_PORT", "8000"))

# --- Logging Settings ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.path.join(LOG_DIR, "memory_bank_test.log")

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'detailed',
            'level': LOG_LEVEL,
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': LOG_FILE,
            'maxBytes': 1024*1024*2, # 2 MB
            'backupCount': 2,
            'formatter': 'detailed',
            'level': LOG_LEVEL,
        }
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': LOG_LEVEL,
    },
}
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)
logger.info("Test configuration loaded. LOG_LEVEL: %s, USE_DUMMY_EMBEDDER: %s", LOG_LEVEL, USE_DUMMY_EMBEDDER)

# --- Original Monitoring/Fallback Settings (can be removed if not used by new design) ---
MONITORING_INTERVAL = int(os.environ.get("MONITORING_INTERVAL", "300")) # Less frequent for tests
METRICS_ENABLED = os.environ.get("METRICS_ENABLED", "False").lower() == "true"
SQLITE_VEC_LIB_PATH = os.environ.get("SQLITE_VEC_LIB_PATH", "") # VSS path, VectorStore handles finding it
FALLBACK_TO_BASIC_SEARCH = os.environ.get("FALLBACK_TO_BASIC_SEARCH", "True").lower() == 'true' # Covered by VECTOR_DB_VSS_ENABLED