import os
import logging
import logging.config

# デフォルト設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "models"))
VECTOR_DB_PATH = os.environ.get("VECTOR_DB_PATH", os.path.join(BASE_DIR, "vector_store", "vector_store.db"))
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", os.path.join(BASE_DIR, "models", "plamo-embedding-1b"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "DEBUG")  # デバッグ用にDEBUGレベルに変更

# ログファイルのパス
LOG_DIR = os.environ.get("LOG_DIR", os.path.join(BASE_DIR, "logs"))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "vectorization.log")

# 詳細なロギング設定
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
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': LOG_FILE,
            'formatter': 'detailed',
            'level': LOG_LEVEL,
        }
    },
    'loggers': {
        'vectorize': {
            'handlers': ['console', 'file'],
            'level': LOG_LEVEL,
            'propagate': True
        },
        'workers': {
            'handlers': ['console', 'file'],
            'level': LOG_LEVEL,
            'propagate': True
        },
        'db': {
            'handlers': ['console', 'file'],
            'level': LOG_LEVEL,
            'propagate': True
        }
    }
}

# ロギング設定を適用
logging.config.dictConfig(LOGGING_CONFIG)

# モニタリング関連設定
MONITORING_INTERVAL = int(os.environ.get("MONITORING_INTERVAL", "30"))
METRICS_ENABLED = os.environ.get("METRICS_ENABLED", "true").lower() == "true"

# SQLite vector extension related settings
SQLITE_VEC_LIB_PATH = os.environ.get("SQLITE_VEC_LIB_PATH", "")
FALLBACK_TO_BASIC_SEARCH = os.environ.get("FALLBACK_TO_BASIC_SEARCH", "true").lower() == "true"