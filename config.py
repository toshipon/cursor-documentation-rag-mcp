import os

# デフォルト設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "models"))
VECTOR_DB_PATH = os.environ.get("VECTOR_DB_PATH", os.path.join(BASE_DIR, "vector_store", "vector_store.db"))
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", os.path.join(MODEL_DIR, "plamo-embedding-1b"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# モニタリング関連設定
MONITORING_INTERVAL = int(os.environ.get("MONITORING_INTERVAL", "30"))  # 秒単位
METRICS_ENABLED = os.environ.get("METRICS_ENABLED", "true").lower() == "true"

# その他設定を必要に応じて追加