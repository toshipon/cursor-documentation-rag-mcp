import os

# デフォルト設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "models"))
VECTOR_DB_PATH = os.environ.get("VECTOR_DB_PATH", os.path.join(BASE_DIR, "vector_store", "vector_store.db"))
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", os.path.join(MODEL_DIR, "plamo-embedding-1b"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# その他設定を必要に応じて追加