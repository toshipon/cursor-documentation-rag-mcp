import sqlite3
import os

class VectorStore:
    def __init__(self, db_path):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT,
            source TEXT,
            source_type TEXT,
            vector BLOB
        )
        ''')
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS file_metadata (
            file_path TEXT PRIMARY KEY,
            last_modified INTEGER,
            file_hash TEXT,
            last_vectorized INTEGER,
            status TEXT
        )
        ''')
        self.conn.commit()

    def add_documents(self, docs, vectors, file_path=None):
        import time, hashlib
        for doc, vec in zip(docs, vectors):
            self.conn.execute(
                "INSERT INTO documents (content, source, source_type, vector) VALUES (?, ?, ?, ?)",
                (doc["content"], doc["metadata"]["source"], doc["metadata"]["source_type"], bytes(str(vec), "utf-8"))
            )
        # ファイルメタデータも登録
        if file_path and os.path.exists(file_path):
            mtime = int(os.path.getmtime(file_path))
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            now = int(time.time())
            self.conn.execute(
                "INSERT OR REPLACE INTO file_metadata (file_path, last_modified, file_hash, last_vectorized, status) VALUES (?, ?, ?, ?, ?)",
                (file_path, mtime, file_hash, now, "processed")
            )
        self.conn.commit()

    def similarity_search(self, query_vector, top_k=5):
        # ダミー: 全件からランダムにtop_k件返す（本来はベクトル検索）
        cursor = self.conn.execute("SELECT content, source, source_type FROM documents LIMIT ?", (top_k,))
        return [
            {
                "content": row[0],
                "metadata": {
                    "source": row[1],
                    "source_type": row[2]
                },
                "score": 0.0
            }
            for row in cursor.fetchall()
        ]