import os
import json
import time
import sqlite3
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class VectorStore:
    """SQLiteを使用したベクトルストアの実装"""
    
    def __init__(self, db_path: str, vector_dimension: int = 512):
        """初期化"""
        self.db_path = db_path
        self.vector_dimension = vector_dimension
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
        logger.info(f"Initialized SQLite vector store at {db_path}")
    
    def _init_db(self):
        """データベースとテーブルを初期化"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content TEXT NOT NULL,
                        source TEXT,
                        source_type TEXT,
                        metadata TEXT,
                        vector TEXT,
                        created_at INTEGER
                    )
                """)
                
                # インデックスを作成
                conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON documents(source)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_source_type ON documents(source_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON documents(created_at)")
                
                logger.debug("Database tables and indices initialized")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def add_documents(self, docs: List[Dict[str, Any]], vectors: List[List[float]], file_path: Optional[str] = None):
        """ドキュメントとベクトルを追加"""
        if not docs or not vectors:
            logger.warning("No documents or vectors provided to add_documents")
            return
            
        if len(docs) != len(vectors):
            raise ValueError(f"Number of documents ({len(docs)}) doesn't match number of vectors ({len(vectors)})")
        
        now = int(time.time())
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for i, (doc, vec) in enumerate(zip(docs, vectors)):
                    metadata = doc.get("metadata", {})
                    conn.execute(
                        """
                        INSERT INTO documents 
                        (content, source, source_type, metadata, vector, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            doc["content"],
                            metadata.get("source", file_path),
                            metadata.get("source_type", "unknown"),
                            json.dumps(metadata),
                            json.dumps(vec),
                            now
                        )
                    )
                
            logger.info(f"Added {len(docs)} documents to SQLite store")
            
        except Exception as e:
            logger.error(f"Error adding documents to SQLite: {e}")
            raise
    
    def similarity_search(self, query_vector: List[float], top_k: int = 5, 
                        filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """ベクトル類似度検索を実行"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query_np = np.array(query_vector)
                
                # すべてのベクトルを取得して類似度を計算
                cursor = conn.execute("SELECT id, content, metadata, vector FROM documents")
                results = []
                
                for row in cursor:
                    doc_id, content, metadata_json, vector_json = row
                    vector = np.array(json.loads(vector_json))
                    
                    # コサイン類似度を計算
                    similarity = self._cosine_similarity(query_np, vector)
                    
                    metadata = json.loads(metadata_json)
                    if self._passes_filter(metadata, filter_criteria):
                        results.append({
                            "id": doc_id,
                            "content": content,
                            "metadata": metadata,
                            "score": similarity
                        })
                
                # スコアでソートして上位k件を返す
                results.sort(key=lambda x: x["score"], reverse=True)
                return results[:top_k]
                
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """コサイン類似度を計算"""
        try:
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(vec1, vec2) / (norm1 * norm2)
        except Exception:
            return 0.0
    
    def _passes_filter(self, metadata: Dict[str, Any], 
                      filter_criteria: Optional[Dict[str, Any]]) -> bool:
        """フィルタ条件をチェック"""
        if not filter_criteria:
            return True
            
        if "source_type" in filter_criteria:
            if metadata.get("source_type") != filter_criteria["source_type"]:
                return False
                
        if "source" in filter_criteria:
            if metadata.get("source") != filter_criteria["source"]:
                return False
                
        return True
    
    def file_exists(self, file_path: str) -> bool:
        """ファイルが存在するかチェック"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM documents WHERE source = ?",
                    (file_path,)
                )
                count = cursor.fetchone()[0]
                return count > 0
        except Exception as e:
            logger.error(f"Error checking file existence: {e}")
            return False
    
    def delete_file(self, file_path: str) -> int:
        """ファイルに関連するドキュメントを削除"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM documents WHERE source = ?",
                    (file_path,)
                )
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return 0
    
    def file_needs_update(self, file_path: str) -> bool:
        """ファイルが更新が必要かチェック"""
        return True  # 簡易実装
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM documents")
                total_docs = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(DISTINCT source) FROM documents")
                total_files = cursor.fetchone()[0]
                
                return {
                    "total_documents": total_docs,
                    "total_files": total_files,
                    "vector_dimension": self.vector_dimension
                }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "total_documents": 0,
                "total_files": 0,
                "vector_dimension": self.vector_dimension
            }
    
    def close(self):
        """クリーンアップ処理（SQLiteは明示的なクローズは不要）"""
        pass
