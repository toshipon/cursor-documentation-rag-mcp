import sqlite3
import os
import json
import time
import hashlib
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import config

# ロギング設定
logger = logging.getLogger(__name__)

class VectorStore:
    """
    SQLite-VSSを利用したベクトルストアクラス
    テキストデータとベクトルを保存し、ベクトル類似度検索を行う
    """
    def __init__(self, db_path: str, vector_dimension: int = 512, create_indices: bool = True):
        """
        初期化
        
        Args:
            db_path: データベースファイルのパス
            vector_dimension: ベクトルの次元数
            create_indices: インデックスを自動作成するかどうか
        """
        self.db_path = db_path
        self.vector_dimension = vector_dimension
        
        # DBディレクトリがなければ作成
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # SQLiteに接続
        self.conn = sqlite3.connect(db_path)
        
        # vss拡張モジュールをロード（存在しなければインストール）
        self._load_vss_extension()
        
        # テーブルを初期化
        self._init_db()
        
        # インデックスを作成
        if create_indices:
            self.create_indices()
        
        logger.info(f"VectorStore initialized at {db_path} with dimension {vector_dimension}")
    
    def _load_vss_extension(self):
        """SQLite-VSS拡張をロードする"""
        try:
            # sqlite-vss拡張をロード
            self.conn.enable_load_extension(True)
            self.conn.load_extension("vss0")
            logger.info("SQLite-VSS extension loaded successfully")
        except sqlite3.OperationalError:
            logger.error("Failed to load SQLite-VSS extension. Make sure it's installed.")
            raise RuntimeError(
                "SQLite-VSS extension not found. Please install it with: "
                "pip install sqlite-vss && python -c 'import sqlite_vss; sqlite_vss.load()'"
            )
    
    def _init_db(self):
        """必要なテーブルとインデックスを初期化"""
        # ドキュメントテーブル（テキストとベクトルを格納）
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,               -- テキスト内容
            source TEXT NOT NULL,                -- ソースファイルパス
            source_type TEXT NOT NULL,           -- ソースタイプ（markdown, code, pdfなど）
            metadata TEXT NOT NULL,              -- メタデータ（JSON形式）
            vector BLOB NOT NULL,                -- 埋め込みベクトル
            created_at INTEGER NOT NULL          -- 作成日時（UNIX時間）
        )
        ''')
        
        # ファイルメタデータテーブル（処理済みファイルの情報）
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS file_metadata (
            file_path TEXT PRIMARY KEY,          -- ファイルパス
            last_modified INTEGER NOT NULL,      -- 最終更新日時（UNIX時間）
            file_hash TEXT NOT NULL,             -- ファイルハッシュ
            last_vectorized INTEGER NOT NULL,    -- 最終ベクトル化日時（UNIX時間）
            status TEXT NOT NULL,                -- 処理ステータス
            chunk_count INTEGER NOT NULL DEFAULT 0 -- チャンク数
        )
        ''')
        
        # ベクトル類似度検索用のvssテーブルが存在するか確認
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vss_documents'"
        )
        if not cursor.fetchone():
            # vssテーブルを作成
            self.conn.execute(f'''
            CREATE VIRTUAL TABLE IF NOT EXISTS vss_documents USING vss0(
                vector({self.vector_dimension}),
                tokenize='porter'
            )
            ''')
        
        self.conn.commit()
        logger.info("Database tables initialized")

        # インデックスを作成
        self.create_indices()

    def create_indices(self):
        """
        検索パフォーマンス向上のためのインデックスを作成
        """
        try:
            logger.info("Creating indices on documents table...")
            
            # ソースタイプに対するインデックス（フィルタリング高速化）
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_source_type
                ON documents(source_type)
            """)
            
            # ソースパスに対するインデックス（削除操作高速化）
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_source
                ON documents(source)
            """)
            
            # 作成日時に対するインデックス（時間範囲フィルタリング高速化）
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_created_at
                ON documents(created_at)
            """)
            
            # メタデータに対する関数ベースインデックスの作成（サンプル）
            # よく使用されるメタデータフィールドに対してカスタムインデックスを作成
            # ここでは例としてauthorフィールドを想定
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_metadata_author
                ON documents(json_extract(metadata, '$.author'))
            """)
            
            self.conn.commit()
            logger.info("Database indices created successfully")
            
        except Exception as e:
            logger.error(f"Error creating indices: {e}")
            raise
    
    def add_documents(self, docs: List[Dict[str, Any]], vectors: List[List[float]], file_path: Optional[str] = None):
        """
        ドキュメントとそのベクトル表現をデータベースに追加
        
        Args:
            docs: ドキュメントのリスト（各ドキュメントはcontent, metadataを含む辞書）
            vectors: ドキュメントのベクトル表現のリスト
            file_path: ドキュメントの元ファイルパス（オプション）
        """
        if not docs or not vectors:
            logger.warning("No documents or vectors provided to add_documents")
            return
            
        if len(docs) != len(vectors):
            raise ValueError(f"Number of documents ({len(docs)}) doesn't match number of vectors ({len(vectors)})")
        
        now = int(time.time())
        try:
            # トランザクション開始
            self.conn.execute("BEGIN")
            
            # ドキュメントとベクトルを挿入
            for doc, vec in zip(docs, vectors):
                # メタデータをJSON形式に変換
                metadata_json = json.dumps(doc["metadata"])
                
                # ドキュメントテーブルに挿入
                cursor = self.conn.execute(
                    "INSERT INTO documents (content, source, source_type, metadata, vector, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        doc["content"], 
                        doc["metadata"]["source"], 
                        doc["metadata"]["source_type"], 
                        metadata_json,
                        json.dumps(vec),  # ベクトルをJSON文字列として保存
                        now
                    )
                )
                
                # 挿入されたIDを取得
                doc_id = cursor.lastrowid
                
                # vssテーブルに挿入
                self.conn.execute(
                    "INSERT INTO vss_documents(rowid, vector) VALUES (?, ?)",
                    (doc_id, vec)
                )
            
            # ファイルメタデータも登録
            if file_path and os.path.exists(file_path):
                mtime = int(os.path.getmtime(file_path))
                with open(file_path, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO file_metadata 
                    (file_path, last_modified, file_hash, last_vectorized, status, chunk_count) 
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (file_path, mtime, file_hash, now, "processed", len(docs))
                )
            
            # コミット
            self.conn.commit()
            logger.info(f"Added {len(docs)} documents to vector store")
            
        except Exception as e:
            # エラー発生時はロールバック
            self.conn.rollback()
            logger.error(f"Error adding documents to vector store: {e}")
            raise

    def _build_search_query(self, filter_criteria: Dict[str, Any] = None) -> Tuple[str, List[Any]]:
        """
        検索クエリとパラメータを構築する内部メソッド
        
        Args:
            filter_criteria: フィルタリング条件（オプション）
            
        Returns:
            クエリ文字列とフィルタパラメータのタプル
        """
        # 基本クエリの構築
        query = """
            SELECT 
                d.id, d.content, d.source, d.source_type, d.metadata,
                vss_documents.distance AS score
            FROM 
                vss_documents
            JOIN 
                documents d ON vss_documents.rowid = d.id
            WHERE 
                vss_documents.vector_search(?)
        """
        
        # 共通のフィルタ構築メソッドを使用
        filter_query, filter_params = self._build_filter_clauses(filter_criteria)
        
        # フィルタリング条件を追加
        if filter_query:
            query += f" AND {filter_query}"
        
        return query, filter_params
        
        return query, filter_params

    def similarity_search(self, query_vector: List[float], top_k: int = 5, filter_criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        ベクトル類似度に基づいてドキュメントを検索
        
        Args:
            query_vector: 検索クエリのベクトル表現
            top_k: 返却する結果の最大数
            filter_criteria: フィルタリング条件（オプション）
            
        Returns:
            類似ドキュメントのリスト（スコア付き）
        """
        if not query_vector:
            logger.warning("Empty query vector provided to similarity_search")
            return []
        
        try:
            # 検索クエリを構築
            query, filter_params = self._build_search_query(filter_criteria)
            
            # 結果の制限を追加
            query += " LIMIT ?"
            
            # クエリパラメータを準備
            params = [query_vector] + filter_params + [top_k]
            
            # クエリ実行
            cursor = self.conn.execute(query, params)
            
            # 結果を整形
            results = []
            for row in cursor.fetchall():
                # JSON形式のメタデータをパース
                metadata = json.loads(row[4])
                
                results.append({
                    "id": row[0],
                    "content": row[1],
                    "metadata": metadata,
                    "score": row[5]  # 類似度スコア
                })
                
            return results
        
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise
            
    def batch_similarity_search(self, query_vectors: List[List[float]], top_k: int = 5, filter_criteria: Dict[str, Any] = None) -> List[List[Dict[str, Any]]]:
        """
        複数ベクトルの一括類似度検索
        
        Args:
            query_vectors: 検索クエリのベクトル表現のリスト
            top_k: 各クエリで返却する結果の最大数
            filter_criteria: フィルタリング条件（オプション）
            
        Returns:
            クエリごとの類似ドキュメントのリストのリスト
        """
        if not query_vectors:
            logger.warning("Empty query vectors provided to batch_similarity_search")
            return []
        
        try:
            # 検索クエリを構築
            query, filter_params = self._build_search_query(filter_criteria)
            
            # 結果の制限を追加
            query += " LIMIT ?"
            
            # 各クエリベクトルに対して検索を実行
            all_results = []
            for q_vec in query_vectors:
                # クエリパラメータを準備
                params = [q_vec] + filter_params + [top_k]
                
                # クエリ実行
                cursor = self.conn.execute(query, params)
                
                # 結果を整形
                query_results = []
                for row in cursor.fetchall():
                    # JSON形式のメタデータをパース
                    metadata = json.loads(row[4])
                    
                    query_results.append({
                        "id": row[0],
                        "content": row[1],
                        "metadata": metadata,
                        "score": row[5]  # 類似度スコア
                    })
                
                all_results.append(query_results)
            
            return all_results
            
        except Exception as e:
            logger.error(f"Error during batch similarity search: {e}")
            raise
            
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def file_exists(self, file_path: str) -> bool:
        """
        指定されたファイルが既にベクトル化されているか確認
        
        Args:
            file_path: チェックするファイルパス
            
        Returns:
            ファイルが既にベクトル化されていればTrue、そうでなければFalse
        """
        cursor = self.conn.execute(
            "SELECT file_path FROM file_metadata WHERE file_path = ?", 
            (file_path,)
        )
        return cursor.fetchone() is not None
    
    def file_needs_update(self, file_path: str) -> bool:
        """
        ファイルが変更されていて再ベクトル化が必要かどうかをチェック
        
        Args:
            file_path: チェックするファイルパス
            
        Returns:
            再ベクトル化が必要ならTrue、そうでなければFalse
        """
        if not os.path.exists(file_path):
            return False
            
        # 現在のファイル情報
        current_mtime = int(os.path.getmtime(file_path))
        with open(file_path, "rb") as f:
            current_hash = hashlib.md5(f.read()).hexdigest()
        
        # DBに保存されている情報を取得
        cursor = self.conn.execute(
            "SELECT last_modified, file_hash FROM file_metadata WHERE file_path = ?", 
            (file_path,)
        )
        
        row = cursor.fetchone()
        if row:
            stored_mtime, stored_hash = row
            
            # ハッシュが異なる場合は再ベクトル化が必要
            if stored_hash != current_hash:
                return True
                
            # 更新日時が新しい場合も再ベクトル化が必要
            if current_mtime > stored_mtime:
                return True
                
            return False
        
        # ファイルがDBに存在しない場合は、ベクトル化が必要
        return True
    
    def delete_file(self, file_path: str) -> int:
        """
        指定されたファイルに関連するすべてのドキュメントを削除
        
        Args:
            file_path: 削除するファイルパス
            
        Returns:
            削除されたドキュメント数
        """
        try:
            # トランザクション開始
            self.conn.execute("BEGIN")
            
            # ファイルに関連するドキュメントのIDを取得
            cursor = self.conn.execute(
                "SELECT id FROM documents WHERE source = ?", 
                (file_path,)
            )
            
            doc_ids = [row[0] for row in cursor.fetchall()]
            
            if not doc_ids:
                self.conn.rollback()
                return 0
            
            # ドキュメントとVSSエントリを削除
            for doc_id in doc_ids:
                self.conn.execute(
                    "DELETE FROM vss_documents WHERE rowid = ?", 
                    (doc_id,)
                )
                
            deleted_count = len(doc_ids)
            
            # ドキュメントを削除
            self.conn.execute(
                "DELETE FROM documents WHERE source = ?", 
                (file_path,)
            )
            
            # ファイルメタデータを削除
            self.conn.execute(
                "DELETE FROM file_metadata WHERE file_path = ?", 
                (file_path,)
            )
            
            # コミット
            self.conn.commit()
            
            logger.info(f"Deleted {deleted_count} documents related to {file_path}")
            return deleted_count
            
        except Exception as e:
            # エラー発生時はロールバック
            self.conn.rollback()
            logger.error(f"Error deleting file from vector store: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        ベクトルストアの統計情報を取得
        
        Returns:
            統計情報の辞書
        """
        stats = {}
        
        try:
            # ドキュメント総数
            cursor = self.conn.execute("SELECT COUNT(*) FROM documents")
            stats["total_documents"] = cursor.fetchone()[0]
            
            # ファイル総数
            cursor = self.conn.execute("SELECT COUNT(*) FROM file_metadata")
            stats["total_files"] = cursor.fetchone()[0]
            
            # ソースタイプごとのドキュメント数
            cursor = self.conn.execute(
                "SELECT source_type, COUNT(*) FROM documents GROUP BY source_type"
            )
            stats["documents_by_type"] = {row[0]: row[1] for row in cursor.fetchall()}
            
            # データベースファイルサイズ
            if os.path.exists(self.db_path):
                stats["db_size_bytes"] = os.path.getsize(self.db_path)
                stats["db_size_mb"] = round(stats["db_size_bytes"] / (1024 * 1024), 2)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {"error": str(e)}
    
    def close(self):
        """データベース接続を閉じる"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def hybrid_search(self, 
                query_text: str, 
                query_vector: List[float], 
                top_k: int = 5, 
                filter_criteria: Dict[str, Any] = None,
                vector_weight: float = 0.7,
                text_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        ハイブリッド検索：ベクトル類似度検索とテキスト検索を組み合わせる
        
        Args:
            query_text: テキスト検索クエリ
            query_vector: ベクトル検索クエリ
            top_k: 返却する結果の最大数
            filter_criteria: フィルタリング条件
            vector_weight: ベクトル検索結果のスコアに対する重み（0.0〜1.0）
            text_weight: テキスト検索結果のスコアに対する重み（0.0〜1.0）
            
        Returns:
            ハイブリッド検索結果のリスト（スコア付き）
        """
        if vector_weight + text_weight != 1.0:
            # 重みの正規化
            total = vector_weight + text_weight
            vector_weight = vector_weight / total
            text_weight = text_weight / total
            
        # 基本クエリの構築（ベクトル部分）
        query = """
            SELECT 
                d.id, d.content, d.source, d.source_type, d.metadata,
                vss_documents.distance AS vector_score,
                CASE 
                    WHEN d.content LIKE ? THEN 1.0
                    ELSE (LENGTH(d.content) - LENGTH(REPLACE(LOWER(d.content), LOWER(?), ''))) / (LENGTH(?) * 1.0)
                END AS text_score
            FROM 
                vss_documents
            JOIN 
                documents d ON vss_documents.rowid = d.id
            WHERE 
                vss_documents.vector_search(?)
        """

        # フィルタリング条件を追加
        filter_clauses = []
        filter_params = []
        
        if filter_criteria:
            # 既存のフィルタリングロジックを活用
            _, filter_params = self._build_search_query(filter_criteria)
            
            # ソースタイプでフィルタリング
            if "source_type" in filter_criteria:
                source_types = filter_criteria["source_type"]
                if isinstance(source_types, str):
                    source_types = [source_types]
                placeholders = ", ".join("?" for _ in source_types)
                filter_clauses.append(f"d.source_type IN ({placeholders})")
            
            # ソースパスでフィルタリング
            if "source" in filter_criteria:
                source_path = filter_criteria["source"]
                if isinstance(source_path, list):
                    # 複数パスのOR条件
                    path_clauses = []
                    for _ in source_path:
                        path_clauses.append("d.source LIKE ?")
                    filter_clauses.append(f"({' OR '.join(path_clauses)})")
                else:
                    filter_clauses.append("d.source LIKE ?")
            
            # 作成日時でフィルタリング
            if "created_after" in filter_criteria:
                filter_clauses.append("d.created_at >= ?")
            if "created_before" in filter_criteria:
                filter_clauses.append("d.created_at <= ?")
            
            # メタデータのJSONフィールドでフィルタリング
            if "metadata" in filter_criteria:
                for key, value in filter_criteria["metadata"].items():
                    if isinstance(value, list):
                        # リスト値（IN句）
                        placeholders = ", ".join("?" for _ in value)
                        filter_clauses.append(f"json_extract(d.metadata, '$.{key}') IN ({placeholders})")
                    elif isinstance(value, dict) and "operator" in value:
                        op = value["operator"]
                        if op == "contains":
                            filter_clauses.append(f"json_extract(d.metadata, '$.{key}') LIKE ?")
                        elif op in ["=", "!=", ">", ">=", "<", "<="]:
                            filter_clauses.append(f"json_extract(d.metadata, '$.{key}') {op} ?")
                        elif op == "between" and isinstance(value["value"], list) and len(value["value"]) == 2:
                            filter_clauses.append(f"json_extract(d.metadata, '$.{key}') BETWEEN ? AND ?")
                    else:
                        filter_clauses.append(f"json_extract(d.metadata, '$.{key}') = ?")
        
        # 結果並べ替えとスコア計算の追加
        query += f"""
            ORDER BY ({vector_weight} * vector_score + {text_weight} * text_score) DESC
            LIMIT ?
        """
        
        # クエリパラメータを準備
        # テキスト検索パラメータを2つ追加（LIKE検索とワード出現頻度計算用）
        text_search_params = [f'%{query_text}%', query_text.lower(), query_text]
        params = text_search_params + [query_vector] + filter_params + [top_k]
        
        try:
            # クエリ実行
            cursor = self.conn.execute(query, params)
            
            # 結果を整形
            results = []
            for row in cursor.fetchall():
                # JSON形式のメタデータをパース
                metadata = json.loads(row[4])
                
                # 最終スコア計算
                vector_score = float(row[5])
                text_score = float(row[6])
                combined_score = vector_weight * vector_score + text_weight * text_score
                
                results.append({
                    "id": row[0],
                    "content": row[1],
                    "metadata": metadata,
                    "score": combined_score,
                    "vector_score": vector_score,
                    "text_score": text_score
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error during hybrid search: {e}")
            raise

    def optimize_database(self):
        """
        データベースのパフォーマンス最適化を実行
        - VACUUM: 未使用スペースの回収
        - ANALYZE: クエリオプティマイザのための統計収集
        """
        try:
            logger.info("Optimizing database...")
            
            # プラグマ設定
            self.conn.execute("PRAGMA optimize")
            
            # ANALYZE実行（クエリプランナー向け統計情報の更新）
            self.conn.execute("ANALYZE")
            
            # VACUUM実行（未使用スペース回収とDB最適化）
            self.conn.execute("VACUUM")
            
            logger.info("Database optimization completed")
            
        except Exception as e:
            logger.error(f"Error optimizing database: {e}")
            raise
    
    def keyword_search(self, 
                  keyword: str, 
                  top_k: int = 5, 
                  filter_criteria: Dict[str, Any] = None,
                  match_type: str = "contains") -> List[Dict[str, Any]]:
        """
        キーワードベースのテキスト検索（ベクトル検索を使用しない）
        
        Args:
            keyword: 検索キーワード
            top_k: 返却する結果の最大数
            filter_criteria: フィルタリング条件
            match_type: 一致タイプ("contains", "exact", "starts_with", "ends_with")
            
        Returns:
            検索結果のリスト
        """
        if not keyword:
            logger.warning("Empty keyword provided to keyword_search")
            return []
        
        # 基本クエリの構築
        query = """
            SELECT 
                d.id, d.content, d.source, d.source_type, d.metadata
            FROM 
                documents d
            WHERE 
        """
        
        # 検索タイプに基づいてWHERE句を構築
        if match_type == "exact":
            query += "d.content = ?"
            keyword_param = keyword
        elif match_type == "starts_with":
            query += "d.content LIKE ?"
            keyword_param = f"{keyword}%"
        elif match_type == "ends_with":
            query += "d.content LIKE ?"
            keyword_param = f"%{keyword}"
        else:  # contains (default)
            query += "d.content LIKE ?"
            keyword_param = f"%{keyword}%"
        
        params = [keyword_param]
        
        # フィルタリング条件があれば追加
        if filter_criteria:
            filter_query, filter_params = self._build_filter_clauses(filter_criteria)
            if filter_query:
                query += f" AND {filter_query}"
                params.extend(filter_params)
        
        # 結果の制限を追加
        query += " LIMIT ?"
        params.append(top_k)
        
        try:
            # クエリ実行
            cursor = self.conn.execute(query, params)
            
            # 結果を整形
            results = []
            for row in cursor.fetchall():
                # JSON形式のメタデータをパース
                metadata = json.loads(row[4])
                
                results.append({
                    "id": row[0],
                    "content": row[1],
                    "metadata": metadata,
                    "score": 1.0  # テキスト検索なのでスコアは意味を持たない
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error during keyword search: {e}")
            raise
    
    def _build_filter_clauses(self, filter_criteria: Dict[str, Any]) -> Tuple[str, List[Any]]:
        """
        フィルタリング条件からSQLのWHERE句を構築する
        （_build_search_queryから分離してロジックを再利用可能に）
        
        Args:
            filter_criteria: フィルタリング条件
            
        Returns:
            WHERE句の文字列とパラメータリストのタプル
        """
        filter_clauses = []
        filter_params = []
        
        if not filter_criteria:
            return "", []
        
        # ソースタイプでフィルタリング
        if "source_type" in filter_criteria:
            source_types = filter_criteria["source_type"]
            if isinstance(source_types, str):
                source_types = [source_types]
            placeholders = ", ".join("?" for _ in source_types)
            filter_clauses.append(f"source_type IN ({placeholders})")
            filter_params.extend(source_types)
        
        # ソースパスでフィルタリング
        if "source" in filter_criteria:
            source_path = filter_criteria["source"]
            if isinstance(source_path, list):
                # 複数パスのOR条件
                path_clauses = []
                for path in source_path:
                    path_clauses.append("source LIKE ?")
                    filter_params.append(f"%{path}%")
                filter_clauses.append(f"({' OR '.join(path_clauses)})")
            else:
                filter_clauses.append("source LIKE ?")
                filter_params.append(f"%{source_path}%")
        
        # 作成日時でフィルタリング
        if "created_after" in filter_criteria:
            filter_clauses.append("created_at >= ?")
            filter_params.append(filter_criteria["created_after"])
            
        if "created_before" in filter_criteria:
            filter_clauses.append("created_at <= ?")
            filter_params.append(filter_criteria["created_before"])
        
        # メタデータのJSONフィールドでフィルタリング
        if "metadata" in filter_criteria:
            for key, value in filter_criteria["metadata"].items():
                if isinstance(value, list):
                    # リスト値（IN句）
                    placeholders = ", ".join("?" for _ in value)
                    filter_clauses.append(f"json_extract(metadata, '$.{key}') IN ({placeholders})")
                    filter_params.extend(value)
                elif isinstance(value, dict) and "operator" in value:
                    # 比較演算子を使用する高度なフィルタリング
                    op = value["operator"]
                    val = value["value"]
                    
                    if op == "contains":
                        filter_clauses.append(f"json_extract(metadata, '$.{key}') LIKE ?")
                        filter_params.append(f"%{val}%")
                    elif op in ["=", "!=", ">", ">=", "<", "<="]:
                        filter_clauses.append(f"json_extract(metadata, '$.{key}') {op} ?")
                        filter_params.append(val)
                    elif op == "between" and isinstance(val, list) and len(val) == 2:
                        filter_clauses.append(f"json_extract(metadata, '$.{key}') BETWEEN ? AND ?")
                        filter_params.extend(val)
                else:
                    # 通常の等価比較
                    filter_clauses.append(f"json_extract(metadata, '$.{key}') = ?")
                    filter_params.append(value)
        
        return " AND ".join(filter_clauses), filter_params