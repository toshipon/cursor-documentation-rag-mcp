import os
import json
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams, PointStruct

logger = logging.getLogger(__name__)

class QdrantVectorStore:
    """
    Qdrantを利用したベクトルストアクラス
    テキストデータとベクトルを保存し、ベクトル類似度検索を行う
    """
    def __init__(self, url=None, collection_name="documents", vector_dimension=512):
        """
        初期化
        
        Args:
            url: Qdrantサーバーの接続URL
            collection_name: コレクション名
            vector_dimension: ベクトルの次元数
        """
        self.url = url or os.getenv("QDRANT_URL", "http://qdrant:6334")
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension
        self.client = QdrantClient(url=self.url)
        self._ensure_collection()
        logger.info(f"QdrantVectorStore initialized with URL {self.url}, collection {collection_name}")
    
    def _ensure_collection(self):
        """コレクションが存在しない場合は作成する"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_dimension,
                        distance=Distance.COSINE
                    )
                )
                
                # 基本的なペイロードインデックスを作成
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="source_type",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="source",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="created_at",
                    field_schema=models.PayloadSchemaType.INTEGER
                )
        except Exception as e:
            logger.error(f"Error ensuring collection: {e}")
            raise

    def add_documents(self, docs: List[Dict[str, Any]], vectors: List[List[float]], file_path: Optional[str] = None):
        """
        ドキュメントとそのベクトル表現をQdrantに追加
        
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
        points = []
        
        for i, (doc, vec) in enumerate(zip(docs, vectors)):
            # ユニークIDを生成
            doc_id = hashlib.md5(f"{doc['content'][:100]}_{now}_{i}".encode()).hexdigest()
            doc_id = int.from_bytes(doc_id.encode(), byteorder='big') % (2**63 - 1)  # 64bit整数に変換
            
            # ペイロードを準備
            payload = {
                "content": doc["content"],
                "source": doc["metadata"]["source"],
                "source_type": doc["metadata"]["source_type"],
                "metadata": doc["metadata"],
                "created_at": now
            }
            
            points.append(PointStruct(
                id=doc_id,
                vector=vec,
                payload=payload
            ))
        
        try:
            # バッチでポイントを追加
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True  # データの一貫性を保証
            )
            logger.info(f"Added {len(docs)} documents to Qdrant collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error adding documents to Qdrant: {e}")
            raise
    
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
        
        # フィルタを構築
        search_filter = None
        if filter_criteria:
            filter_conditions = []
            
            # ソースタイプでフィルタリング
            if "source_type" in filter_criteria:
                source_types = filter_criteria["source_type"]
                if isinstance(source_types, str):
                    source_types = [source_types]
                filter_conditions.append(
                    models.FieldCondition(
                        key="source_type",
                        match=models.MatchAny(any=source_types)
                    )
                )
            
            # ソースパスでフィルタリング
            if "source" in filter_criteria:
                source_path = filter_criteria["source"]
                if isinstance(source_path, list):
                    filter_conditions.append(
                        models.FieldCondition(
                            key="source",
                            match=models.MatchAny(any=source_path)
                        )
                    )
                else:
                    filter_conditions.append(
                        models.FieldCondition(
                            key="source",
                            match=models.MatchText(text=source_path)
                        )
                    )
            
            # 作成日時でフィルタリング
            if "created_after" in filter_criteria:
                filter_conditions.append(
                    models.FieldCondition(
                        key="created_at",
                        range=models.Range(
                            gt=filter_criteria["created_after"]
                        )
                    )
                )
            
            if "created_before" in filter_criteria:
                filter_conditions.append(
                    models.FieldCondition(
                        key="created_at",
                        range=models.Range(
                            lt=filter_criteria["created_before"]
                        )
                    )
                )
            
            # メタデータのフィールドでフィルタリング
            if "metadata" in filter_criteria:
                for key, value in filter_criteria["metadata"].items():
                    meta_key = f"metadata.{key}"
                    if isinstance(value, list):
                        filter_conditions.append(
                            models.FieldCondition(
                                key=meta_key,
                                match=models.MatchAny(any=value)
                            )
                        )
                    else:
                        filter_conditions.append(
                            models.FieldCondition(
                                key=meta_key,
                                match=models.MatchValue(value=value)
                            )
                        )
            
            # フィルタ条件を組み合わせる
            if filter_conditions:
                search_filter = models.Filter(
                    must=filter_conditions
                )
        
        try:
            # 検索を実行
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                filter=search_filter
            )
            
            # 結果を整形
            results = []
            for res in search_result:
                results.append({
                    "id": res.id,
                    "content": res.payload.get("content"),
                    "metadata": res.payload.get("metadata"),
                    "score": res.score
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search in Qdrant: {e}")
            return []
    
    def delete_file(self, file_path: str) -> int:
        """
        指定されたファイルに関連するすべてのドキュメントを削除
        
        Args:
            file_path: 削除するファイルパス
            
        Returns:
            削除されたドキュメント数
        """
        try:
            # ファイルに一致するドキュメントを検索して数をカウント
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=file_path)
                        )
                    ]
                ),
                limit=1,  # まず1件だけ取得して存在確認
                with_payload=False,
                with_vectors=False
            )
            
            points = search_result[0]
            if not points:
                logger.info(f"No documents found for file path: {file_path}")
                return 0
            
            # ドキュメントを削除
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=file_path)
                        )
                    ]
                ),
                wait=True
            )
            
            # 削除されたドキュメント数を取得（概算）
            count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=file_path)
                        )
                    ]
                )
            )
            
            deleted_count = len(points)
            logger.info(f"Deleted documents related to {file_path} from Qdrant")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting file from Qdrant vector store: {e}")
            return 0
    
    def file_exists(self, file_path: str) -> bool:
        """
        指定されたファイルがベクトルストアに存在するかチェック
        
        Args:
            file_path: チェックするファイルパス
            
        Returns:
            ファイルが存在するかどうか
        """
        try:
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=file_path)
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False
            )
            
            points = search_result[0]
            return len(points) > 0
            
        except Exception as e:
            logger.error(f"Error checking if file exists in Qdrant: {e}")
            return False
    
    def file_needs_update(self, file_path: str) -> bool:
        """
        ファイルが更新されているかチェック（常にTrueを返す簡易実装）
        
        Args:
            file_path: チェックするファイルパス
            
        Returns:
            ファイルが更新されているかどうか
        """
        # 簡易実装：常に更新が必要とする
        # より高度な実装では、ファイルの最終更新時刻とベクトルストア内の作成時刻を比較する
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        ベクトルストアの統計情報を取得
        
        Returns:
            統計情報の辞書
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            # ユニークなソースファイル数を取得
            unique_sources_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # 大きな値を設定
                with_payload=["source"],
                with_vectors=False
            )
            
            unique_sources = set()
            for point in unique_sources_result[0]:
                if point.payload and "source" in point.payload:
                    unique_sources.add(point.payload["source"])
            
            return {
                "total_documents": collection_info.points_count,
                "total_files": len(unique_sources),
                "vector_dimension": collection_info.config.params.vectors.size,
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Error getting stats from Qdrant: {e}")
            return {
                "total_documents": 0,
                "total_files": 0,
                "vector_dimension": self.vector_dimension,
                "collection_name": self.collection_name
            }

    def batch_similarity_search(self, query_vectors: List[List[float]], top_k: int = 5, filter_criteria: Dict[str, Any] = None) -> List[List[Dict[str, Any]]]:
        """
        バッチベクトル類似度検索
        
        Args:
            query_vectors: 検索クエリのベクトル表現のリスト
            top_k: 返却する結果の最大数
            filter_criteria: フィルタリング条件（オプション）
            
        Returns:
            各クエリに対する類似ドキュメントのリスト
        """
        results = []
        for query_vector in query_vectors:
            result = self.similarity_search(query_vector, top_k, filter_criteria)
            results.append(result)
        return results
    
    def hybrid_search(self, query_text: str, query_vector: List[float], top_k: int = 5, 
                     filter_criteria: Dict[str, Any] = None, vector_weight: float = 0.7, 
                     text_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        ハイブリッド検索（ベクトル検索のみ実装、テキスト検索は未実装）
        
        Args:
            query_text: 検索クエリテキスト
            query_vector: 検索クエリのベクトル表現
            top_k: 返却する結果の最大数
            filter_criteria: フィルタリング条件
            vector_weight: ベクトル検索の重み
            text_weight: テキスト検索の重み
            
        Returns:
            類似ドキュメントのリスト
        """
        # 簡易実装：ベクトル検索のみ
        return self.similarity_search(query_vector, top_k, filter_criteria)
    
    def keyword_search(self, keyword: str, top_k: int = 5, filter_criteria: Dict[str, Any] = None, 
                      match_type: str = "contains") -> List[Dict[str, Any]]:
        """
        キーワード検索
        
        Args:
            keyword: 検索キーワード
            top_k: 返却する結果の最大数
            filter_criteria: フィルタリング条件
            match_type: 一致タイプ
            
        Returns:
            マッチしたドキュメントのリスト
        """
        try:
            # キーワードでフィルタリング
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="content",
                        match=models.MatchText(text=keyword)
                    )
                ]
            )
            
            # 追加のフィルタ条件があれば組み合わせる
            if filter_criteria:
                # 簡易実装：既存のフィルタ条件は無視
                pass
            
            # スクロール検索を実行
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=search_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )
            
            # 結果を整形
            results = []
            for point in search_result[0]:
                results.append({
                    "id": point.id,
                    "content": point.payload.get("content"),
                    "metadata": point.payload.get("metadata"),
                    "score": 1.0  # キーワード検索では固定スコア
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error during keyword search in Qdrant: {e}")
            return []
    
    def optimize_database(self):
        """
        データベース最適化（Qdrantでは特に何もしない）
        """
        logger.info("Database optimization requested for Qdrant (no action needed)")
        pass

    def close(self):
        """接続を閉じる"""
        # Qdrantクライアントには明示的にクローズするメソッドはないが、将来的な拡張性のために残しておく
        logger.info("QdrantVectorStore connection closed")