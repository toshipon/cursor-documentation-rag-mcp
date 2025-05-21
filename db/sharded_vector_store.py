import os
import sys
import json
import time
import hashlib
import logging
import sqlite3
import threading
from typing import List, Dict, Any, Optional, Tuple

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from db.vector_store import VectorStore

# ロギング設定
logger = logging.getLogger(__name__)

class ShardedVectorStore:
    """
    複数のシャードに分散されたベクトルストア
    大規模なデータセットを複数のデータベースに分散して管理します
    """
    
    def __init__(self, base_dir: str, num_shards: int = 4, vector_dimension: int = 512):
        """
        初期化
        
        Args:
            base_dir: シャードDBの保存先ディレクトリ
            num_shards: シャード数
            vector_dimension: ベクトルの次元数
        """
        self.base_dir = base_dir
        self.num_shards = num_shards
        self.vector_dimension = vector_dimension
        
        # シャードディレクトリの作成
        os.makedirs(base_dir, exist_ok=True)
        
        # シャードの初期化
        self.shards = []
        for i in range(num_shards):
            shard_path = os.path.join(base_dir, f"shard_{i}.db")
            self.shards.append(VectorStore(shard_path, vector_dimension))
        
        logger.info(f"Initialized sharded vector store with {num_shards} shards in {base_dir}")
        
        # シャードアクセス用ロック
        self.locks = [threading.RLock() for _ in range(num_shards)]
        
        # パフォーマンス指標の初期化
        self.shard_access_count = [0] * num_shards
        self.shard_query_time = [0.0] * num_shards
    
    def _get_shard_index(self, file_path: str) -> int:
        """
        ファイルパスからシャードインデックスを決定
        
        Args:
            file_path: ファイルパス
            
        Returns:
            シャードインデックス（0 〜 num_shards-1）
        """
        # パスのハッシュからシャードインデックスを計算
        hash_val = int(hashlib.md5(file_path.encode()).hexdigest(), 16)
        return hash_val % self.num_shards
    
    def _get_shard_for_file(self, file_path: str) -> Tuple[VectorStore, threading.RLock]:
        """
        ファイルに対応するシャードとロックを取得
        
        Args:
            file_path: ファイルパス
            
        Returns:
            (シャード, シャードロック)
        """
        shard_idx = self._get_shard_index(file_path)
        return self.shards[shard_idx], self.locks[shard_idx]
    
    def add_documents(self, docs: List[Dict[str, Any]], vectors: List[List[float]], file_path: Optional[str] = None):
        """
        ドキュメントとそのベクトル表現をシャードに追加
        
        Args:
            docs: ドキュメントのリスト（各ドキュメントはcontent, metadataを含む辞書）
            vectors: ドキュメントのベクトル表現のリスト
            file_path: ドキュメントの元ファイルパス（オプション）
        """
        if not file_path:
            # file_pathがない場合は最初のドキュメントのsourceを使用
            if docs and "metadata" in docs[0] and "source" in docs[0]["metadata"]:
                file_path = docs[0]["metadata"]["source"]
            else:
                # ランダムにシャードを選択
                shard_idx = int(time.time()) % self.num_shards
                with self.locks[shard_idx]:
                    self.shards[shard_idx].add_documents(docs, vectors, file_path)
                return
        
        # ファイルパスに基づいてシャードを選択
        shard, lock = self._get_shard_for_file(file_path)
        
        # ロックを取得して操作
        with lock:
            shard.add_documents(docs, vectors, file_path)
    
    def similarity_search(self, query_vector: List[float], top_k: int = 5, 
                          filter_criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        全シャードに対してベクトル類似度検索を実行し、結果をマージ
        
        Args:
            query_vector: 検索クエリのベクトル表現
            top_k: 返却する結果の最大数
            filter_criteria: フィルタリング条件（オプション）
            
        Returns:
            類似ドキュメントのリスト（スコア付き）
        """
        all_results = []
        start_time = time.time()
        
        # 各シャードを並列処理
        threads = []
        results = [[] for _ in range(self.num_shards)]
        
        def search_shard(shard_idx):
            shard = self.shards[shard_idx]
            shard_start = time.time()
            try:
                with self.locks[shard_idx]:
                    results[shard_idx] = shard.similarity_search(
                        query_vector=query_vector,
                        # 各シャードからtop_k件ずつ取得し後でマージするため、top_kを増やす
                        top_k=top_k * 2,  
                        filter_criteria=filter_criteria
                    )
                    # シャードアクセス統計を更新
                    self.shard_access_count[shard_idx] += 1
                    self.shard_query_time[shard_idx] += (time.time() - shard_start)
            except Exception as e:
                logger.error(f"Error searching shard {shard_idx}: {e}")
                results[shard_idx] = []
        
        # 各シャードのスレッドを作成・起動
        for i in range(self.num_shards):
            thread = threading.Thread(target=search_shard, args=(i,))
            threads.append(thread)
            thread.start()
        
        # すべてのスレッドが完了するのを待機
        for thread in threads:
            thread.join()
        
        # 全シャードの結果をマージ
        for shard_results in results:
            all_results.extend(shard_results)
        
        # スコアでソート
        all_results.sort(key=lambda x: x["score"] if "score" in x else 0, reverse=True)
        
        logger.info(f"Vector search complete across {self.num_shards} shards in {time.time() - start_time:.2f}s")
        
        # top_k件を返却
        return all_results[:top_k]
    
    def batch_similarity_search(self, query_vectors: List[List[float]], top_k: int = 5,
                                filter_criteria: Dict[str, Any] = None) -> List[List[Dict[str, Any]]]:
        """
        バッチ検索を全シャードに対して実行
        
        Args:
            query_vectors: 検索クエリのベクトル表現のリスト
            top_k: 各クエリで返却する結果の最大数
            filter_criteria: フィルタリング条件（オプション）
            
        Returns:
            クエリごとの類似ドキュメントのリストのリスト
        """
        results = []
        
        # クエリごとに検索を実行
        for query_vector in query_vectors:
            query_results = self.similarity_search(
                query_vector=query_vector,
                top_k=top_k,
                filter_criteria=filter_criteria
            )
            results.append(query_results)
        
        return results
        
    def file_exists(self, file_path: str) -> bool:
        """
        指定されたファイルが既にベクトル化されているか確認
        
        Args:
            file_path: チェックするファイルパス
            
        Returns:
            ファイルが既にベクトル化されていればTrue、そうでなければFalse
        """
        shard, lock = self._get_shard_for_file(file_path)
        with lock:
            return shard.file_exists(file_path)
    
    def file_needs_update(self, file_path: str) -> bool:
        """
        ファイルが変更されていて再ベクトル化が必要かどうかをチェック
        
        Args:
            file_path: チェックするファイルパス
            
        Returns:
            再ベクトル化が必要ならTrue、そうでなければFalse
        """
        shard, lock = self._get_shard_for_file(file_path)
        with lock:
            return shard.file_needs_update(file_path)
    
    def delete_file(self, file_path: str) -> int:
        """
        指定されたファイルに関連するすべてのドキュメントを削除
        
        Args:
            file_path: 削除するファイルパス
            
        Returns:
            削除されたドキュメント数
        """
        shard, lock = self._get_shard_for_file(file_path)
        with lock:
            return shard.delete_file(file_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        全シャードの統計情報を統合して取得
        
        Returns:
            統計情報の辞書
        """
        stats = {
            "shard_count": self.num_shards,
            "total_documents": 0,
            "total_files": 0,
            "documents_by_type": {},
            "db_size_bytes": 0,
            "db_size_mb": 0,
            "shards": []
        }
        
        # 各シャードの統計を集計
        for i, shard in enumerate(self.shards):
            with self.locks[i]:
                shard_stats = shard.get_stats()
                stats["shards"].append({
                    "shard_id": i,
                    "documents": shard_stats.get("total_documents", 0),
                    "files": shard_stats.get("total_files", 0),
                    "size_mb": shard_stats.get("db_size_mb", 0),
                    "access_count": self.shard_access_count[i],
                    "avg_query_time_ms": round((self.shard_query_time[i] * 1000) / max(1, self.shard_access_count[i]), 2)
                })
                
                # 合計値を更新
                stats["total_documents"] += shard_stats.get("total_documents", 0)
                stats["total_files"] += shard_stats.get("total_files", 0)
                stats["db_size_bytes"] += shard_stats.get("db_size_bytes", 0)
                
                # ドキュメントタイプごとの統計をマージ
                for doc_type, count in shard_stats.get("documents_by_type", {}).items():
                    if doc_type in stats["documents_by_type"]:
                        stats["documents_by_type"][doc_type] += count
                    else:
                        stats["documents_by_type"][doc_type] = count
        
        # MB単位のサイズを計算
        stats["db_size_mb"] = round(stats["db_size_bytes"] / (1024 * 1024), 2)
        
        # パフォーマンス指標を追加
        stats["performance"] = {
            "total_queries": sum(self.shard_access_count),
            "total_query_time_ms": round(sum(self.shard_query_time) * 1000, 2),
            "avg_query_time_ms": round((sum(self.shard_query_time) * 1000) / max(1, sum(self.shard_access_count)), 2),
            "load_distribution": self.shard_access_count
        }
        
        return stats
    
    def close(self):
        """全シャードの接続を閉じる"""
        for i, shard in enumerate(self.shards):
            with self.locks[i]:
                shard.close()
        logger.info("All shard connections closed")
    
    def hybrid_search(self, query_text: str, query_vector: List[float], top_k: int = 5, 
                      filter_criteria: Dict[str, Any] = None, vector_weight: float = 0.7, text_weight: float = 0.3) -> List[Dict[str, Any]]:
        """
        全シャードに対してハイブリッド検索を実行し、結果をマージ
        
        Args:
            query_text: 検索クエリのテキスト
            query_vector: 検索クエリのベクトル表現
            top_k: 返却する結果の最大数
            filter_criteria: フィルタリング条件（オプション）
            vector_weight: ベクトル検索の重み（0.0〜1.0）
            text_weight: テキスト検索の重み（0.0〜1.0）
            
        Returns:
            類似ドキュメントのリスト（スコア付き）
        """
        all_results = []
        start_time = time.time()
        
        # 各シャードを並列処理
        threads = []
        results = [[] for _ in range(self.num_shards)]
        
        def search_shard(shard_idx):
            shard = self.shards[shard_idx]
            shard_start = time.time()
            try:
                with self.locks[shard_idx]:
                    results[shard_idx] = shard.hybrid_search(
                        query_text=query_text,
                        query_vector=query_vector,
                        top_k=top_k * 2,  # 各シャードからtop_k*2件ずつ取得し後でマージ
                        filter_criteria=filter_criteria,
                        vector_weight=vector_weight,
                        text_weight=text_weight
                    )
                    # シャードアクセス統計を更新
                    self.shard_access_count[shard_idx] += 1
                    self.shard_query_time[shard_idx] += (time.time() - shard_start)
            except Exception as e:
                logger.error(f"Error in hybrid search on shard {shard_idx}: {e}")
                results[shard_idx] = []
        
        # 各シャードのスレッドを作成・起動
        for i in range(self.num_shards):
            thread = threading.Thread(target=search_shard, args=(i,))
            threads.append(thread)
            thread.start()
        
        # すべてのスレッドが完了するのを待機
        for thread in threads:
            thread.join()
        
        # 全シャードの結果をマージ
        for shard_results in results:
            all_results.extend(shard_results)
        
        # スコアでソート
        all_results.sort(key=lambda x: x["score"] if "score" in x else 0, reverse=True)
        
        logger.info(f"Hybrid search complete across {self.num_shards} shards in {time.time() - start_time:.2f}s")
        
        # top_k件を返却
        return all_results[:top_k]
    
    def keyword_search(self, keyword: str, top_k: int = 5, filter_criteria: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        全シャードに対してキーワード検索を実行し、結果をマージ
        
        Args:
            keyword: 検索キーワード
            top_k: 返却する結果の最大数
            filter_criteria: フィルタリング条件（オプション）
            
        Returns:
            キーワードにマッチするドキュメントのリスト（スコア付き）
        """
        all_results = []
        start_time = time.time()
        
        # 各シャードを並列処理
        threads = []
        results = [[] for _ in range(self.num_shards)]
        
        def search_shard(shard_idx):
            shard = self.shards[shard_idx]
            shard_start = time.time()
            try:
                with self.locks[shard_idx]:
                    results[shard_idx] = shard.keyword_search(
                        keyword=keyword,
                        top_k=top_k * 2,  # 各シャードからtop_k*2件ずつ取得し後でマージ
                        filter_criteria=filter_criteria
                    )
                    # シャードアクセス統計を更新
                    self.shard_access_count[shard_idx] += 1
                    self.shard_query_time[shard_idx] += (time.time() - shard_start)
            except Exception as e:
                logger.error(f"Error in keyword search on shard {shard_idx}: {e}")
                results[shard_idx] = []
        
        # 各シャードのスレッドを作成・起動
        for i in range(self.num_shards):
            thread = threading.Thread(target=search_shard, args=(i,))
            threads.append(thread)
            thread.start()
        
        # すべてのスレッドが完了するのを待機
        for thread in threads:
            thread.join()
        
        # 全シャードの結果をマージ
        for shard_results in results:
            all_results.extend(shard_results)
        
        # スコアでソート
        all_results.sort(key=lambda x: x["score"] if "score" in x else 0, reverse=True)
        
        logger.info(f"Keyword search complete across {self.num_shards} shards in {time.time() - start_time:.2f}s")
        
        # top_k件を返却
        return all_results[:top_k]
    
    def create_indices(self):
        """全シャードのデータベースにインデックスを作成"""
        for i, shard in enumerate(self.shards):
            logger.info(f"Creating indices for shard {i}")
            with self.locks[i]:
                shard.create_indices()
    
    def optimize_database(self):
        """全シャードのデータベースを最適化（VACUUM, ANALYZE）"""
        for i, shard in enumerate(self.shards):
            logger.info(f"Optimizing shard {i}")
            with self.locks[i]:
                shard.optimize_database()
    
    def get_shard_stats(self) -> List[Dict[str, Any]]:
        """各シャードの詳細な統計情報を取得"""
        stats = []
        for i, shard in enumerate(self.shards):
            with self.locks[i]:
                shard_stats = shard.get_stats()
                # アクセス統計を追加
                shard_stats.update({
                    "shard_id": i,
                    "access_count": self.shard_access_count[i],
                    "avg_query_time_ms": round((self.shard_query_time[i] * 1000) / max(1, self.shard_access_count[i]), 2),
                    "total_query_time_ms": round(self.shard_query_time[i] * 1000, 2)
                })
                stats.append(shard_stats)
        return stats
    
    def rebalance_shards(self, threshold: float = 0.3) -> Dict[str, Any]:
        """
        シャード間のバランスを調整（実験的機能）
        ドキュメント数の偏りが大きい場合に再分配を行う
        
        Args:
            threshold: 再分配を行う不均衡閾値（0.0〜1.0）
            
        Returns:
            再分配の結果情報
        """
        # 各シャードのドキュメント数を取得
        doc_counts = []
        for i, shard in enumerate(self.shards):
            with self.locks[i]:
                stats = shard.get_stats()
                doc_counts.append(stats.get("total_documents", 0))
        
        total_docs = sum(doc_counts)
        avg_docs = total_docs / self.num_shards if self.num_shards > 0 else 0
        
        # 最大と最小の差が閾値を超えているか確認
        if total_docs == 0 or avg_docs == 0:
            return {"status": "skipped", "reason": "No documents to rebalance"}
            
        max_docs = max(doc_counts)
        min_docs = min(doc_counts)
        imbalance = (max_docs - min_docs) / avg_docs if avg_docs > 0 else 0
        
        if imbalance <= threshold:
            return {
                "status": "skipped", 
                "reason": f"Imbalance {imbalance:.2f} below threshold {threshold}",
                "stats": {
                    "doc_counts": doc_counts,
                    "imbalance": imbalance
                }
            }
        
        logger.info(f"Shard imbalance detected: {imbalance:.2f}, rebalancing shards...")
        
        # ここに再分配ロジックを実装（将来の拡張）
        # TODO: 実装
        
        return {
            "status": "pending",
            "message": "Rebalance functionality is experimental and not fully implemented yet",
            "stats": {
                "doc_counts": doc_counts,
                "imbalance": imbalance
            }
        }
