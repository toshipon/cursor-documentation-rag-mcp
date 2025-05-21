import os
import sys
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from db.vector_store import VectorStore

# ロギング設定
logger = logging.getLogger(__name__)

class IncrementalUpdater:
    """ベクトルストアの増分更新を管理するクラス"""
    
    def __init__(self, vector_store: VectorStore):
        """
        初期化
        
        Args:
            vector_store: ベクトルストアのインスタンス
        """
        self.vector_store = vector_store
        self.update_stats = {
            "added_files": 0,
            "updated_files": 0, 
            "deleted_files": 0,
            "added_documents": 0,
            "deleted_documents": 0,
            "unchanged_files": 0,
            "errors": 0
        }
    
    def scan_directory(self, directory: str, extensions: List[str] = None) -> List[str]:
        """
        ディレクトリを再帰的にスキャンしてファイルリストを取得
        
        Args:
            directory: スキャンするディレクトリパス
            extensions: 対象とするファイル拡張子のリスト（例: ['.md', '.py']）
            
        Returns:
            検出されたファイルパスのリスト
        """
        if not os.path.exists(directory) or not os.path.isdir(directory):
            logger.warning(f"Directory not found: {directory}")
            return []
        
        # 拡張子が指定されていない場合は全ファイルを対象とする
        if extensions is None:
            extensions = []
        
        # 小文字に変換（大文字小文字を区別しない）
        extensions = [ext.lower() for ext in extensions]
        
        found_files = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                # 隠しファイルやバックアップファイルをスキップ
                if file.startswith('.') or file.startswith('~') or file.endswith('~'):
                    continue
                
                file_path = os.path.join(root, file)
                
                # 拡張子フィルタリング
                if extensions:
                    _, ext = os.path.splitext(file_path)
                    if ext.lower() not in extensions:
                        continue
                
                found_files.append(file_path)
        
        logger.info(f"Found {len(found_files)} files in {directory}")
        return found_files
    
    def get_tracked_files(self) -> Dict[str, Dict[str, Any]]:
        """
        ベクトルストアでトラッキングされているファイルの一覧を取得
        
        Returns:
            ファイルパスをキー、メタデータを値とする辞書
        """
        try:
            cursor = self.vector_store.conn.execute(
                """
                SELECT 
                    file_path, last_modified, file_hash, 
                    last_vectorized, status, chunk_count
                FROM 
                    file_metadata
                """
            )
            
            tracked_files = {}
            for row in cursor.fetchall():
                file_path, last_modified, file_hash, last_vectorized, status, chunk_count = row
                tracked_files[file_path] = {
                    "last_modified": last_modified,
                    "file_hash": file_hash,
                    "last_vectorized": last_vectorized,
                    "status": status,
                    "chunk_count": chunk_count
                }
            
            return tracked_files
            
        except Exception as e:
            logger.error(f"Error getting tracked files: {e}")
            return {}
    
    def incremental_update(self, directory: str, extensions: List[str] = None, 
                           vectorize_func = None, delete_missing: bool = True) -> Dict[str, Any]:
        """
        ディレクトリ内のファイルを増分的に更新
        
        Args:
            directory: 更新するディレクトリパス
            extensions: 対象とするファイル拡張子のリスト
            vectorize_func: ファイルをベクトル化する関数（引数としてファイルパスを受け取る）
            delete_missing: 存在しないファイルを削除するかどうか
            
        Returns:
            更新統計情報
        """
        if not vectorize_func:
            logger.error("Vectorize function is required for incremental update")
            return self.update_stats
        
        # 現在のファイルリストを取得
        current_files = set(self.scan_directory(directory, extensions))
        
        # トラッキング済みファイルを取得
        tracked_files = self.get_tracked_files()
        tracked_file_paths = set(tracked_files.keys())
        
        # 新規ファイル（追加が必要）
        new_files = current_files - tracked_file_paths
        
        # 削除されたファイル（存在していないファイル）
        missing_files = tracked_file_paths - current_files
        
        # 共通ファイル（更新が必要かチェックが必要）
        common_files = current_files.intersection(tracked_file_paths)
        
        # 1. 新規ファイルの処理
        for file_path in new_files:
            try:
                logger.info(f"Vectorizing new file: {file_path}")
                vectorize_func(file_path)
                self.update_stats["added_files"] += 1
                
                # 追加されたドキュメント数をカウント
                cursor = self.vector_store.conn.execute(
                    "SELECT chunk_count FROM file_metadata WHERE file_path = ?",
                    (file_path,)
                )
                row = cursor.fetchone()
                if row:
                    self.update_stats["added_documents"] += row[0]
                    
            except Exception as e:
                logger.error(f"Error vectorizing new file {file_path}: {e}")
                self.update_stats["errors"] += 1
        
        # 2. 共通ファイルの更新チェック
        for file_path in common_files:
            try:
                if self.vector_store.file_needs_update(file_path):
                    logger.info(f"Updating changed file: {file_path}")
                    
                    # 現在のチャンク数を記録
                    old_chunks = tracked_files[file_path]["chunk_count"]
                    
                    # 一度削除してから再ベクトル化
                    deleted_count = self.vector_store.delete_file(file_path)
                    self.update_stats["deleted_documents"] += deleted_count
                    
                    # 再ベクトル化
                    vectorize_func(file_path)
                    self.update_stats["updated_files"] += 1
                    
                    # 新たに追加されたチャンク数を記録
                    cursor = self.vector_store.conn.execute(
                        "SELECT chunk_count FROM file_metadata WHERE file_path = ?",
                        (file_path,)
                    )
                    row = cursor.fetchone()
                    if row:
                        self.update_stats["added_documents"] += row[0]
                        
                    logger.info(f"Updated file {file_path}: removed {deleted_count} chunks, added {row[0] if row else 0} chunks")
                else:
                    self.update_stats["unchanged_files"] += 1
            except Exception as e:
                logger.error(f"Error updating file {file_path}: {e}")
                self.update_stats["errors"] += 1
        
        # 3. 削除されたファイルの処理
        if delete_missing:
            for file_path in missing_files:
                try:
                    logger.info(f"Removing deleted file: {file_path}")
                    deleted_count = self.vector_store.delete_file(file_path)
                    self.update_stats["deleted_files"] += 1
                    self.update_stats["deleted_documents"] += deleted_count
                except Exception as e:
                    logger.error(f"Error removing file {file_path}: {e}")
                    self.update_stats["errors"] += 1
        
        # 結果サマリー
        logger.info(f"Incremental update complete: {self.update_stats}")
        return self.update_stats
    
    def get_update_stats(self) -> Dict[str, Any]:
        """更新統計情報を取得"""
        return self.update_stats
