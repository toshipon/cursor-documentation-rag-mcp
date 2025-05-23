#!/usr/bin/env python3
"""
File Watcher統合テストスクリプト
file watcherがQdrantにデータを正しく投入できるかを検証する
"""

import os
import sys
import time
import json
import logging
import requests
from typing import Dict, Any

# プロジェクトのルートディレクトリをシステムパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FileWatcherIntegrationTest:
    """File Watcher統合テストクラス"""
    
    def __init__(self):
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = os.getenv("QDRANT_COLLECTION", "documents")
        self.test_file_path = "data/documents/integration_test.md"
        
    def check_qdrant_connection(self) -> bool:
        """Qdrantへの接続をチェック"""
        try:
            response = requests.get(f"{self.qdrant_url}/collections")
            if response.status_code == 200:
                logger.info("Qdrant connection successful")
                return True
            else:
                logger.error(f"Qdrant connection failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """コレクション情報を取得"""
        try:
            response = requests.get(f"{self.qdrant_url}/collections/{self.collection_name}")
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Collection {self.collection_name} not found")
                return {}
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
    
    def create_test_file(self) -> None:
        """テスト用ファイルを作成"""
        test_content = f"""# File Watcher統合テスト

このファイルは統合テストのために作成されました。
作成時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}

## テスト目的

1. File watcherがファイル作成を検出できるか
2. ベクトル化処理が正常に実行されるか
3. Qdrantにデータが正しく投入されるか

## 期待される結果

このファイルの内容がベクトル化され、Qdrantの`{self.collection_name}`コレクションに保存される。

## テストデータ

- ファイルパス: {self.test_file_path}
- コレクション: {self.collection_name}
- Qdrant URL: {self.qdrant_url}
"""
        
        # ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(self.test_file_path), exist_ok=True)
        
        with open(self.test_file_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        logger.info(f"Test file created: {self.test_file_path}")
    
    def wait_for_processing(self, timeout: int = 60) -> bool:
        """ベクトル化処理の完了を待機"""
        logger.info(f"Waiting for file processing (timeout: {timeout}s)...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # コレクション内のドキュメント数をチェック
                collection_info = self.get_collection_info()
                if collection_info and 'result' in collection_info:
                    points_count = collection_info['result'].get('points_count', 0)
                    if points_count > 0:
                        logger.info(f"Documents found in collection: {points_count}")
                        return True
                
                time.sleep(5)  # 5秒待機
                
            except Exception as e:
                logger.error(f"Error checking collection: {e}")
                time.sleep(5)
        
        logger.warning("Timeout waiting for file processing")
        return False
    
    def search_test_document(self) -> bool:
        """テストドキュメントが検索できるかチェック"""
        try:
            # 簡単なダミーベクトルで検索（実際のベクトルは不要）
            search_payload = {
                "vector": [0.1] * 512,  # 512次元のダミーベクトル
                "limit": 10,
                "with_payload": True
            }
            
            response = requests.post(
                f"{self.qdrant_url}/collections/{self.collection_name}/points/search",
                json=search_payload
            )
            
            if response.status_code == 200:
                results = response.json()
                if 'result' in results and results['result']:
                    # テストファイルのパスを含むドキュメントを探す
                    for result in results['result']:
                        payload = result.get('payload', {})
                        source = payload.get('source', '')
                        if self.test_file_path in source:
                            logger.info(f"Test document found in search results: {source}")
                            return True
                    
                    logger.warning("Test document not found in search results")
                    return False
                else:
                    logger.warning("No search results returned")
                    return False
            else:
                logger.error(f"Search request failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error searching for test document: {e}")
            return False
    
    def cleanup_test_file(self) -> None:
        """テストファイルを削除"""
        try:
            if os.path.exists(self.test_file_path):
                os.remove(self.test_file_path)
                logger.info(f"Test file removed: {self.test_file_path}")
        except Exception as e:
            logger.error(f"Error removing test file: {e}")
    
    def run_test(self) -> bool:
        """統合テストを実行"""
        logger.info("Starting File Watcher Integration Test")
        
        # 1. Qdrant接続チェック
        if not self.check_qdrant_connection():
            logger.error("Qdrant connection failed")
            return False
        
        # 2. 初期状態のコレクション情報を取得
        initial_info = self.get_collection_info()
        initial_count = 0
        if initial_info and 'result' in initial_info:
            initial_count = initial_info['result'].get('points_count', 0)
        logger.info(f"Initial document count: {initial_count}")
        
        # 3. テストファイルを作成
        self.create_test_file()
        
        # 4. ベクトル化処理の完了を待機
        if not self.wait_for_processing():
            logger.error("File processing timeout")
            self.cleanup_test_file()
            return False
        
        # 5. 最終状態のコレクション情報を取得
        final_info = self.get_collection_info()
        final_count = 0
        if final_info and 'result' in final_info:
            final_count = final_info['result'].get('points_count', 0)
        logger.info(f"Final document count: {final_count}")
        
        # 6. ドキュメント数の増加をチェック
        if final_count > initial_count:
            logger.info(f"Document count increased: {initial_count} -> {final_count}")
        else:
            logger.warning(f"Document count did not increase: {initial_count} -> {final_count}")
        
        # 7. テストドキュメントの検索
        search_success = self.search_test_document()
        
        # 8. テストファイルをクリーンアップ
        self.cleanup_test_file()
        
        # 9. 結果判定
        success = (final_count > initial_count) and search_success
        
        if success:
            logger.info("✅ File Watcher Integration Test PASSED")
        else:
            logger.error("❌ File Watcher Integration Test FAILED")
        
        return success

def main():
    """メイン実行関数"""
    test = FileWatcherIntegrationTest()
    success = test.run_test()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()