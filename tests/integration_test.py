#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Docker環境でのインテグレーションテスト
MCPサーバーとVectorStoreの連携を確認するテスト
"""

import os
import sys
import json
import time
import pytest
import requests
import tempfile
import subprocess
from pathlib import Path

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from db.vector_store import VectorStore
from vectorize.embeddings import PLaMoEmbedder, DummyEmbedder

# テスト設定
TEST_SERVER_URL = "http://localhost:8000"
TEST_TIMEOUT = 30  # サーバー起動待機最大秒数

class TestDockerIntegration:
    """Dockerコンテナでのインテグレーションテスト"""
    
    @classmethod
    def setup_class(cls):
        """テスト前の準備"""
        print("Setting up integration test environment...")
        
        # テスト用のテンポラリディレクトリを作成
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.data_dir = Path(cls.temp_dir.name) / "data"
        cls.data_dir.mkdir(exist_ok=True)
        
        # テスト用のドキュメントを作成
        test_docs = [
            {
                "title": "Docker環境構築ガイド",
                "content": "Dockerを使用して簡単に環境を構築できます。まずDockerfileを作成し、必要なパッケージをインストールします。",
                "filename": "docker_guide.md"
            },
            {
                "title": "Pythonパフォーマンス最適化",
                "content": "Pythonアプリケーションのパフォーマンスを向上させるには、キャッシュの活用とバッチ処理が効果的です。",
                "filename": "python_optimization.md"
            },
            {
                "title": "ベクトル検索入門",
                "content": "ベクトル検索は類似度に基づいてドキュメントを検索する方法です。埋め込みモデルを使用してテキストをベクトル化します。",
                "filename": "vector_search.md" 
            }
        ]
        
        # テストドキュメントをファイルに書き込む
        for doc in test_docs:
            with open(cls.data_dir / doc["filename"], "w", encoding="utf-8") as f:
                f.write(f"# {doc['title']}\n\n{doc['content']}")
        
        # Dockerコンテナを起動
        print("Starting Docker containers...")
        cls.docker_process = subprocess.Popen(
            ["docker-compose", "up", "-d"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # サーバーが起動するまで待機
        cls._wait_for_server()
    
    @classmethod
    def teardown_class(cls):
        """テスト後のクリーンアップ"""
        print("Cleaning up test environment...")
        
        # Dockerコンテナを停止
        subprocess.run(["docker-compose", "down"], check=True)
        
        # テンポラリディレクトリを削除
        cls.temp_dir.cleanup()
    
    @classmethod
    def _wait_for_server(cls):
        """サーバーが起動するまで待機"""
        start_time = time.time()
        while time.time() - start_time < TEST_TIMEOUT:
            try:
                response = requests.get(f"{TEST_SERVER_URL}/health", timeout=5)
                if response.status_code == 200:
                    print(f"Server is up after {time.time() - start_time:.2f} seconds")
                    return
            except requests.RequestException:
                pass
            
            time.sleep(1)
            print("Waiting for server to start...")
        
        # タイムアウト時は強制終了
        raise TimeoutError(f"Server did not start within {TEST_TIMEOUT} seconds")
    
    def test_server_health(self):
        """ヘルスチェックエンドポイントのテスト"""
        response = requests.get(f"{TEST_SERVER_URL}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_server_stats(self):
        """統計情報エンドポイントのテスト"""
        response = requests.get(f"{TEST_SERVER_URL}/stats")
        assert response.status_code == 200
        stats = response.json()
        assert "document_count" in stats
        assert "cache_info" in stats
    
    def test_simple_query(self):
        """基本的なクエリのテスト"""
        query_data = {
            "query": "Dockerの使い方",
            "top_k": 3
        }
        response = requests.post(
            f"{TEST_SERVER_URL}/query",
            json=query_data
        )
        assert response.status_code == 200
        result = response.json()
        assert "results" in result
        assert "query_time_ms" in result
    
    def test_batch_query(self):
        """バッチクエリのテスト"""
        batch_query_data = {
            "queries": [
                "Dockerの使い方",
                "Pythonパフォーマンス",
                "ベクトル検索"
            ],
            "top_k": 2
        }
        response = requests.post(
            f"{TEST_SERVER_URL}/batch_query",
            json=batch_query_data
        )
        assert response.status_code == 200
        result = response.json()
        assert "results" in result
        assert len(result["results"]) == 3  # 3つのクエリ結果があるか確認
    
    def test_caching(self):
        """キャッシュ機能のテスト"""
        query_data = {
            "query": "キャッシュテスト",
            "top_k": 3,
            "cache": True
        }
        
        # 1回目のクエリ（キャッシュなし）
        response1 = requests.post(
            f"{TEST_SERVER_URL}/query",
            json=query_data
        )
        assert response1.status_code == 200
        result1 = response1.json()
        assert not result1["cached"]
        
        # 2回目のクエリ（キャッシュあり）
        response2 = requests.post(
            f"{TEST_SERVER_URL}/query",
            json=query_data
        )
        assert response2.status_code == 200
        result2 = response2.json()
        assert result2["cached"]
        
        # キャッシュ利用時の方がレスポンス時間が短いことを確認
        assert result2["query_time_ms"] < result1["query_time_ms"]
    
    def test_filter_query(self):
        """フィルタリング機能のテスト"""
        # ソースタイプでフィルタリング
        filter_query = {
            "query": "検索",
            "top_k": 5,
            "filters": {
                "source_type": "markdown"
            }
        }
        response = requests.post(
            f"{TEST_SERVER_URL}/query",
            json=filter_query
        )
        assert response.status_code == 200
        result = response.json()
        # すべての結果がmarkdownタイプであることを確認
        for doc in result["results"]:
            assert doc["metadata"]["source_type"] == "markdown"
    
    def test_hybrid_search(self):
        """ハイブリッド検索のテスト"""
        hybrid_query = {
            "query": "ベクトル検索と類似度",
            "top_k": 3,
            "vector_weight": 0.6,
            "text_weight": 0.4
        }
        
        response = requests.post(
            f"{TEST_SERVER_URL}/hybrid_search",
            json=hybrid_query
        )
        assert response.status_code == 200
        result = response.json()
        assert "results" in result
        assert "query_time_ms" in result
        
        # ハイブリッド検索はキャッシュが効くことを確認
        # 2回目の呼び出し
        response2 = requests.post(
            f"{TEST_SERVER_URL}/hybrid_search",
            json=hybrid_query
        )
        assert response2.status_code == 200
        result2 = response2.json()
        assert result2["cached"]  # キャッシュされたレスポンス
    
    def test_keyword_search(self):
        """キーワード検索のテスト"""
        keyword_query = {
            "keyword": "Docker環境",
            "top_k": 2
        }
        
        response = requests.post(
            f"{TEST_SERVER_URL}/keyword_search",
            json=keyword_query
        )
        assert response.status_code == 200
        result = response.json()
        assert "results" in result
        
        # Dockerに関連する結果が返ってくることを確認
        found = False
        for doc in result["results"]:
            if "Docker" in doc["content"]:
                found = True
                break
        assert found, "Docker関連の結果が含まれていません"
    
    def test_database_optimization(self):
        """データベース最適化のテスト"""
        # データベースの最適化を実行
        response = requests.post(
            f"{TEST_SERVER_URL}/maintenance/optimize_db"
        )
        assert response.status_code == 200
        result = response.json()
        assert "status" in result
        assert result["status"] == "success"
        
        # 最適化処理の詳細情報が含まれていることを確認
        assert "details" in result
        assert "execution_time_ms" in result["details"]

if __name__ == "__main__":
    # PyTestを使わず直接実行する場合
    test = TestDockerIntegration()
    test.setup_class()
    try:
        test.test_server_health()
        test.test_server_stats()
        test.test_simple_query()
        test.test_batch_query()
        test.test_caching()
        test.test_filter_query()
        test.test_hybrid_search()
        test.test_keyword_search()
        test.test_database_optimization()
        print("All tests passed!")
    finally:
        test.teardown_class()