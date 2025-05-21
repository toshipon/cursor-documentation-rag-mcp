#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MCPサーバーに対する負荷テスト
Locustフレームワークを使用した高負荷シミュレーション
"""

import os
import sys
import json
import random
import time
from locust import HttpUser, task, between, events
from typing import List, Dict, Any

# サンプルクエリ（テスト用）
SAMPLE_QUERIES = [
    "Docker環境構築方法",
    "FastAPIとは何か",
    "ベクトルデータベースの特徴",
    "PLaMoエンベディングモデル",
    "Pythonパフォーマンス最適化",
    "分散システムアーキテクチャ",
    "自然言語処理技術",
    "マイクロサービス設計パターン",
    "RESTful API設計",
    "テキスト分類手法",
    "データ分析フレームワーク",
    "AI開発環境セットアップ",
    "クラウドネイティブアプリケーション",
    "Kubernetes基本概念",
    "APIゲートウェイの役割"
]

# 各リクエストのメトリクスを保存するグローバル辞書
metrics = {
    "query": {"latencies": [], "failures": 0, "successes": 0},
    "batch_query": {"latencies": [], "failures": 0, "successes": 0}
}

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """テスト開始時に実行"""
    print("Starting load test...")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """テスト終了時に実行"""
    print("\nLoad test completed. Summary:")

    for endpoint, data in metrics.items():
        if data["latencies"]:
            avg_latency = sum(data["latencies"]) / len(data["latencies"])
            max_latency = max(data["latencies"]) if data["latencies"] else 0
            min_latency = min(data["latencies"]) if data["latencies"] else 0
            p95_latency = sorted(data["latencies"])[int(len(data["latencies"]) * 0.95)] if len(data["latencies"]) > 20 else max_latency
            total_requests = data["successes"] + data["failures"]
            success_rate = data["successes"] / total_requests * 100 if total_requests > 0 else 0

            print(f"\nEndpoint: {endpoint}")
            print(f"  Total requests: {total_requests}")
            print(f"  Success rate: {success_rate:.2f}%")
            print(f"  Avg latency: {avg_latency:.2f}ms")
            print(f"  Min latency: {min_latency:.2f}ms")
            print(f"  Max latency: {max_latency:.2f}ms")
            print(f"  95th percentile: {p95_latency:.2f}ms")

class MCPUser(HttpUser):
    """MCPサーバーに対するテストユーザー"""
    wait_time = between(1, 3)  # リクエスト間の待機時間（秒）

    def on_start(self):
        """ユーザーセッション開始時に実行"""
        pass

    @task(8)
    def query_single(self):
        """単一クエリを実行（高頻度）"""
        query = random.choice(SAMPLE_QUERIES)
        start_time = time.time()
        
        with self.client.post(
            "/query",
            json={
                "query": query,
                "top_k": random.randint(3, 10),
                "cache": bool(random.getrandbits(1))
            },
            catch_response=True
        ) as response:
            latency = (time.time() - start_time) * 1000  # ms単位
            
            if response.status_code == 200:
                metrics["query"]["latencies"].append(latency)
                metrics["query"]["successes"] += 1
            else:
                metrics["query"]["failures"] += 1
                response.failure(f"Failed with status code: {response.status_code}")

    @task(2)
    def query_batch(self):
        """バッチクエリを実行（低頻度）"""
        # ランダムな数のクエリを選択
        batch_size = random.randint(2, 5)
        queries = random.sample(SAMPLE_QUERIES, batch_size)
        
        start_time = time.time()
        with self.client.post(
            "/batch_query",
            json={
                "queries": queries,
                "top_k": random.randint(3, 5),
                "cache": bool(random.getrandbits(1))
            },
            catch_response=True
        ) as response:
            latency = (time.time() - start_time) * 1000  # ms単位
            
            if response.status_code == 200:
                metrics["batch_query"]["latencies"].append(latency)
                metrics["batch_query"]["successes"] += 1
            else:
                metrics["batch_query"]["failures"] += 1
                response.failure(f"Failed with status code: {response.status_code}")

    @task(1)
    def get_stats(self):
        """統計情報を取得（最低頻度）"""
        self.client.get("/stats")

def run_load_test():
    """直接実行された場合にテストを実行"""
    import subprocess
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Server Load Test")
    parser.add_argument("--users", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--time", type=str, default="1m", help="Test duration (e.g., 30s, 1m, 5m)")
    parser.add_argument("--host", type=str, default="http://localhost:8000", help="MCP server URL")
    args = parser.parse_args()
    
    # コマンドラインからLocustを実行
    cmd = [
        "locust", 
        "-f", __file__,
        "--headless",
        "--host", args.host,
        "--users", str(args.users),
        "--spawn-rate", "5",
        "--run-time", args.time
    ]
    
    print(f"Running load test with {args.users} users for {args.time}...")
    subprocess.run(cmd)

if __name__ == "__main__":
    run_load_test()