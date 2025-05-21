import os
import sys
import json
import time
import logging
import threading
import psutil
import socket
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, Gauge, Summary, REGISTRY, generate_latest

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# ロギング設定
logger = logging.getLogger(__name__)

class MCPMonitor:
    """
    MCPサーバーのモニタリングを行うクラス
    - リクエスト数、レスポンス時間などのメトリクスを収集
    - サーバーのリソース使用状況を監視
    - Prometheusエクスポーター機能を提供
    """
    
    def __init__(self, app: Optional[FastAPI] = None, monitoring_interval: int = 30):
        """
        初期化
        
        Args:
            app: FastAPIアプリケーション（ミドルウェア登録用）
            monitoring_interval: リソース監視の間隔（秒）
        """
        # 基本設定
        self.hostname = socket.gethostname()
        self.start_time = time.time()
        self.monitoring_interval = monitoring_interval
        self.process = psutil.Process(os.getpid())
        
        # メトリクス
        self._init_metrics()
        
        # リソース統計データ
        self.resource_stats = {
            "cpu_percent": [],
            "memory_percent": [],
            "memory_mb": [],
            "connections": [],
            "timestamps": []
        }
        
        # 監視スレッド
        self.monitoring_active = False
        self.monitor_thread = None
        
        # FastAPIアプリが提供された場合はミドルウェアを登録
        if app:
            self.register_middleware(app)
    
    def _init_metrics(self):
        """Prometheusメトリクスの初期化"""
        # カウンター: リクエスト総数
        self.request_counter = Counter(
            "mcp_requests_total", 
            "Total number of requests",
            ["method", "endpoint", "status"]
        )
        
        # ヒストグラム: リクエスト処理時間
        self.request_latency = Histogram(
            "mcp_request_latency_seconds",
            "Request latency in seconds",
            ["method", "endpoint"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
        )
        
        # サマリー: クエリ処理時間
        self.query_processing_time = Summary(
            "mcp_query_processing_seconds", 
            "Time spent processing query requests",
            ["type"]  # シングルクエリかバッチクエリか
        )
        
        # カウンター: 処理したクエリ数
        self.query_counter = Counter(
            "mcp_queries_processed_total",
            "Total number of queries processed",
            ["type", "cached"]  # タイプ（シングル/バッチ）とキャッシュヒットかどうか
        )
        
        # カウンター: エラー数
        self.error_counter = Counter(
            "mcp_errors_total",
            "Total number of errors",
            ["type"]  # エラータイプ
        )
        
        # ゲージ: 現在の接続数
        self.connections_gauge = Gauge(
            "mcp_active_connections",
            "Number of currently active connections"
        )
        
        # ゲージ: CPU使用率
        self.cpu_gauge = Gauge(
            "mcp_cpu_usage_percent",
            "CPU usage in percent"
        )
        
        # ゲージ: メモリ使用量
        self.memory_gauge = Gauge(
            "mcp_memory_usage_mb",
            "Memory usage in MB"
        )
        
        # ゲージ: キャッシュサイズ
        self.cache_size_gauge = Gauge(
            "mcp_cache_size",
            "Number of items in cache",
            ["cache_type"]  # クエリキャッシュか埋め込みキャッシュか
        )
    
    def register_middleware(self, app: FastAPI):
        """
        FastAPIアプリケーションにミドルウェアを登録
        
        Args:
            app: FastAPIアプリケーション
        """
        @app.middleware("http")
        async def monitoring_middleware(request, call_next):
            # リクエスト開始時間
            start_time = time.time()
            
            # 接続数を増加
            self.connections_gauge.inc()
            
            try:
                # リクエストを処理
                response = await call_next(request)
                
                # メトリクスを記録
                latency = time.time() - start_time
                method = request.method
                endpoint = request.url.path
                status = response.status_code
                
                # リクエスト数とレイテンシーを記録
                self.request_counter.labels(method=method, endpoint=endpoint, status=status).inc()
                self.request_latency.labels(method=method, endpoint=endpoint).observe(latency)
                
                return response
            except Exception as e:
                # エラーを記録
                self.error_counter.labels(type=type(e).__name__).inc()
                raise
            finally:
                # 接続数を減少
                self.connections_gauge.dec()
        
        @app.on_event("startup")
        async def start_monitoring():
            self.start_monitoring()
        
        @app.on_event("shutdown")
        async def stop_monitoring():
            self.stop_monitoring()
    
    def record_query(self, query_type: str, processing_time: float, cached: bool = False):
        """
        クエリ処理を記録
        
        Args:
            query_type: クエリタイプ（"single"または"batch"）
            processing_time: 処理時間（秒）
            cached: キャッシュヒットかどうか
        """
        self.query_processing_time.labels(type=query_type).observe(processing_time)
        self.query_counter.labels(type=query_type, cached=str(cached).lower()).inc()
    
    def record_error(self, error_type: str):
        """
        エラーを記録
        
        Args:
            error_type: エラータイプ
        """
        self.error_counter.labels(type=error_type).inc()
    
    def update_cache_size(self, cache_type: str, size: int):
        """
        キャッシュサイズを更新
        
        Args:
            cache_type: キャッシュタイプ（"query"または"embedding"）
            size: キャッシュ内のアイテム数
        """
        self.cache_size_gauge.labels(cache_type=cache_type).set(size)
    
    def _collect_resource_metrics(self):
        """サーバーリソースのメトリクスを収集"""
        # CPU使用率
        cpu_percent = self.process.cpu_percent(interval=0.1)
        self.cpu_gauge.set(cpu_percent)
        
        # メモリ使用量
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # MBに変換
        memory_percent = self.process.memory_percent()
        self.memory_gauge.set(memory_mb)
        
        # 現在のネットワーク接続数
        connections = len(self.process.connections())
        
        # 統計データの保存（直近10分のデータを保持）
        now = datetime.now().isoformat()
        self.resource_stats["cpu_percent"].append(cpu_percent)
        self.resource_stats["memory_percent"].append(memory_percent)
        self.resource_stats["memory_mb"].append(memory_mb)
        self.resource_stats["connections"].append(connections)
        self.resource_stats["timestamps"].append(now)
        
        # 直近10分のデータだけを保持
        max_samples = 10 * 60 // self.monitoring_interval
        if len(self.resource_stats["timestamps"]) > max_samples:
            for key in self.resource_stats:
                self.resource_stats[key] = self.resource_stats[key][-max_samples:]
    
    def _monitoring_task(self):
        """モニタリングスレッドのメインタスク"""
        while self.monitoring_active:
            try:
                self._collect_resource_metrics()
            except Exception as e:
                logger.error(f"Error in monitoring task: {e}")
            
            # 次の収集までスリープ
            time.sleep(self.monitoring_interval)
    
    def start_monitoring(self):
        """モニタリングの開始"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitoring_task, daemon=True)
            self.monitor_thread.start()
            logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """モニタリングの停止"""
        if self.monitoring_active:
            self.monitoring_active = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=2.0)
            logger.info("Resource monitoring stopped")
    
    def get_metrics(self) -> str:
        """
        Prometheusメトリクスを取得
        
        Returns:
            Prometheus形式のメトリクス文字列
        """
        return generate_latest().decode("utf-8")
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """
        リソース使用状況の統計を取得
        
        Returns:
            リソース統計データの辞書
        """
        result = self.resource_stats.copy()
        
        # 基本的な統計情報を追加
        result["uptime_seconds"] = int(time.time() - self.start_time)
        result["hostname"] = self.hostname
        
        # 最新のリソース使用状況
        if result["cpu_percent"]:
            result["current"] = {
                "cpu_percent": result["cpu_percent"][-1],
                "memory_mb": result["memory_mb"][-1],
                "memory_percent": result["memory_percent"][-1],
                "connections": result["connections"][-1],
                "timestamp": result["timestamps"][-1]
            }
        
        # 平均値
        if result["cpu_percent"]:
            result["average"] = {
                "cpu_percent": sum(result["cpu_percent"]) / len(result["cpu_percent"]),
                "memory_mb": sum(result["memory_mb"]) / len(result["memory_mb"]),
                "memory_percent": sum(result["memory_percent"]) / len(result["memory_percent"]),
                "connections": sum(result["connections"]) / len(result["connections"])
            }
        
        return result