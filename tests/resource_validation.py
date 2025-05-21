#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dockerコンテナのリソース使用状況検証ツール
メモリ使用量とCPU使用率を監視し、パフォーマンス最適化の効果を検証します
"""

import os
import sys
import time
import json
import argparse
import subprocess
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple

# 監視間隔（秒）
MONITOR_INTERVAL = 2
# デフォルトの監視時間（分）
DEFAULT_DURATION = 10

def get_container_stats(container_name: str) -> Dict:
    """
    指定したコンテナのリソース使用状況を取得
    
    Args:
        container_name: 監視対象のコンテナ名
        
    Returns:
        リソース使用状況を含む辞書
    """
    try:
        result = subprocess.run(
            ["docker", "stats", container_name, "--no-stream", "--format", "{{json .}}"],
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError) as e:
        print(f"Error getting stats for container {container_name}: {e}")
        return {}

def parse_memory_usage(memory_str: str) -> float:
    """
    メモリ使用量文字列をMB単位の数値に変換
    
    Args:
        memory_str: "123.4MiB / 8GiB" 形式のメモリ使用量文字列
        
    Returns:
        メモリ使用量（MB）
    """
    try:
        # "123.4MiB / 8GiB" -> "123.4MiB"
        usage_part = memory_str.split('/')[0].strip()
        
        # 単位を処理
        if 'GiB' in usage_part:
            return float(usage_part.replace('GiB', '')) * 1024
        elif 'MiB' in usage_part:
            return float(usage_part.replace('MiB', ''))
        elif 'KiB' in usage_part:
            return float(usage_part.replace('KiB', '')) / 1024
        else:
            return 0
    except Exception as e:
        print(f"Error parsing memory usage {memory_str}: {e}")
        return 0

def parse_cpu_usage(cpu_str: str) -> float:
    """
    CPU使用率文字列を数値に変換
    
    Args:
        cpu_str: "12.34%" 形式のCPU使用率文字列
        
    Returns:
        CPU使用率（%）
    """
    try:
        return float(cpu_str.replace('%', ''))
    except Exception as e:
        print(f"Error parsing CPU usage {cpu_str}: {e}")
        return 0

def monitor_containers(containers: List[str], duration_minutes: int = DEFAULT_DURATION, plot: bool = True) -> Dict:
    """
    指定した時間、コンテナのリソース使用状況を監視
    
    Args:
        containers: 監視対象のコンテナ名リスト
        duration_minutes: 監視時間（分）
        plot: グラフを生成するかどうか
        
    Returns:
        コンテナごとのリソース使用状況の時系列データ
    """
    # 結果格納用辞書
    results = {container: {"time": [], "memory_mb": [], "cpu_percent": []} for container in containers}
    
    # 監視回数を計算
    iterations = int(duration_minutes * 60 / MONITOR_INTERVAL)
    start_time = time.time()
    
    print(f"Starting resource monitoring for {duration_minutes} minutes...")
    print(f"Monitoring containers: {', '.join(containers)}")
    
    try:
        for i in range(iterations):
            current_time = time.time() - start_time
            
            for container in containers:
                stats = get_container_stats(container)
                if not stats:
                    continue
                
                # メモリとCPU使用率を解析
                memory_mb = parse_memory_usage(stats.get("MemUsage", "0MiB / 0MiB"))
                cpu_percent = parse_cpu_usage(stats.get("CPUPerc", "0%"))
                
                # 結果を記録
                results[container]["time"].append(current_time / 60)  # 分単位
                results[container]["memory_mb"].append(memory_mb)
                results[container]["cpu_percent"].append(cpu_percent)
                
                print(f"[{i+1}/{iterations}] {container}: Memory: {memory_mb:.2f}MB, CPU: {cpu_percent:.2f}%")
            
            # 次の監視までスリープ
            if i < iterations - 1:
                time.sleep(MONITOR_INTERVAL)
    
    except KeyboardInterrupt:
        print("Monitoring interrupted by user.")
    
    # 結果を表示
    print("\nMonitoring complete. Summary:")
    for container in containers:
        if results[container]["memory_mb"]:
            avg_memory = sum(results[container]["memory_mb"]) / len(results[container]["memory_mb"])
            max_memory = max(results[container]["memory_mb"])
            avg_cpu = sum(results[container]["cpu_percent"]) / len(results[container]["cpu_percent"])
            max_cpu = max(results[container]["cpu_percent"])
            
            print(f"{container}:")
            print(f"  Average Memory: {avg_memory:.2f}MB (Max: {max_memory:.2f}MB)")
            print(f"  Average CPU: {avg_cpu:.2f}% (Max: {max_cpu:.2f}%)")
    
    # グラフを生成
    if plot and any(results[container]["memory_mb"] for container in containers):
        plot_results(results, containers)
    
    return results

def plot_results(results: Dict, containers: List[str]):
    """
    監視結果をグラフ化
    
    Args:
        results: 監視結果データ
        containers: コンテナ名リスト
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # タイムスタンプ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # メモリ使用量のグラフ
    for container in containers:
        if results[container]["memory_mb"]:
            ax1.plot(
                results[container]["time"], 
                results[container]["memory_mb"],
                label=container,
                marker='o',
                markersize=3,
                linestyle='-'
            )
    
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Memory Usage (MB)')
    ax1.set_title('Container Memory Usage Over Time')
    ax1.grid(True)
    ax1.legend()
    
    # CPU使用率のグラフ
    for container in containers:
        if results[container]["cpu_percent"]:
            ax2.plot(
                results[container]["time"], 
                results[container]["cpu_percent"],
                label=container,
                marker='o',
                markersize=3,
                linestyle='-'
            )
    
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('CPU Usage (%)')
    ax2.set_title('Container CPU Usage Over Time')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    # ディレクトリが存在しない場合は作成
    os.makedirs("reports", exist_ok=True)
    
    # グラフを保存
    filename = f"reports/resource_usage_{timestamp}.png"
    plt.savefig(filename)
    print(f"Graph saved as {filename}")
    
    # グラフを表示
    plt.show()

def load_test_query(query_count: int = 100, concurrent: int = 10):
    """
    MCPサーバーに対する負荷テスト（クエリ）
    
    Args:
        query_count: 実行するクエリの総数
        concurrent: 同時接続数
    """
    print(f"Running load test with {query_count} queries ({concurrent} concurrent)...")
    
    # loadtestツールを使った負荷テスト
    cmd = [
        "python", "-m", "locust",
        "--host=http://localhost:8000",
        "--headless",
        "--users", str(concurrent),
        "--spawn-rate", "5",
        "--run-time", "2m",
        "--csv=reports/loadtest"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("Load test completed")
    except subprocess.SubprocessError as e:
        print(f"Error running load test: {e}")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Docker container resource monitoring tool")
    parser.add_argument("--containers", nargs="+", default=["mcp-server", "file-watcher", "scheduled-vectorization"],
                       help="Container names to monitor")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION,
                       help=f"Monitoring duration in minutes (default: {DEFAULT_DURATION})")
    parser.add_argument("--no-plot", action="store_true",
                       help="Disable plotting graphs")
    parser.add_argument("--load-test", action="store_true",
                       help="Run load test after monitoring")
    
    args = parser.parse_args()
    
    # コンテナ監視
    results = monitor_containers(args.containers, args.duration, not args.no_plot)
    
    # 負荷テスト（オプショナル）
    if args.load_test:
        load_test_query()

if __name__ == "__main__":
    main()