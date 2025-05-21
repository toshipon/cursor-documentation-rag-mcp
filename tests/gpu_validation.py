#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPUサポート検証ツール
GPUが利用可能かどうかを確認し、埋め込み処理の速度をCPUとGPUで比較します
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from vectorize.embeddings import PLaMoEmbedder, DummyEmbedder

def check_gpu_availability() -> Tuple[bool, str]:
    """
    GPUが使用可能かどうかを確認
    
    Returns:
        (GPUの利用可能フラグ, 詳細情報)
    """
    if not torch.cuda.is_available():
        return False, "GPUは検出されませんでした"
    
    # CUDA情報の取得
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
    cuda_version = torch.version.cuda
    
    return True, f"GPU検出: {device_name} (CUDA {cuda_version}), デバイス数: {device_count}"

def generate_test_data(size: int, length_range: Tuple[int, int] = (50, 500)) -> List[str]:
    """
    テスト用のランダムテキストデータを生成
    
    Args:
        size: 生成するテキスト数
        length_range: テキストの長さ範囲（最小, 最大）
        
    Returns:
        テキストのリスト
    """
    vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ,.!?-"
    
    texts = []
    for _ in range(size):
        length = np.random.randint(length_range[0], length_range[1])
        text = ''.join(np.random.choice(list(vocab), size=length))
        texts.append(text)
    
    return texts

def benchmark_embedding(embedder, texts: List[str], batch_size: int = 8) -> Dict:
    """
    埋め込み処理のベンチマーク
    
    Args:
        embedder: 埋め込みモデル
        texts: テキストのリスト
        batch_size: バッチサイズ
        
    Returns:
        ベンチマーク結果
    """
    results = {
        "single_text_times": [],
        "batch_times": [],
        "total_time": 0,
        "total_tokens": 0
    }
    
    # 1. 単一テキスト処理のベンチマーク
    print(f"単一テキスト処理のベンチマーク ({len(texts)} サンプル)...")
    start_time = time.time()
    
    for text in tqdm(texts):
        text_start = time.time()
        embedding = embedder.embed(text)
        text_time = time.time() - text_start
        results["single_text_times"].append(text_time)
    
    single_total_time = time.time() - start_time
    
    # 2. バッチ処理のベンチマーク
    print(f"バッチ処理のベンチマーク (バッチサイズ: {batch_size})...")
    batch_start_time = time.time()
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_start = time.time()
        embeddings = embedder.embed_batch(batch)
        batch_time = time.time() - batch_start
        results["batch_times"].append(batch_time)
    
    batch_total_time = time.time() - batch_start_time
    
    # 結果の集計
    avg_single_time = sum(results["single_text_times"]) / len(results["single_text_times"])
    avg_batch_time = sum(results["batch_times"]) / len(results["batch_times"])
    avg_per_text_in_batch = batch_total_time / len(texts)
    
    # まとめ
    results["summary"] = {
        "single_text_total_time": single_total_time,
        "batch_total_time": batch_total_time,
        "avg_single_text_time": avg_single_time,
        "avg_batch_time": avg_batch_time,
        "avg_per_text_in_batch": avg_per_text_in_batch,
        "speedup_factor": avg_single_time / avg_per_text_in_batch if avg_per_text_in_batch > 0 else 0
    }
    
    return results

def compare_cpu_gpu(model_path: str = None, sample_size: int = 100, batch_size: int = 8):
    """
    CPUとGPUの処理速度を比較
    
    Args:
        model_path: モデルパス（指定なしの場合はconfig.EMBEDDING_MODEL_PATH）
        sample_size: テストサンプル数
        batch_size: バッチサイズ
    """
    # GPUの利用可否を確認
    gpu_available, gpu_info = check_gpu_availability()
    print(f"GPU検証: {gpu_info}")
    
    if not gpu_available:
        print("GPUが利用できないため、CPUのみでベンチマークを実行します")
    
    # テストデータの生成
    print(f"{sample_size}個のテストデータを生成中...")
    test_texts = generate_test_data(sample_size)
    
    # CPUでの処理
    print("\n--- CPU処理 ---")
    cpu_embedder = PLaMoEmbedder(model_path=model_path, device="cpu")
    cpu_results = benchmark_embedding(cpu_embedder, test_texts, batch_size)
    
    # GPUでの処理（利用可能な場合）
    gpu_results = None
    if gpu_available:
        print("\n--- GPU処理 ---")
        gpu_embedder = PLaMoEmbedder(model_path=model_path, device="cuda")
        gpu_results = benchmark_embedding(gpu_embedder, test_texts, batch_size)
    
    # 結果の表示
    print("\n===== ベンチマーク結果 =====")
    print(f"サンプル数: {sample_size}, バッチサイズ: {batch_size}")
    
    print("\nCPU結果:")
    print(f"  単一テキスト処理平均: {cpu_results['summary']['avg_single_text_time']:.4f}秒")
    print(f"  バッチ処理平均: {cpu_results['summary']['avg_batch_time']:.4f}秒")
    print(f"  バッチ内1テキストあたり平均: {cpu_results['summary']['avg_per_text_in_batch']:.4f}秒")
    print(f"  バッチ処理の高速化率: {cpu_results['summary']['speedup_factor']:.2f}倍")
    
    if gpu_results:
        print("\nGPU結果:")
        print(f"  単一テキスト処理平均: {gpu_results['summary']['avg_single_text_time']:.4f}秒")
        print(f"  バッチ処理平均: {gpu_results['summary']['avg_batch_time']:.4f}秒")
        print(f"  バッチ内1テキストあたり平均: {gpu_results['summary']['avg_per_text_in_batch']:.4f}秒")
        print(f"  バッチ処理の高速化率: {gpu_results['summary']['speedup_factor']:.2f}倍")
        
        # CPUとGPUの比較
        cpu_time = cpu_results['summary']['single_text_total_time']
        gpu_time = gpu_results['summary']['single_text_total_time']
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        print(f"\nCPU vs GPU 高速化率: {speedup:.2f}倍")
        
        # バッチ処理の比較
        cpu_batch_time = cpu_results['summary']['batch_total_time']
        gpu_batch_time = gpu_results['summary']['batch_total_time']
        batch_speedup = cpu_batch_time / gpu_batch_time if gpu_batch_time > 0 else 0
        
        print(f"バッチ処理でのCPU vs GPU 高速化率: {batch_speedup:.2f}倍")
    
    # グラフの表示
    plot_results(cpu_results, gpu_results, sample_size, batch_size)

def plot_results(cpu_results, gpu_results, sample_size, batch_size):
    """
    ベンチマーク結果をグラフ化
    
    Args:
        cpu_results: CPU処理の結果
        gpu_results: GPU処理の結果
        sample_size: サンプル数
        batch_size: バッチサイズ
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 処理速度の比較グラフ
    categories = ['単一テキスト', 'バッチ内1テキスト']
    cpu_times = [
        cpu_results['summary']['avg_single_text_time'],
        cpu_results['summary']['avg_per_text_in_batch']
    ]
    
    ax1.bar(categories, cpu_times, width=0.4, label='CPU', alpha=0.7, color='blue')
    
    if gpu_results:
        gpu_times = [
            gpu_results['summary']['avg_single_text_time'],
            gpu_results['summary']['avg_per_text_in_batch']
        ]
        # 横にずらして表示
        ax1.bar([c + 0.4 for c in range(len(categories))], gpu_times, width=0.4, label='GPU', alpha=0.7, color='orange')
    
    ax1.set_ylabel('処理時間（秒）')
    ax1.set_title('テキスト処理時間の比較')
    ax1.legend()
    
    # バッチサイズによる処理時間の変化（仮想データ）
    batch_sizes = list(range(1, min(sample_size, 32) + 1, 4))
    if 1 not in batch_sizes:
        batch_sizes = [1] + batch_sizes
    
    # 簡易的な処理時間推定
    cpu_batch_times = []
    for bs in batch_sizes:
        if bs == 1:
            time_per_sample = cpu_results['summary']['avg_single_text_time']
        else:
            time_per_sample = cpu_results['summary']['avg_per_text_in_batch'] * (batch_size / bs if bs < batch_size else 1)
        cpu_batch_times.append(time_per_sample * sample_size)
    
    ax2.plot(batch_sizes, cpu_batch_times, marker='o', label='CPU', color='blue')
    
    if gpu_results:
        gpu_batch_times = []
        for bs in batch_sizes:
            if bs == 1:
                time_per_sample = gpu_results['summary']['avg_single_text_time']
            else:
                time_per_sample = gpu_results['summary']['avg_per_text_in_batch'] * (batch_size / bs if bs < batch_size else 1)
            gpu_batch_times.append(time_per_sample * sample_size)
        
        ax2.plot(batch_sizes, gpu_batch_times, marker='o', label='GPU', color='orange')
    
    ax2.set_xlabel('バッチサイズ')
    ax2.set_ylabel('全サンプル処理時間（秒）')
    ax2.set_title(f'{sample_size}サンプルの処理時間とバッチサイズの関係')
    ax2.legend()
    
    plt.tight_layout()
    
    # ディレクトリが存在しない場合は作成
    os.makedirs("reports", exist_ok=True)
    
    # グラフを保存
    filename = f"reports/gpu_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename)
    print(f"\nグラフを保存しました: {filename}")
    
    plt.show()

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="GPUサポート検証ツール")
    parser.add_argument("--model-path", type=str, default=None,
                       help="埋め込みモデルのパス（デフォルト: config.EMBEDDING_MODEL_PATH）")
    parser.add_argument("--samples", type=int, default=100,
                       help="ベンチマーク用のサンプル数（デフォルト: 100）")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="バッチ処理のサイズ（デフォルト: 8）")
    
    args = parser.parse_args()
    
    compare_cpu_gpu(args.model_path, args.samples, args.batch_size)

if __name__ == "__main__":
    main()