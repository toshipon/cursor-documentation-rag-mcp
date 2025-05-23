#!/usr/bin/env python3
"""
テストファイルをQdrantに直接追加するためのスクリプト
"""
import os
import sys
import numpy as np
import requests
import json

# Qdrantの接続情報
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "documents"
VECTOR_SIZE = 512

def add_test_document():
    """テストドキュメントを追加"""
    # ランダムベクトルを生成
    vector = np.random.rand(VECTOR_SIZE).tolist()
    
    # テストドキュメントデータ
    point_data = {
        "id": 999999,  # 明示的なID
        "vector": vector,
        "payload": {
            "content": "これはテスト文書です。ファイルウォッチャーと検索機能のテストをしています。",
            "source": "manual_test",
            "source_type": "text",
            "metadata": {
                "path": "/app/data/documents/manual_test.md",
                "title": "テスト文書"
            },
            "created_at": 1716500000  # Unixタイムスタンプ
        }
    }
    
    # ポイントを追加
    url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points"
    headers = {"Content-Type": "application/json"}
    data = {"points": [point_data]}
    
    try:
        response = requests.put(url, headers=headers, data=json.dumps(data))
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.json()}")
        
        if response.status_code == 200:
            print("Document added successfully")
            return True
        else:
            print(f"Failed to add document: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def check_document_count():
    """ドキュメント数の確認"""
    url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/count"
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, headers=headers, data="{}")
        data = response.json()
        count = data.get("result", {}).get("count", 0)
        print(f"Document count: {count}")
        return count
    except Exception as e:
        print(f"Error checking document count: {e}")
        return 0

def test_search():
    """MCPサーバーで検索テスト"""
    url = "http://localhost:8000/query"
    headers = {"Content-Type": "application/json"}
    data = {
        "query": "テスト文書",
        "top_k": 5
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        print(f"Search response status: {response.status_code}")
        if response.status_code == 200:
            results = response.json()
            print(f"Search results: {json.dumps(results, ensure_ascii=False, indent=2)}")
        else:
            print(f"Search failed: {response.text}")
    except Exception as e:
        print(f"Error during search: {e}")

if __name__ == "__main__":
    print("Checking initial document count...")
    initial_count = check_document_count()
    
    print("\nAdding test document...")
    if add_test_document():
        print("\nChecking updated document count...")
        new_count = check_document_count()
        
        if new_count > initial_count:
            print("\nDocument added successfully. Testing search...")
            test_search()
        else:
            print("\nDocument may not have been added. Testing search anyway...")
            test_search()
    else:
        print("Failed to add document. Exiting.")
