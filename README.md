# cursor-documentation-rag-mcp

## 概要
社内ドキュメント（Confluence, PDF, ソースコード等）をベクター化し、MCP（Memory Context Provider）経由で検索・利用できるシステムです。Cursorから関連情報を素早く取得し、効率的なドキュメンテーションを実現します。

## 主な機能
- ドキュメントのベクター化（PLaMo-Embedding-1B利用）
- SQLite-VSSによるベクトルDB格納・検索
- FastAPIベースのMCPサーバー
- Markdown/PDF/コード対応
- Dockerによる簡単なデプロイ

## セットアップ

```bash
# 依存パッケージのインストール
pip install -r requirements.txt
```

## 使い方

### ドキュメントのベクター化

```bash
python scripts/vectorize_docs.py --input_dir [ドキュメントディレクトリ] --output_db [DBパス]
```

### MCPサーバーの起動

```bash
python scripts/start_mcp_server.py
```

またはDockerを利用

```bash
docker-compose -f docker/docker-compose.yml up -d
```

## ディレクトリ構成（抜粋）

- vectorize/ : ベクター化・テキスト分割・各種プロセッサ
- db/ : ベクトルDB管理
- mcp/ : サーバー実装
- scripts/ : 実行スクリプト
- docker/ : Docker関連ファイル

## ライセンス
MIT