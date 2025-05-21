# cursor-documentation-rag-mcp - ドキュメント検索システム設計書

## 1. プロジェクト概要

cursor-documentation-rag-mcp は、社内ドキュメント（Confluence, PDF, ソースコード等）をベクター化し、それらを MCP（Memory Context Provider）経由で検索・利用できるようにするシステムです。これにより、Cursor 内から関連性の高い情報を素早く取得し、ドキュメンテーションをより効率的に行うことを可能にします。

## 2. システム設計

### 2.1 アーキテクチャ概要

```
User (Cursor) <---> MCP Server <---> SQLite-VSS <--- Document Vectorizer <--- Raw Documents
```

### 2.2 主要コンポーネント

#### 2.2.1 Document Vectorizer

- ドキュメント取得機能
- テキスト分割機能（MarkdownTextSplitter 等）
- ベクター化機能（PLaMo-Embedding-1B）

#### 2.2.2 Vector Database (SQLite-VSS)

- ベクトルデータ保存
- 類似度検索機能

#### 2.2.3 MCP Server

- クエリ受付・処理
- 検索結果フォーマット
- Cursor との連携インターフェース

### 2.3 データフロー

1. ドキュメント収集: 各ソース（Confluence, ローカルファイル等）からドキュメントを取得
2. 前処理: ドキュメントのフォーマットに応じた処理（PDF→ テキスト変換など）
3. テキスト分割: 適切な大きさにチャンク分割
4. ベクター化: PLaMo-Embedding-1B を使用してチャンクをベクター化
5. データベース格納: SQLite-VSS にベクターとメタデータを保存
6. クエリ処理: MCP Server がクエリを受け取り、ベクター化してデータベースに問い合わせ
7. 結果返却: 関連性の高い情報を Cursor に返却

## 3. 技術選定

| コンポーネント | 選定技術                       | 選定理由                                     |
| -------------- | ------------------------------ | -------------------------------------------- |
| 埋め込みモデル | PLaMo-Embedding-1B             | 日本語に強い埋め込みモデル、オープンソース   |
| テキスト分割   | LangChain MarkdownTextSplitter | マークダウンの構造を考慮した分割が可能       |
| ベクター DB    | SQLite-VSS                     | 軽量、セットアップが容易、ローカル環境で動作 |
| サーバー       | FastAPI                        | 高速、Python ベース、非同期処理サポート      |

## 4. ディレクトリ構成（予定）

```
cursor-documentation-rag-mcp/
├── README.md
├── requirements.txt
├── config.py
├── vectorize/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── text_splitters.py
│   └── processors/
│       ├── __init__.py
│       ├── markdown_processor.py
│       ├── pdf_processor.py
│       └── code_processor.py
├── db/
│   ├── __init__.py
│   └── vector_store.py
├── mcp/
│   ├── __init__.py
│   └── server.py
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh
└── scripts/
    ├── vectorize_docs.py
    └── start_mcp_server.py
```

## 5. API 設計（概要）

### MCP Server API

#### Query Endpoint

- **エンドポイント**: `/query`
- **メソッド**: POST
- **リクエスト例**:
  ```json
  {
    "query": "検索クエリテキスト",
    "top_k": 5,
    "filters": {
      "source_type": ["markdown", "pdf"],
      "path": "optional/path/pattern"
    }
  }
  ```
- **レスポンス例**:
  ```json
  {
    "results": [
      {
        "content": "見つかったテキスト内容",
        "metadata": {
          "source": "ファイルパス",
          "source_type": "ファイルタイプ",
          "chunk_id": "チャンクID",
          "created_at": "作成日時"
        },
        "score": 0.87
      }
    ]
  }
  ```

## 6. 実装計画

- 基本機能実装（ベクター化・DB 格納・MCP サーバー）
- ドキュメント処理機能拡張（PDF/コード対応）
- 検索・フィルタリング機能強化
- パフォーマンス・UX 改善

## 7. 注意点

- 埋め込みモデルの計算リソース
- ドキュメント更新時の再ベクター化
- プライバシー・セキュリティ
- スケーリング

## 8. Docker 化

### 8.1 Docker 構成

MCP サーバーを Docker コンテナとして提供するための設定を定義します。

#### ディレクトリ構成への追加

```
cursor-documentation-rag-mcp/
├── README.md
├── requirements.txt
├── config.py
├── vectorize/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── text_splitters.py
│   └── processors/
│       ├── __init__.py
│       ├── markdown_processor.py
│       ├── pdf_processor.py
│       └── code_processor.py
├── db/
│   ├── __init__.py
│   └── vector_store.py
├── mcp/
│   ├── __init__.py
│   └── server.py
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── entrypoint.sh
└── scripts/
    ├── vectorize_docs.py
    └── start_mcp_server.py
```

#### Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 依存パッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# SQLite-VSSのセットアップ
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# PLaMo-Embedding-1Bモデルのダウンロード（キャッシュ利用）
RUN mkdir -p /app/models
# モデルは事前にダウンロードしてビルド時にコピーするか、初回起動時にダウンロード

# アプリケーションコードのコピー
COPY . .

# 起動スクリプトの設定
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
```

#### docker-compose.yml

```yaml
version: "3"

services:
  mcp-server:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ../data:/app/data
      - ../models:/app/models
      - ../vector_store:/app/vector_store
    environment:
      - EMBEDDING_MODEL_PATH=/app/models/plamo-embedding-1b
      - VECTOR_DB_PATH=/app/vector_store/vector_store.db
      - MCP_SERVER_PORT=8000
      - LOG_LEVEL=INFO
```

#### entrypoint.sh

```bash
#!/bin/bash
set -e

# モデルがない場合はダウンロード
if [ ! -d "$EMBEDDING_MODEL_PATH" ] || [ -z "$(ls -A $EMBEDDING_MODEL_PATH)" ]; then
  echo "Downloading PLaMo-Embedding-1B model..."
  python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='pfnet/plamo-embedding-1b', local_dir='$EMBEDDING_MODEL_PATH')"
fi

# データベースの初期化チェック
if [ ! -f "$VECTOR_DB_PATH" ]; then
  echo "Vector database not found. Please mount a pre-built database or run vectorization first."
fi

# MCPサーバー起動
echo "Starting MCP Server..."
exec python -m mcp.server
```

### 8.2 Docker の使用方法

#### ベクター化の実行

```bash
# ドキュメントをマウントして、ベクター化を実行
docker compose -f docker/docker compose.yml run --rm -v /path/to/documents:/app/data/documents mcp-server python -m scripts.vectorize_docs --input_dir /app/data/documents --output_db /app/vector_store/vector_store.db
```

#### MCP サーバーの起動

```bash
# サーバーの起動
docker compose -f docker/docker compose.yml up -d

# ログの確認
docker compose -f docker/docker compose.yml logs -f
```

#### Curl でのテスト

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "検索クエリテキスト",
    "top_k": 5
  }'
```

### 8.3 デプロイ戦略

- **開発環境**: ローカルの Docker 開発環境
- **テスト環境**: CI/CD パイプラインで Docker イメージをビルド・テスト
- **本番環境**: ビルド済みのイメージをコンテナレジストリ（DockerHub、ECR 等）に保存し、サーバーにデプロイ

### 8.4 スケーリング

- 複数の MCP サーバーインスタンスを起動し、ロードバランサーで負荷分散
- ベクター DB を共有ストレージやマネージドサービスに移行して永続化
- モデルのキャッシングと最適化でリソース使用を効率化

## 9. ドキュメント取り込み自動化

### 9.1 監視ベースの自動取り込み

指定フォルダを監視し、新規・更新ファイルを自動的にベクター化するシステムを構築できます。

#### 9.1.1 ファイル監視ワーカー方式

```
cursor-documentation-rag-mcp/
├── // ...existing code...
└── workers/
    ├── __init__.py
    ├── file_watcher.py
    └── vectorization_worker.py
```

**特徴**:

- `watchdog` ライブラリを使用してファイルシステムイベントを監視
- 変更を検知したらベクター化キューに追加
- 定期的にキューからドキュメントを取り出してベクター化処理を実行

**実装例（file_watcher.py）**:

```python
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import queue

# ベクター化キュー
vectorization_queue = queue.Queue()

class DocumentEventHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and self._is_target_file(event.src_path):
            print(f"New file detected: {event.src_path}")
            vectorization_queue.put({"path": event.src_path, "action": "create"})

    def on_modified(self, event):
        if not event.is_directory and self._is_target_file(event.src_path):
            print(f"File modified: {event.src_path}")
            vectorization_queue.put({"path": event.src_path, "action": "update"})

    def _is_target_file(self, path):
        # 対象とするファイル拡張子を判定
        return path.endswith(('.md', '.pdf', '.txt', '.py', '.js'))

def start_file_watcher(path_to_watch):
    event_handler = DocumentEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path_to_watch, recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
```

**実装例（vectorization_worker.py）**:

```python
import time
from file_watcher import vectorization_queue
from vectorize.processors import get_processor_for_file
from db.vector_store import VectorStore

def process_document_queue(vector_db_path):
    vector_store = VectorStore(vector_db_path)

    while True:
        try:
            if not vectorization_queue.empty():
                item = vectorization_queue.get()
                file_path = item["path"]
                action = item["action"]

                print(f"Processing {action} for {file_path}")

                # ファイルタイプに合わせたプロセッサを取得
                processor = get_processor_for_file(file_path)

                # ドキュメント処理
                chunks = processor.process_file(file_path)

                # ベクター化して保存
                if action == "update":
                    # 既存エントリを削除
                    vector_store.delete_by_source(file_path)

                vector_store.add_documents(chunks)

                print(f"Completed processing {file_path}")

            # 短い間隔でキューをチェック
            time.sleep(1)
        except Exception as e:
            print(f"Error processing document: {e}")
            time.sleep(5)
```

### 9.2 スケジュールベースの取り込み

定期的にドキュメントソースをスキャンしてベクター化する方法です。

#### 9.2.1 cron/スケジューラ方式

**特徴**:

- Cron ジョブやスケジューラを使用して定期実行
- 変更されたファイルのみを効率的に処理
- リソース使用を予測しやすい

**Docker での実装例**:

```dockerfile
# worker.Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "scripts.scheduled_vectorization"]
```

**docker-compose.yml への追加**:

```yaml
services:
  # MCP Server
  mcp-server:
    # ... 既存の設定 ...

  # ベクター化ワーカー
  vectorization-worker:
    build:
      context: ..
      dockerfile: docker/worker.Dockerfile
    volumes:
      - ../data:/app/data
      - ../models:/app/models
      - ../vector_store:/app/vector_store
    environment:
      - SCAN_INTERVAL=3600 # 1時間ごとにスキャン
      - DOCUMENT_DIRS=/app/data/documents,/app/data/external
      - EMBEDDING_MODEL_PATH=/app/models/plamo-embedding-1b
      - VECTOR_DB_PATH=/app/vector_store/vector_store.db
    restart: always
```

### 9.3 ハイブリッドアプローチ

最も効果的なのは、上記のアプローチを組み合わせる方法です：

1. **ファイル監視**: ローカルファイルの変更を即座に検知
2. **定期スキャン**: 監視漏れを防ぐためのフルスキャン
3. **手動トリガー**: 必要に応じて特定ドキュメントを手動で再処理

### 9.4 変更検出の最適化

大量のドキュメントを効率的に処理するための工夫：

#### 9.4.1 差分管理戦略

ファイルの変更を効率的に検出するために、以下の方法を組み合わせて実装できます：

**1. メタデータテーブルによる管理**

```python
# vector_store.py 内に差分管理機能を追加

class VectorStore:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        # ... 既存のDB初期化コード ...

        # ファイルメタデータテーブルの作成
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS file_metadata (
            file_path TEXT PRIMARY KEY,
            last_modified INTEGER,
            file_hash TEXT,
            last_vectorized INTEGER,
            status TEXT
        )
        ''')

    def get_files_to_process(self, document_dirs):
        """処理が必要なファイルのリストを取得"""
        files_to_process = []

        for doc_dir in document_dirs:
            for root, _, files in os.walk(doc_dir):
                for file_name in files:
                    if not self._is_target_file(file_name):
                        continue

                    file_path = os.path.join(root, file_name)
                    if self._needs_processing(file_path):
                        files_to_process.append(file_path)

        return files_to_process

    def _needs_processing(self, file_path):
        """ファイルが処理が必要かどうかを判断"""
        try:
            # ファイルの最終更新時刻とハッシュを取得
            mtime = os.path.getmtime(file_path)
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            # DBから最後の処理情報を取得
            cursor = self.conn.execute(
                "SELECT last_modified, file_hash FROM file_metadata WHERE file_path = ?",
                (file_path,)
            )
            row = cursor.fetchone()

            if row is None:
                # 初めて見るファイル
                return True

            last_mtime, last_hash = row

            # 更新日時またはハッシュが変わっていればTrue
            return mtime > last_mtime or file_hash != last_hash

        except Exception as e:
            print(f"Error checking file {file_path}: {e}")
            # エラーの場合は処理が必要と判断
            return True

    def update_file_metadata(self, file_path, status="processed"):
        """処理後にファイルのメタデータを更新"""
        mtime = os.path.getmtime(file_path)
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        now = int(time.time())

        self.conn.execute(
            """
            INSERT OR REPLACE INTO file_metadata
            (file_path, last_modified, file_hash, last_vectorized, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (file_path, mtime, file_hash, now, status)
        )
        self.conn.commit()
```

**2. 差分更新の処理フロー**

```python
# scheduled_vectorization.py

import os
import time
from vectorize.processors import get_processor_for_file
from db.vector_store import VectorStore

def run_scheduled_vectorization(document_dirs, vector_db_path):
    """
    定期的なベクター化を実行
    document_dirs: 監視対象のディレクトリリスト
    vector_db_path: ベクターDBのパス
    """
    print(f"Starting scheduled vectorization at {time.ctime()}")

    vector_store = VectorStore(vector_db_path)

    # 処理が必要なファイルを取得
    files_to_process = vector_store.get_files_to_process(document_dirs)
    print(f"Found {len(files_to_process)} files to process")

    # ファイルごとに処理
    for file_path in files_to_process:
        try:
            print(f"Processing file: {file_path}")

            # ファイルタイプに合わせたプロセッサを取得
            processor = get_processor_for_file(file_path)

            # ドキュメント処理
            chunks = processor.process_file(file_path)

            # 既存データの削除（同一ソースからのデータ）
            vector_store.delete_by_source(file_path)

            # 新しいチャンクを追加
            vector_store.add_documents(chunks)

            # メタデータを更新
            vector_store.update_file_metadata(file_path)

            print(f"Successfully processed: {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            # エラーステータスを記録
            vector_store.update_file_metadata(file_path, status="error")

    print(f"Completed scheduled vectorization at {time.ctime()}")
```

**3. 削除されたファイルの検出と処理**

```python
def cleanup_deleted_files(vector_store):
    """DBに存在するが実際には削除されたファイルの処理"""
    # DBから全ファイルパスを取得
    cursor = vector_store.conn.execute("SELECT file_path FROM file_metadata")
    all_db_files = [row[0] for row in cursor.fetchall()]

    # 存在しないファイルをチェック
    for file_path in all_db_files:
        if not os.path.exists(file_path):
            print(f"File no longer exists: {file_path}")
            # DBからデータを削除
            vector_store.delete_by_source(file_path)
            # メタデータからも削除
            vector_store.conn.execute("DELETE FROM file_metadata WHERE file_path = ?", (file_path,))

    vector_store.conn.commit()
```

#### 9.4.2 その他の最適化

1. **ハッシュベースの変更検出**: 上記のように、ファイルの MD5 ハッシュを計算・保存して内容の変更を正確に検出
2. **メタデータ更新日時の利用**: ファイルシステムの更新日時をファーストチェックとして利用（ハッシュ計算より軽量）
3. **部分更新**: 特に大きなドキュメントの場合、変更された部分のみを再ベクター化する機能も検討可能
4. **バッチ処理**: 複数のドキュメントをまとめて処理する際の並列化とリソース管理

#### 9.4.3 差分管理プロセスの図式化

```
┌───────────────┐           ┌──────────────────┐
│  ファイル変更  │──┐       │   メタデータテーブル  │
└───────────────┘  │       └──────────────────┘
                  │                │
┌───────────────┐  │  ┌──────────────────────────────┐
│ スケジューラ起動 │──┼─▶│ ファイルの変更チェック       │
└───────────────┘  │  │ 1. 新規ファイルか？          │
                  │  │ 2. 更新日時が変わったか？      │
┌───────────────┐  │  │ 3. ハッシュ値が変わったか？   │
│ 監視システム検知 │──┘  └──────────────────────────────┘
└───────────────┘                  │
                                  │ 変更あり
                                  ▼
                     ┌──────────────────────────────┐
                     │ ベクター化処理                │
                     │ 1. 既存データ削除            │
                     │ 2. 新規チャンク作成          │
                     │ 3. ベクター埋め込み          │
                     │ 4. DB保存                   │
                     └──────────────────────────────┘
                                  │
                                  ▼
                     ┌──────────────────────────────┐
                     │ メタデータ更新                │
                     │ 1. 最終更新日時更新           │
                     │ 2. ハッシュ値更新            │
                     │ 3. 処理日時記録              │
                     └──────────────────────────────┘
```

この差分管理アプローチにより、既に処理済みのファイルを再度処理することなく、変更のあったファイルのみを効率的に処理できます。
