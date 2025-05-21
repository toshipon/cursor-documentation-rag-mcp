# cursor-documentation-rag-mcp プロジェクト進捗管理

## 1. 開発ロードマップ

### Phase 1: 基盤実装 (1-2 週目)

- [x] プロジェクト構造のセットアップ
  - [x] ディレクトリ構造の作成
  - [x] requirements.txt の作成
  - [x] config.py の実装
  - [x] README.md の作成
- [x] PLaMo-Embedding-1B の統合
  - [x] モデルのダウンロードとテスト
  - [x] 埋め込み処理の実装
  - [x] バッチ処理の最適化
- [x] テキストチャンク処理
  - [x] MarkdownTextSplitter の実装
  - [x] 適切なチャンクサイズの検証
  - [x] メタデータ抽出ロジックの実装
- [x] SQLite-VSS のセットアップ
  - [x] テーブルスキーマの設計
  - [x] ベクトル検索機能の実装
  - [x] CRUD 操作の実装
- [x] 基本的な MCP サーバー実装
  - [x] FastAPI サーバーのセットアップ
  - [x] クエリエンドポイントの実装
  - [x] 検索結果のフォーマット処理

### Phase 2: 拡張機能実装 (3-4 週目)

- [x] ドキュメント処理の拡張
  - [x] PDF プロセッサの実装
  - [x] ソースコードプロセッサの実装
  - [x] 各ファイルタイプのメタデータ拡張
- [ ] 監視ベースの自動取り込み機能
  - [ ] file_watcher.py の実装
  - [ ] vectorization_worker.py の実装
  - [ ] イベントキューの管理
- [ ] スケジュールベースの取り込み機能
  - [ ] scheduled_vectorization.py の実装
  - [ ] 差分管理機能の実装
  - [ ] メタデータテーブルの設計と実装
- [ ] Docker 環境の構築
  - [ ] Dockerfile の作成
  - [ ] docker-compose.yml の作成
  - [ ] entrypoint.sh の作成
  - [ ] マウントボリュームの設定

### Phase 3: 最適化・拡張 (5-6 週目)

- [ ] 検索パフォーマンスの最適化
  - [ ] クエリ処理の高速化
  - [ ] フィルタリング機能の強化
  - [ ] メタデータ検索の実装
- [ ] スケーリング対応
  - [ ] 大規模データ処理のテスト
  - [ ] パフォーマンスボトルネックの特定と解消
  - [ ] リソース使用量の最適化
- [ ] UI/UX 改善
  - [ ] 検索結果表示の改善
  - [ ] メタデータフィルタリング UI の改善
  - [ ] キーワード+ベクトル検索のハイブリッド実装
- [ ] テスト・ドキュメント
  - [ ] ユニットテストの作成
  - [ ] 統合テストの作成
  - [ ] ユーザードキュメントの整備

## 2. 詳細タスクリスト

### 初期セットアップ

#### ディレクトリ構造と基本ファイル

- [x] ディレクトリ構造を作成
  ```bash
  mkdir -p cursor-documentation-rag-mcp/{vectorize/processors,db,mcp,scripts,docker,workers}
  touch cursor-documentation-rag-mcp/{README.md,requirements.txt,config.py}
  touch cursor-documentation-rag-mcp/vectorize/{__init__.py,embeddings.py,text_splitters.py}
  touch cursor-documentation-rag-mcp/vectorize/processors/{__init__.py,markdown_processor.py,pdf_processor.py,code_processor.py}
  touch cursor-documentation-rag-mcp/db/{__init__.py,vector_store.py}
  touch cursor-documentation-rag-mcp/mcp/{__init__.py,server.py}
  touch cursor-documentation-rag-mcp/scripts/{vectorize_docs.py,start_mcp_server.py}
  touch cursor-documentation-rag-mcp/docker/{Dockerfile,docker-compose.yml,entrypoint.sh}
  ```
  - 2025-05-21: 雛形ディレクトリ・主要ファイル作成、requirements.txt・README.md・初期スクリプト実装（vectorize_docs.py, start_mcp_server.py）
  - 2025-05-21: config.py を実装（環境変数・デフォルトパス・主要設定の一元管理）
- [x] requirements.txt に依存関係を記述
  ```
  fastapi==0.103.1
  uvicorn==0.23.2
  langchain==0.0.267
  sqlite-vss==0.1.2
  huggingface-hub==0.16.4
  transformers==4.33.2
  torch==2.0.1
  python-multipart==0.0.6
  watchdog==3.0.0
  pdfplumber==0.9.0
  ```
- [x] config.py で設定を一元管理
  - [x] 環境変数の処理
  - [x] デフォルト設定
  - [x] ファイルパス設定

### ベクトル化コンポーネント

#### PLaMo-Embedding-1B 統合

- [x] embeddings.py を実装
  - [x] モデルロード処理
  - [x] テキスト → ベクトル変換
  - [x] バッチ処理
- [x] モデルダウンロードスクリプト
  - [x] HuggingFace 連携

#### テキスト分割処理

- [x] text_splitters.py を実装
  - [x] 基本分割クラス
  - [x] マークダウン分割クラス
  - [x] コード分割クラス
- [x] 最適なチャンクサイズの実験と設定

#### ドキュメント処理

- [x] markdown_processor.py 実装
  - [x] ファイル読み込み
  - [x] メタデータ抽出
  - [x] チャンク処理
- [x] pdf_processor.py 実装
  - [x] PDF→ テキスト変換
  - [x] レイアウト考慮した分割
  - [x] メタデータ抽出
- [x] code_processor.py 実装
  - [x] 言語検出
  - [x] コメント処理
  - [x] 構造を考慮した分割

### ベクトルデータベース

#### SQLite-VSS 設定

- [x] vector_store.py を実装
  - [x] DB コネクション管理
  - [x] テーブルスキーマ定義
  - [x] ベクトル保存機能
  - [x] 類似検索機能

#### メタデータ管理

- [x] ファイルメタデータテーブル設計
  - [x] スキーマ定義
  - [x] インデックス設定
- [x] 差分管理機能
  - [x] ハッシュベースの変更検出
  - [x] 部分更新機能

### MCP サーバー

#### FastAPI サーバー

- [x] server.py を実装
  - [x] エンドポイント定義
  - [x] リクエスト/レスポンスモデル
  - [x] エラーハンドリング
- [x] クエリ処理
  - [x] テキスト → ベクトル変換
  - [x] 類似度検索
  - [x] フィルタリング処理

### 自動化コンポーネント

#### ファイル監視機能

- [ ] file_watcher.py 実装
  - [ ] watchdog によるファイルシステム監視
  - [ ] イベントハンドラー
  - [ ] キュー管理
- [ ] vectorization_worker.py 実装
  - [ ] キュー処理
  - [ ] ファイル処理ロジック
  - [ ] エラーハンドリング

#### スケジュール実行

- [ ] scheduled_vectorization.py 実装
  - [ ] ディレクトリスキャン
  - [ ] 差分検出
  - [ ] バッチ処理

### Docker 環境

#### Docker 構成

- [ ] Dockerfile 作成
  - [ ] ベースイメージ選定
  - [ ] 依存関係インストール
  - [ ] アプリケーションセットアップ
- [ ] docker-compose.yml 作成
  - [ ] サービス定義
  - [ ] ボリューム設定
  - [ ] 環境変数設定
- [ ] entrypoint.sh スクリプト作成
  - [ ] 初期化処理
  - [ ] モデルダウンロード
  - [ ] サーバー起動

## 3. マイルストーン

1. **基本プロトタイプ (Week 2 終了)**

   - [x] 基本的な Markdown ファイル処理
   - [x] SQLite-VSS による検索
   - [x] シンプルな MCP エンドポイント

2. **拡張プロトタイプ (Week 4 終了)**

   - [ ] 複数ファイルタイプの処理
   - [ ] 自動取り込み機能
   - [ ] Docker 環境

3. **最終リリース (Week 6 終了)**
   - [ ] パフォーマンス最適化
   - [ ] 本番環境デプロイ
   - [ ] ドキュメント完備

### Phase 2: 拡張機能実装 (3-4 週目)

- [ ] ドキュメント処理の拡張
  - [ ] PDF プロセッサの実装
  - [ ] ソースコードプロセッサの実装
  - [ ] 各ファイルタイプのメタデータ拡張
- [ ] 監視ベースの自動取り込み機能
  - [ ] file_watcher.py の実装
  - [ ] vectorization_worker.py の実装
  - [ ] イベントキューの管理
- [ ] スケジュールベースの取り込み機能
  - [ ] scheduled_vectorization.py の実装
  - [ ] 差分管理機能の実装
  - [ ] メタデータテーブルの設計と実装
- [ ] Docker 環境の構築
  - [ ] Dockerfile の作成
  - [ ] docker-compose.yml の作成
  - [ ] entrypoint.sh の作成
  - [ ] マウントボリュームの設定

### Phase 3: 最適化・拡張 (5-6 週目)

- [ ] 検索パフォーマンスの最適化
  - [ ] クエリ処理の高速化
  - [ ] フィルタリング機能の強化
  - [ ] メタデータ検索の実装
- [ ] スケーリング対応
  - [ ] 大規模データ処理のテスト
  - [ ] パフォーマンスボトルネックの特定と解消
  - [ ] リソース使用量の最適化
- [ ] UI/UX 改善
  - [ ] 検索結果表示の改善
  - [ ] メタデータフィルタリング UI の改善
  - [ ] キーワード+ベクトル検索のハイブリッド実装
- [ ] テスト・ドキュメント
  - [ ] ユニットテストの作成
  - [ ] 統合テストの作成
  - [ ] ユーザードキュメントの整備

## 2. 詳細タスクリスト

### 初期セットアップ

#### ディレクトリ構造と基本ファイル

- [ ] ディレクトリ構造を作成
  ```bash
  mkdir -p cursor-documentation-rag-mcp/{vectorize/processors,db,mcp,scripts,docker,workers}
  touch cursor-documentation-rag-mcp/{README.md,requirements.txt,config.py}
  touch cursor-documentation-rag-mcp/vectorize/{__init__.py,embeddings.py,text_splitters.py}
  touch cursor-documentation-rag-mcp/vectorize/processors/{__init__.py,markdown_processor.py,pdf_processor.py,code_processor.py}
  touch cursor-documentation-rag-mcp/db/{__init__.py,vector_store.py}
  touch cursor-documentation-rag-mcp/mcp/{__init__.py,server.py}
  touch cursor-documentation-rag-mcp/scripts/{vectorize_docs.py,start_mcp_server.py}
  touch cursor-documentation-rag-mcp/docker/{Dockerfile,docker-compose.yml,entrypoint.sh}
  ```
  - 2025-05-21: 雛形ディレクトリ・主要ファイル作成、requirements.txt・README.md・初期スクリプト実装（vectorize_docs.py, start_mcp_server.py）
  - 2025-05-21: config.py を実装（環境変数・デフォルトパス・主要設定の一元管理）
- [ ] requirements.txt に依存関係を記述
  ```
  fastapi==0.103.1
  uvicorn==0.23.2
  langchain==0.0.267
  sqlite-vss==0.1.2
  huggingface-hub==0.16.4
  transformers==4.33.2
  torch==2.0.1
  python-multipart==0.0.6
  watchdog==3.0.0
  pdfplumber==0.9.0
  ```
- [ ] config.py で設定を一元管理
  - [ ] 環境変数の処理
  - [ ] デフォルト設定
  - [ ] ファイルパス設定

### ベクトル化コンポーネント

#### PLaMo-Embedding-1B 統合

- [x] embeddings.py を実装
  - [x] モデルロード処理（ダミー）
  - [x] テキスト → ベクトル変換（ダミー）
  - [x] バッチ処理（ダミー）
- [ ] モデルダウンロードスクリプト
  - [ ] HuggingFace 連携

#### テキスト分割処理

- [x] text_splitters.py を実装
  - [x] 基本分割クラス
  - [x] マークダウン分割クラス
  - [x] コード分割クラス
- [ ] 最適なチャンクサイズの実験と設定

#### ドキュメント処理

- [ ] markdown_processor.py 実装
  - [ ] ファイル読み込み
  - [ ] メタデータ抽出
  - [ ] チャンク処理
- [ ] pdf_processor.py 実装
  - [ ] PDF→ テキスト変換
  - [ ] レイアウト考慮した分割
  - [ ] メタデータ抽出
- [ ] code_processor.py 実装
  - [ ] 言語検出
  - [ ] コメント処理
  - [ ] 構造を考慮した分割

### ベクトルデータベース

#### SQLite-VSS 設定

- [ ] vector_store.py を実装
  - [ ] DB コネクション管理
  - [ ] テーブルスキーマ定義
  - [ ] ベクトル保存機能
  - [ ] 類似検索機能

#### メタデータ管理

- [ ] ファイルメタデータテーブル設計
  - [ ] スキーマ定義
  - [ ] インデックス設定
- [ ] 差分管理機能
  - [ ] ハッシュベースの変更検出
  - [ ] 部分更新機能

### MCP サーバー

#### FastAPI サーバー

- [ ] server.py を実装
  - [ ] エンドポイント定義
  - [ ] リクエスト/レスポンスモデル
  - [ ] エラーハンドリング
- [ ] クエリ処理
  - [ ] テキスト → ベクトル変換
  - [ ] 類似度検索
  - [ ] フィルタリング処理

### 自動化コンポーネント

#### ファイル監視機能

- [ ] file_watcher.py 実装
  - [ ] watchdog によるファイルシステム監視
  - [ ] イベントハンドラー
  - [ ] キュー管理
- [ ] vectorization_worker.py 実装
  - [ ] キュー処理
  - [ ] ファイル処理ロジック
  - [ ] エラーハンドリング

#### スケジュール実行

- [ ] scheduled_vectorization.py 実装
  - [ ] ディレクトリスキャン
  - [ ] 差分検出
  - [ ] バッチ処理

### Docker 環境

#### Docker 構成

- [ ] Dockerfile 作成
  - [ ] ベースイメージ選定
  - [ ] 依存関係インストール
  - [ ] アプリケーションセットアップ
- [ ] docker-compose.yml 作成
  - [ ] サービス定義
  - [ ] ボリューム設定
  - [ ] 環境変数設定
- [ ] entrypoint.sh スクリプト作成
  - [ ] 初期化処理
  - [ ] モデルダウンロード
  - [ ] サーバー起動

## 3. マイルストーン

1. **基本プロトタイプ (Week 2 終了)**

   - 基本的な Markdown ファイル処理
   - SQLite-VSS による検索
   - シンプルな MCP エンドポイント

2. **拡張プロトタイプ (Week 4 終了)**

   - 複数ファイルタイプの処理
   - 自動取り込み機能
   - Docker 環境

3. **最終リリース (Week 6 終了)**
   - パフォーマンス最適化
   - 本番環境デプロイ
   - ドキュメント完備
