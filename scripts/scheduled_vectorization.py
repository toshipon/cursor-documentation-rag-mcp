import os
import sys
import argparse
import logging
import time
import datetime
import signal
from typing import List, Dict, Any, Set

# プロジェクトのルートディレクトリをシステムパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from db.vector_store import VectorStore
from vectorize.embeddings import PLaMoEmbedder, DummyEmbedder
from vectorize.processors.markdown_processor import process_markdown_file
from vectorize.processors.pdf_processor import process_pdf_file
from vectorize.processors.code_processor import process_code_file

# ロギング設定
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# ファイル拡張子とプロセッサのマッピング
FILE_PROCESSORS = {
    '.md': process_markdown_file,
    '.markdown': process_markdown_file,
    '.pdf': process_pdf_file,
    '.py': process_code_file,
    '.js': process_code_file,
    '.ts': process_code_file,
    '.jsx': process_code_file,
    '.tsx': process_code_file,
    '.java': process_code_file,
    '.c': process_code_file,
    '.cpp': process_code_file,
    '.cs': process_code_file,
    '.go': process_code_file,
    '.rb': process_code_file,
    '.php': process_code_file,
    '.swift': process_code_file,
    '.kt': process_code_file,
    '.rs': process_code_file
}

def scan_directory(directory: str, 
                  ignored_patterns: List[str] = None,
                  recursive: bool = True) -> List[str]:
    """
    ディレクトリをスキャンしてサポートされているファイルを見つける
    
    Args:
        directory: スキャンするディレクトリパス
        ignored_patterns: 無視するパターンのリスト
        recursive: サブディレクトリも再帰的にスキャンするかどうか
        
    Returns:
        ファイルパスのリスト
    """
    if ignored_patterns is None:
        ignored_patterns = []
        
    found_files = []
    
    # ディレクトリが存在しなければエラー
    if not os.path.exists(directory) or not os.path.isdir(directory):
        logger.error(f"Directory does not exist: {directory}")
        return found_files
    
    # ディレクトリをスキャン
    for root, dirs, files in os.walk(directory):
        # 無視するディレクトリをフィルタリング
        dirs_to_remove = []
        for d in dirs:
            if d.startswith('.') or d in ['__pycache__', '.git', 'venv', 'env', '.venv', '.env']:
                dirs_to_remove.append(d)
                
        for d in dirs_to_remove:
            dirs.remove(d)
            
        # 非再帰的モードの場合はサブディレクトリをクリア
        if not recursive:
            dirs.clear()
        
        # 各ファイルを処理
        for filename in files:
            file_path = os.path.join(root, filename)
            
            # 隠しファイルを無視
            if filename.startswith('.'):
                continue
                
            # 無視パターンをチェック
            skip_file = False
            for pattern in ignored_patterns:
                if pattern in file_path:
                    skip_file = True
                    break
            
            if skip_file:
                continue
                
            # 拡張子をチェック
            ext = os.path.splitext(filename)[1].lower()
            if ext in FILE_PROCESSORS:
                found_files.append(file_path)
                
    return found_files

def process_file(file_path: str, 
                vector_store: VectorStore, 
                embedder,
                force_update: bool = False) -> bool:
    """
    単一ファイルを処理してベクトル化
    
    Args:
        file_path: ファイルパス
        vector_store: ベクトルストア
        embedder: 埋め込みモデル
        force_update: 強制的に更新するかどうか
        
    Returns:
        処理が成功したかどうか
    """
    # ファイルが存在しなければエラー
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return False
    
    # 強制更新でなければ、既に処理済みでかつ更新がないときはスキップ
    if not force_update and not vector_store.file_needs_update(file_path):
        logger.info(f"Skipping file (no updates): {file_path}")
        return True
    
    # プロセッサを取得
    ext = os.path.splitext(file_path)[1].lower()
    processor = FILE_PROCESSORS.get(ext)
    
    if processor is None:
        logger.warning(f"Unsupported file type: {file_path}")
        return False
    
    try:
        # ファイルを処理してチャンクを取得
        start_time = time.time()
        docs = processor(file_path)
        
        if not docs:
            logger.warning(f"No content extracted from {file_path}")
            return False
        
        logger.info(f"Extracted {len(docs)} chunks from {file_path}")
        
        # 既に処理済みのファイルは一旦削除
        if vector_store.file_exists(file_path):
            vector_store.delete_file(file_path)
        
        # テキストをベクトル化
        texts = [d["content"] for d in docs]
        vectors = embedder.embed_batch(texts)
        
        # ベクトルストアに保存
        vector_store.add_documents(docs, vectors, file_path=file_path)
        
        elapsed = time.time() - start_time
        logger.info(f"Successfully processed {file_path} in {elapsed:.2f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return False

def get_deleted_files(vector_store: VectorStore, scanned_files: List[str]) -> List[str]:
    """
    ベクトルストアにあるがスキャン結果にないファイルを見つける
    
    Args:
        vector_store: ベクトルストア
        scanned_files: スキャンしたファイルのパスリスト
        
    Returns:
        削除されたファイルのパスリスト
    """
    # ベクトルストアからすべてのファイルパスを取得
    try:
        cursor = vector_store.conn.execute("SELECT file_path FROM file_metadata")
        stored_files = set(row[0] for row in cursor.fetchall())
        
        # スキャン結果をセットに変換
        scanned_files_set = set(scanned_files)
        
        # 差分を計算
        deleted_files = stored_files - scanned_files_set
        
        return list(deleted_files)
        
    except Exception as e:
        logger.error(f"Error getting deleted files: {e}")
        return []

def delete_files(vector_store: VectorStore, file_paths: List[str]) -> int:
    """
    ベクトルストアから複数のファイルを削除
    
    Args:
        vector_store: ベクトルストア
        file_paths: 削除するファイルパスのリスト
        
    Returns:
        正常に削除されたファイル数
    """
    deleted_count = 0
    
    for file_path in file_paths:
        try:
            logger.info(f"Deleting file from vector store: {file_path}")
            docs_deleted = vector_store.delete_file(file_path)
            
            if docs_deleted > 0:
                deleted_count += 1
                logger.info(f"Deleted {docs_deleted} documents for {file_path}")
                
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            
    return deleted_count

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="定期的にドキュメントをスキャンしてベクトル化します")
    parser.add_argument("--scan_dirs", nargs='+', required=True, help="スキャンするディレクトリ（複数指定可）")
    parser.add_argument("--vector_db", default=config.VECTOR_DB_PATH, help=f"ベクトルDBのパス (デフォルト: {config.VECTOR_DB_PATH})")
    parser.add_argument("--use_dummy", action="store_true", help="ダミー埋め込みモデルを使用（テスト用）")
    parser.add_argument("--ignored_patterns", nargs='+', default=[], help="無視するファイルパターン（複数指定可）")
    parser.add_argument("--non_recursive", action="store_true", help="サブディレクトリを再帰的にスキャンしない")
    parser.add_argument("--force_update", action="store_true", help="すべてのファイルを強制的に更新する")
    parser.add_argument("--interval", type=int, default=3600, help="スキャン間隔（秒単位、デフォルトは1時間）")
    parser.add_argument("--run_once", action="store_true", help="一度だけ実行して終了する")
    args = parser.parse_args()
    
    # スキャン対象ディレクトリを絶対パスに変換
    scan_dirs = [os.path.abspath(d) for d in args.scan_dirs]
    
    # 各ディレクトリの存在チェック
    for directory in scan_dirs:
        if not os.path.exists(directory):
            logger.error(f"Directory does not exist: {directory}")
            return 1
    
    # ベクトルDBのディレクトリが存在しなければ作成
    os.makedirs(os.path.dirname(args.vector_db), exist_ok=True)
    
    logger.info(f"Scanning directories: {', '.join(scan_dirs)}")
    logger.info(f"Vector DB path: {args.vector_db}")
    logger.info(f"Scan interval: {args.interval} seconds")
    
    try:
        # ベクトルストアを初期化
        vector_store = VectorStore(args.vector_db)
        
        # 埋め込みモデルを初期化
        if args.use_dummy:
            logger.info("Using dummy embedder")
            embedder = DummyEmbedder()
        else:
            logger.info("Initializing PLaMo-Embedding-1B model")
            try:
                embedder = PLaMoEmbedder(model_path=config.EMBEDDING_MODEL_PATH)
            except Exception as e:
                logger.error(f"Error loading PLaMo model: {e}. Falling back to dummy embedder.")
                embedder = DummyEmbedder()
        
        # メインループ
        run_count = 0
        while True:
            start_time = time.time()
            run_count += 1
            
            logger.info(f"Starting scan #{run_count} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 全ファイルをスキャン
            all_files = []
            for directory in scan_dirs:
                scanned_files = scan_directory(
                    directory=directory,
                    ignored_patterns=args.ignored_patterns,
                    recursive=not args.non_recursive
                )
                all_files.extend(scanned_files)
                
            logger.info(f"Found {len(all_files)} files to process")
            
            # 削除されたファイルを検出して削除
            deleted_files = get_deleted_files(vector_store, all_files)
            if deleted_files:
                logger.info(f"Detected {len(deleted_files)} deleted files")
                deleted_count = delete_files(vector_store, deleted_files)
                logger.info(f"Removed {deleted_count} files from vector store")
            
            # 各ファイルを処理
            processed_count = 0
            skipped_count = 0
            
            for file_path in all_files:
                if process_file(file_path, vector_store, embedder, args.force_update):
                    processed_count += 1
                else:
                    skipped_count += 1
            
            # 統計情報をログに出力
            stats = vector_store.get_stats()
            elapsed = time.time() - start_time
            
            logger.info(f"Scan completed in {elapsed:.2f} seconds")
            logger.info(f"Processed: {processed_count}, Skipped: {skipped_count}, Deleted: {len(deleted_files)}")
            logger.info(f"Vector store stats: {stats.get('total_documents', 0)} documents, "
                      f"{stats.get('total_files', 0)} files")
            
            # 一度だけ実行するモードの場合は終了
            if args.run_once:
                break
                
            # 次回の実行まで待機
            next_run = start_time + args.interval
            wait_time = max(0, next_run - time.time())
            
            if wait_time > 0:
                logger.info(f"Next scan scheduled at {datetime.datetime.fromtimestamp(next_run).strftime('%Y-%m-%d %H:%M:%S')}")
                try:
                    time.sleep(wait_time)
                except KeyboardInterrupt:
                    logger.info("Interrupted by user")
                    break
        
        # ベクトルストアを閉じる
        vector_store.close()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        if 'vector_store' in locals():
            vector_store.close()
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
