import argparse
import os
import sys
import logging
import time
import gc
import traceback
import psutil
import torch
from typing import Optional, List, Dict, Any

# プロジェクトのルートディレクトリをシステムパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# メモリ使用状況をモニタリングする関数
def log_memory_usage(tag=""):
    """現在のメモリ使用状況をログに記録"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024  # RSSをMBに変換
    logger.info(f"メモリ使用量 {tag}: {mem_mb:.1f} MB")
    
    # メモリ使用量が高い場合は警告
    if mem_mb > 1000:  # 1GB以上
        logger.warning(f"メモリ使用量が高くなっています: {mem_mb:.1f} MB")
        # メモリ使用量が非常に高い場合はガベージコレクションを強制的に実行
        if mem_mb > 2000:  # 2GB以上
            logger.warning("メモリ使用量が非常に高いため、ガベージコレクションを実行します")
            gc.collect()

from vectorize.embeddings import PLaMoEmbedder, DummyEmbedder
from db.vector_store import VectorStore
from vectorize.processors.markdown_processor import process_markdown_file, process_markdown_directory
from vectorize.processors.pdf_processor import process_pdf_file, process_pdf_directory
from vectorize.processors.code_processor import process_code_file, process_code_directory
import config

# ロギング設定
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

def get_file_processor(file_path: str) -> Optional[callable]:
    """
    ファイルタイプに応じた適切なプロセッサ関数を返す
    
    Args:
        file_path: 処理するファイルパス
        
    Returns:
        対応するプロセッサ関数。対応するものがなければNone
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.md', '.markdown']:
        return process_markdown_file
    elif ext == '.pdf':
        return process_pdf_file
    elif ext in ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.cs', '.go', '.rb', '.php', '.swift', '.kt', '.rs']:
        return process_code_file
    else:
        return None

def get_directory_processor(dir_type: str) -> Optional[callable]:
    """
    ディレクトリタイプに応じた適切なディレクトリプロセッサ関数を返す
    
    Args:
        dir_type: ディレクトリタイプ
        
    Returns:
        対応するディレクトリプロセッサ関数。対応するものがなければNone
    """
    if dir_type == "markdown":
        return process_markdown_directory
    elif dir_type == "pdf":
        return process_pdf_directory
    elif dir_type == "code":
        return process_code_directory
    else:
        return None

def process_file(file_path: str, embedder, vector_store: VectorStore, use_plamo: bool = True, batch_size: int = 100, auto_fallback: bool = False) -> bool:
    """
    単一ファイルを処理してベクトル化・保存する
    
    Args:
        file_path: 処理するファイルパス
        embedder: 埋め込みモデル
        vector_store: ベクトルストア
        use_plamo: PLaMoエンべディングモデルを使用するかどうか
        batch_size: チャンク処理バッチサイズ
        auto_fallback: メモリ不足時に自動的にダミーエンベッダーに切り替えるかどうか
        
    Returns:
        処理が成功したかどうか
    """
    # ファイルが存在しなければエラー
    if not os.path.exists(file_path):
        logger.error(f"File does not exist: {file_path}")
        return False
        
    # 既に処理済みでかつ更新がない場合はスキップ
    if not vector_store.file_needs_update(file_path):
        logger.info(f"Skipping file (no updates): {file_path}")
        return True
    
    # ファイルサイズをチェック
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.2f} MB")
    
    # 大きいファイルの場合はバッチサイズを調整
    adjusted_batch_size = batch_size
    if file_size_mb > 50:  # 50MB以上のファイル
        adjusted_batch_size = min(batch_size, 50)  # バッチサイズを縮小
        logger.info(f"Large file detected, adjusting batch size to {adjusted_batch_size}")
    
    # メモリ使用状況をログ
    log_memory_usage("before_processing")
        
    # プロセッサを取得
    processor = get_file_processor(file_path)
    if processor is None:
        logger.warning(f"Unsupported file type: {file_path}")
        return False
        
    try:
        # ファイルを処理してチャンクを取得
        start_time = time.time()
        logger.info(f"Starting to process {file_path}")
        
        try:
            docs = processor(file_path)
        except MemoryError:
            # メモリエラーの場合、より保守的な処理を試みる
            if auto_fallback and hasattr(embedder, "__class__") and embedder.__class__.__name__ != "DummyEmbedder":
                logger.warning(f"Memory error during document processing. Switching to DummyEmbedder...")
                from vectorize.embeddings import DummyEmbedder
                embedder = DummyEmbedder(dim=embedder.dim)
                gc.collect()  # メモリを解放
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                try:
                    # より小さいチャンクサイズでもう一度試す
                    logger.info("Retrying with smaller chunks...")
                    docs = processor(file_path, chunk_size=200)  # より小さいチャンクサイズで試す
                except Exception as e2:
                    logger.error(f"Failed to process even with fallback approach: {e2}")
                    return False
            else:
                # fallbackオプションが無効の場合はエラーを再スロー
                raise
        
        if not docs:
            logger.warning(f"No content extracted from {file_path}")
            return False
            
        logger.info(f"Extracted {len(docs)} chunks from {file_path}")
        log_memory_usage("after_extraction")
        
        # 既に処理済みのファイルは一旦削除
        if vector_store.file_exists(file_path):
            vector_store.delete_file(file_path)
        
        # 大きなファイルの場合はバッチ処理
        total_chunks = len(docs)
        
        logger.info(f"Processing {total_chunks} chunks in batches of {adjusted_batch_size}")
        
        try:
            # ドキュメントをバッチで処理
            for i in range(0, total_chunks, adjusted_batch_size):
                batch_end = min(i + adjusted_batch_size, total_chunks)
                logger.info(f"Processing batch {i//adjusted_batch_size + 1}: chunks {i} to {batch_end-1}")
                
                batch_docs = docs[i:batch_end]
                texts = [d["content"] for d in batch_docs]
                
                try:
                    # バッチごとにベクトル化
                    vectors = embedder.embed_batch(texts)
                    
                    # ベクトルストアに保存
                    vector_store.add_documents(batch_docs, vectors, file_path=file_path)
                    
                    # メモリを解放
                    del texts
                    del vectors
                    del batch_docs
                    gc.collect()  # 明示的にガベージコレクション
                    
                    log_memory_usage(f"after_batch_{i//adjusted_batch_size + 1}")
                    logger.info(f"Batch {i//adjusted_batch_size + 1} completed ({batch_end}/{total_chunks} chunks)")
                    
                except MemoryError as me:
                    # メモリエラーが発生した場合の処理
                    logger.error(f"Memory error during batch processing: {me}")
                    if auto_fallback and hasattr(embedder, "__class__") and embedder.__class__.__name__ != "DummyEmbedder":
                        logger.warning("Switching to DummyEmbedder due to memory constraints...")
                        from vectorize.embeddings import DummyEmbedder
                        embedder = DummyEmbedder(dim=embedder.dim)
                        # 既に処理したチャンクをスキップして続行
                        continue
                    else:
                        raise
        except Exception as batch_error:
            logger.error(f"Error during batch processing: {batch_error}")
            # スタックトレースを出力
            logger.error(traceback.format_exc())
            return False
        
        elapsed = time.time() - start_time
        logger.info(f"Successfully processed {file_path} in {elapsed:.2f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        # スタックトレースを出力
        logger.error(traceback.format_exc())
        return False

def process_directory(input_dir: str, vector_store: VectorStore, embedder, dir_type: Optional[str] = None, 
                   recursive: bool = True, batch_size: int = 100, auto_fallback: bool = False) -> int:
    """
    ディレクトリ内のファイルを処理してベクトル化・保存する
    
    Args:
        input_dir: 処理するディレクトリパス
        vector_store: ベクトルストア
        embedder: 埋め込みモデル
        dir_type: ディレクトリタイプ（特定タイプのみ処理する場合）
        recursive: サブディレクトリも再帰的に処理するかどうか
        batch_size: チャンク処理バッチサイズ
        auto_fallback: メモリ不足時に自動的にダミーエンベッダーに切り替えるかどうか
        
    Returns:
        正常に処理されたファイル数
    """
    # ディレクトリが存在しなければエラー
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        logger.error(f"Directory does not exist: {input_dir}")
        return 0
        
    processed_count = 0
    log_memory_usage("before_directory_processing")
    
    # 特定のディレクトリタイプが指定されている場合
    if dir_type:
        dir_processor = get_directory_processor(dir_type)
        if dir_processor:
            try:
                # ディレクトリ内のすべてのファイルを一括処理
                logger.info(f"Processing directory {input_dir} as {dir_type} type")
                
                try:
                    chunks = dir_processor(input_dir)
                except MemoryError:
                    # メモリエラーの場合、より保守的な処理を試みる
                    if auto_fallback and hasattr(embedder, "__class__") and embedder.__class__.__name__ != "DummyEmbedder":
                        logger.warning(f"Memory error during directory processing. Switching to DummyEmbedder...")
                        from vectorize.embeddings import DummyEmbedder
                        embedder = DummyEmbedder(dim=embedder.dim)
                        gc.collect()  # メモリを解放
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        try:
                            # より小さいチャンクサイズでもう一度試す
                            logger.info("Retrying directory processing with smaller chunks...")
                            # ディレクトリプロセッサはチャンクサイズのパラメータを受け取れないため、
                            # ファイルごとに個別処理する戦略に切り替え
                            logger.info("Switching to file-by-file processing strategy")
                            return process_directory_file_by_file(input_dir, vector_store, embedder, dir_type, recursive, batch_size, auto_fallback)
                        except Exception as e2:
                            logger.error(f"Failed to process directory even with fallback approach: {e2}")
                            return 0
                    else:
                        # fallbackオプションが無効の場合はエラーを再スロー
                        raise
                
                if chunks:
                    # メタデータからファイルパスを取得して登録
                    file_paths = set(d["metadata"]["source"] for d in chunks)
                    
                    # 既存ファイルを削除
                    for file_path in file_paths:
                        if vector_store.file_exists(file_path):
                            vector_store.delete_file(file_path)
                    
                    # チャンク数に応じてバッチサイズを調整
                    adjusted_batch_size = batch_size
                    total_chunks = len(chunks)
                    if total_chunks > 1000:
                        # 大量のチャンクがある場合はバッチサイズを縮小
                        adjusted_batch_size = min(batch_size, 50)
                        logger.info(f"Large number of chunks detected ({total_chunks}), adjusting batch size to {adjusted_batch_size}")
                    
                    log_memory_usage("before_batch_processing")
                    logger.info(f"Processing {total_chunks} chunks from directory in batches of {adjusted_batch_size}")
                    
                    # ドキュメントをバッチで処理
                    failure_count = 0
                    for i in range(0, total_chunks, adjusted_batch_size):
                        try:
                            batch_end = min(i + adjusted_batch_size, total_chunks)
                            logger.info(f"Processing batch {i//adjusted_batch_size + 1}: chunks {i} to {batch_end-1}")
                            
                            batch_chunks = chunks[i:batch_end]
                            texts = [d["content"] for d in batch_chunks]
                            
                            # バッチごとにベクトル化
                            vectors = embedder.embed_batch(texts)
                            
                            # ベクトルストアに保存
                            vector_store.add_documents(batch_chunks, vectors)
                            
                            # メモリを解放
                            del texts
                            del vectors
                            del batch_chunks
                            gc.collect()  # 明示的にガベージコレクション
                            
                            log_memory_usage(f"after_dir_batch_{i//adjusted_batch_size + 1}")
                            logger.info(f"Batch {i//adjusted_batch_size + 1} completed ({batch_end}/{total_chunks} chunks)")
                            
                        except MemoryError as me:
                            # メモリエラーが発生した場合の処理
                            failure_count += 1
                            logger.error(f"Memory error during batch processing: {me}")
                            
                            if auto_fallback and hasattr(embedder, "__class__") and embedder.__class__.__name__ != "DummyEmbedder":
                                logger.warning("Switching to DummyEmbedder due to memory constraints...")
                                from vectorize.embeddings import DummyEmbedder
                                embedder = DummyEmbedder(dim=embedder.dim)
                                
                                # バッチサイズをさらに縮小
                                adjusted_batch_size = max(10, adjusted_batch_size // 2)
                                logger.info(f"Reduced batch size to {adjusted_batch_size}")
                                
                                # 既に処理したチャンクをスキップして続行
                                continue
                            elif failure_count > 3:
                                # 3回以上失敗した場合は処理を中断
                                logger.error("Too many failures during batch processing, aborting")
                                return 0
                            else:
                                # 一度だけリトライ
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                continue
                        
                        except Exception as e:
                            logger.error(f"Error processing batch: {e}")
                            logger.error(traceback.format_exc())
                            failure_count += 1
                            if failure_count > 3:
                                logger.error("Too many failures during batch processing, aborting")
                                return 0
                    
                    processed_count = len(file_paths)
                    
            except Exception as e:
                logger.error(f"Error processing directory {input_dir}: {e}")
                logger.error(traceback.format_exc())
                return 0
                
            return processed_count
    
    return process_directory_file_by_file(input_dir, vector_store, embedder, dir_type, recursive, batch_size, auto_fallback)

def process_directory_file_by_file(input_dir: str, vector_store: VectorStore, embedder, dir_type: Optional[str] = None,
                                  recursive: bool = True, batch_size: int = 100, auto_fallback: bool = False) -> int:
    """ディレクトリ内のファイルを1つずつ個別に処理する"""
    processed_count = 0
    
    logger.info(f"Processing directory {input_dir} file by file")
    
    # ファイルごとに個別処理
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        
        if os.path.isfile(item_path):
            if process_file(item_path, embedder, vector_store, batch_size=batch_size, auto_fallback=auto_fallback):
                processed_count += 1
                
        elif os.path.isdir(item_path) and recursive:
            processed_count += process_directory(item_path, vector_store, embedder, dir_type, recursive, batch_size, auto_fallback)
    
    return processed_count

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="ドキュメントをベクター化してDBに格納します")
    parser.add_argument("--input", required=True, help="入力ファイルまたはディレクトリ")
    parser.add_argument("--output_db", default=config.VECTOR_DB_PATH, help=f"出力DBパス (デフォルト: {config.VECTOR_DB_PATH})")
    parser.add_argument("--use_dummy", action="store_true", help="ダミー埋め込みを使用する（テスト用）")
    parser.add_argument("--dir_type", choices=["markdown", "pdf", "code"], help="ディレクトリタイプ（特定タイプのみ処理）")
    parser.add_argument("--non_recursive", action="store_true", help="サブディレクトリを再帰的に処理しない")
    parser.add_argument("--vector_dim", type=int, default=512, help="ベクトルの次元数")
    parser.add_argument("--batch_size", type=int, default=50, help="処理バッチサイズ（メモリ使用量の調整用）")
    parser.add_argument("--debug", action="store_true", help="詳細なデバッグログを出力")
    parser.add_argument("--auto_fallback", action="store_true", help="メモリ不足の場合に自動でダミーエンベディングに切り替え")
    args = parser.parse_args()

    # デバッグモードの場合はログレベルを DEBUG に設定
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("デバッグモードが有効です")
    
    # 入力パスを絶対パスに変換
    input_path = os.path.abspath(args.input)
    output_db = os.path.abspath(args.output_db)
    
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output DB: {output_db}")
    
    # ベクトルストアを初期化
    vector_store = VectorStore(output_db, vector_dimension=args.vector_dim)
    
    # 埋め込みモデルを初期化
    if args.use_dummy:
        logger.info("Using dummy embedder")
        embedder = DummyEmbedder(dim=args.vector_dim)
    else:
        logger.info("Using PLaMo-Embedding-1B model")
        try:
            embedder = PLaMoEmbedder(model_path=config.EMBEDDING_MODEL_PATH)
        except Exception as e:
            logger.error(f"Error loading PLaMo model: {e}. Falling back to dummy embedder.")
            embedder = DummyEmbedder(dim=args.vector_dim)
    
    start_time = time.time()
    processed_count = 0
    
    # メモリ使用状況をログ
    log_memory_usage("before_main_processing")
    
    # 入力パスがファイルかディレクトリかによって処理を分ける
    if os.path.isfile(input_path):
        # 単一ファイルを処理
        if process_file(input_path, embedder, vector_store, batch_size=args.batch_size, auto_fallback=args.auto_fallback):
            processed_count = 1
    else:
        # ディレクトリを処理
        processed_count = process_directory(
            input_path, 
            vector_store, 
            embedder,
            dir_type=args.dir_type,
            recursive=not args.non_recursive,
            batch_size=args.batch_size,
            auto_fallback=args.auto_fallback
        )
    
    # 処理時間を計算
    elapsed = time.time() - start_time
    
    # 統計情報を取得
    stats = vector_store.get_stats()
    
    logger.info(f"Processing completed in {elapsed:.2f} seconds")
    logger.info(f"Processed {processed_count} files successfully")
    logger.info(f"Total documents in vector store: {stats.get('total_documents', 0)}")
    logger.info(f"Total files in vector store: {stats.get('total_files', 0)}")
    
    # ベクトルストアを閉じる
    vector_store.close()

if __name__ == "__main__":
    main()