import argparse
import os
import sys
import logging
import time
from typing import Optional, List, Dict, Any

# プロジェクトのルートディレクトリをシステムパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def process_file(file_path: str, embedder, vector_store: VectorStore, use_plamo: bool = True) -> bool:
    """
    単一ファイルを処理してベクトル化・保存する
    
    Args:
        file_path: 処理するファイルパス
        embedder: 埋め込みモデル
        vector_store: ベクトルストア
        use_plamo: PLaMoエンべディングモデルを使用するかどうか
        
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
        
    # プロセッサを取得
    processor = get_file_processor(file_path)
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

def process_directory(input_dir: str, vector_store: VectorStore, embedder, dir_type: Optional[str] = None, recursive: bool = True) -> int:
    """
    ディレクトリ内のファイルを処理してベクトル化・保存する
    
    Args:
        input_dir: 処理するディレクトリパス
        vector_store: ベクトルストア
        embedder: 埋め込みモデル
        dir_type: ディレクトリタイプ（特定タイプのみ処理する場合）
        recursive: サブディレクトリも再帰的に処理するかどうか
        
    Returns:
        正常に処理されたファイル数
    """
    # ディレクトリが存在しなければエラー
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        logger.error(f"Directory does not exist: {input_dir}")
        return 0
        
    processed_count = 0
    
    # 特定のディレクトリタイプが指定されている場合
    if dir_type:
        dir_processor = get_directory_processor(dir_type)
        if dir_processor:
            try:
                # ディレクトリ内のすべてのファイルを一括処理
                chunks = dir_processor(input_dir)
                
                if chunks:
                    # テキストをベクトル化
                    texts = [d["content"] for d in chunks]
                    vectors = embedder.embed_batch(texts)
                    
                    # メタデータからファイルパスを取得して登録
                    file_paths = set(d["metadata"]["source"] for d in chunks)
                    
                    # 既存ファイルを削除
                    for file_path in file_paths:
                        if vector_store.file_exists(file_path):
                            vector_store.delete_file(file_path)
                    
                    # ベクトルストアに保存
                    vector_store.add_documents(chunks, vectors)
                    processed_count = len(file_paths)
                    
            except Exception as e:
                logger.error(f"Error processing directory {input_dir}: {e}")
                return 0
                
            return processed_count
    
    # ファイルごとに個別処理
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        
        if os.path.isfile(item_path):
            if process_file(item_path, embedder, vector_store):
                processed_count += 1
                
        elif os.path.isdir(item_path) and recursive:
            processed_count += process_directory(item_path, vector_store, embedder, dir_type, recursive)
            
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
    args = parser.parse_args()

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
    
    # 入力パスがファイルかディレクトリかによって処理を分ける
    if os.path.isfile(input_path):
        # 単一ファイルを処理
        if process_file(input_path, embedder, vector_store):
            processed_count = 1
    else:
        # ディレクトリを処理
        processed_count = process_directory(
            input_path, 
            vector_store, 
            embedder,
            dir_type=args.dir_type,
            recursive=not args.non_recursive
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