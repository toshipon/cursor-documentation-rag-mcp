import os
import logging
from typing import List, Dict, Any
from vectorize.text_splitters import MarkdownTextSplitter
import config

# ロギング設定
logger = logging.getLogger(__name__)

def process_markdown_file(file_path: str, chunk_size: int = 400, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    マークダウンファイルを処理して、チャンクとメタデータを抽出する
    
    Args:
        file_path: 処理するマークダウンファイルのパス
        chunk_size: 分割するチャンクのサイズ
        chunk_overlap: チャンク間のオーバーラップ文字数
        
    Returns:
        分割されたテキストとメタデータを含む辞書のリスト
    """
    logger.info(f"Processing markdown file: {file_path}")
    
    try:
        # ファイルを読み込む
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        # ファイル名とパス情報を取得
        file_name = os.path.basename(file_path)
        relative_path = os.path.relpath(file_path, start=config.BASE_DIR)
        
        # 基本メタデータを作成
        metadata = {
            "source": file_path,
            "relative_path": relative_path,
            "file_name": file_name,
            "source_type": "markdown",
            "file_extension": os.path.splitext(file_name)[1],
        }
        
        # マークダウン分割クラスをインスタンス化
        splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # テキストを分割し、メタデータを付与
        chunks = splitter.split_with_metadata(text, metadata)
        
        logger.info(f"Successfully processed {file_path} into {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Error processing markdown file {file_path}: {e}")
        return []
        
def process_markdown_directory(directory: str, chunk_size: int = 400, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    ディレクトリ内のすべてのマークダウンファイルを処理する
    
    Args:
        directory: 処理するディレクトリのパス
        chunk_size: 分割するチャンクのサイズ
        chunk_overlap: チャンク間のオーバーラップ文字数
        
    Returns:
        すべてのファイルの分割されたテキストとメタデータを含む辞書のリスト
    """
    logger.info(f"Processing markdown files in directory: {directory}")
    
    all_chunks = []
    
    # ディレクトリがなければエラー
    if not os.path.exists(directory):
        logger.error(f"Directory does not exist: {directory}")
        return all_chunks
        
    # 再帰的にすべてのマークダウンファイルを処理
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.md', '.markdown')):
                file_path = os.path.join(root, file)
                chunks = process_markdown_file(file_path, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
                
    logger.info(f"Processed {len(all_chunks)} total chunks from directory: {directory}")
    return all_chunks