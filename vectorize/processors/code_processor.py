import os
import logging
from typing import List, Dict, Any
from vectorize.text_splitters import CodeTextSplitter
import config

# ロギング設定
logger = logging.getLogger(__name__)

# 対応するファイル拡張子とプログラミング言語のマッピング
FILE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "javascript",
    ".jsx": "javascript",
    ".tsx": "javascript",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".rs": "rust",
}

def detect_language(file_path: str) -> str:
    """
    ファイル拡張子からプログラミング言語を検出
    
    Args:
        file_path: ソースコードファイルのパス
        
    Returns:
        検出された言語。不明な場合はNone
    """
    ext = os.path.splitext(file_path)[1].lower()
    return FILE_EXTENSIONS.get(ext)

def process_code_file(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    ソースコードファイルを処理して、チャンクとメタデータを抽出する
    
    Args:
        file_path: 処理するソースコードファイルのパス
        chunk_size: 分割するチャンクのサイズ
        chunk_overlap: チャンク間のオーバーラップ文字数
        
    Returns:
        分割されたコードとメタデータを含む辞書のリスト
    """
    logger.info(f"Processing code file: {file_path}")
    
    # 言語を検出
    language = detect_language(file_path)
    
    # 対応言語でなければスキップ
    if not language:
        logger.warning(f"Unsupported file type: {file_path}")
        return []
    
    try:
        # ファイルを読み込む
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
            
        # ファイル名とパス情報を取得
        file_name = os.path.basename(file_path)
        relative_path = os.path.relpath(file_path, start=config.BASE_DIR)
        
        # 基本メタデータを作成
        metadata = {
            "source": file_path,
            "relative_path": relative_path,
            "file_name": file_name,
            "source_type": "code",
            "language": language,
            "file_extension": os.path.splitext(file_name)[1],
        }
        
        # コード分割クラスをインスタンス化
        splitter = CodeTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap,
            language=language
        )
        
        # テキストを分割し、メタデータを付与
        chunks = splitter.split_with_metadata(text, metadata)
        
        logger.info(f"Successfully processed {file_path} into {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Error processing code file {file_path}: {e}")
        return []
        
def process_code_directory(directory: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    ディレクトリ内のすべてのソースコードファイルを処理する
    
    Args:
        directory: 処理するディレクトリのパス
        chunk_size: 分割するチャンクのサイズ
        chunk_overlap: チャンク間のオーバーラップ文字数
        
    Returns:
        すべてのファイルの分割されたコードとメタデータを含む辞書のリスト
    """
    logger.info(f"Processing code files in directory: {directory}")
    
    all_chunks = []
    
    # ディレクトリがなければエラー
    if not os.path.exists(directory):
        logger.error(f"Directory does not exist: {directory}")
        return all_chunks
        
    # 再帰的にすべてのソースコードファイルを処理
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            # 対応する拡張子かチェック
            ext = os.path.splitext(file)[1].lower()
            if ext in FILE_EXTENSIONS:
                chunks = process_code_file(file_path, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
                
    logger.info(f"Processed {len(all_chunks)} total chunks from directory: {directory}")
    return all_chunks