import os
import logging
import pdfplumber
from typing import List, Dict, Any
from vectorize.text_splitters import BaseTextSplitter
import config

# ロギング設定
logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    PDFからテキストとメタデータを抽出
    
    Args:
        file_path: PDFファイルのパス
        
    Returns:
        各ページのテキストとメタデータのリスト
    """
    logger.info(f"Extracting text from PDF: {file_path}")
    
    pages = []
    
    try:
        with pdfplumber.open(file_path) as pdf:
            # PDFのメタデータを取得（存在すれば）
            pdf_metadata = pdf.metadata or {}
            
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                
                # ページ情報
                page_info = {
                    "page_number": i + 1,
                    "page_total": len(pdf.pages),
                    "width": page.width,
                    "height": page.height
                }
                
                pages.append({
                    "text": page_text,
                    "metadata": {
                        **page_info,
                        "pdf_title": pdf_metadata.get("Title", ""),
                        "pdf_author": pdf_metadata.get("Author", ""),
                        "pdf_subject": pdf_metadata.get("Subject", ""),
                        "pdf_creator": pdf_metadata.get("Creator", "")
                    }
                })
                
        logger.info(f"Successfully extracted text from {len(pages)} pages")
        return pages
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {e}")
        return []

def process_pdf_file(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    PDFファイルを処理して、チャンクとメタデータを抽出する
    
    Args:
        file_path: 処理するPDFファイルのパス
        chunk_size: 分割するチャンクのサイズ
        chunk_overlap: チャンク間のオーバーラップ文字数
        
    Returns:
        分割されたテキストとメタデータを含む辞書のリスト
    """
    logger.info(f"Processing PDF file: {file_path}")
    
    try:
        # PDFからテキストとメタデータを抽出
        pdf_pages = extract_text_from_pdf(file_path)
        
        if not pdf_pages:
            logger.warning(f"No text extracted from {file_path}")
            return []
            
        # ファイル名とパス情報を取得
        file_name = os.path.basename(file_path)
        relative_path = os.path.relpath(file_path, start=config.BASE_DIR)
        
        # テキスト分割器を初期化
        splitter = BaseTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        all_chunks = []
        
        # 各ページを処理
        for page in pdf_pages:
            page_text = page["text"]
            page_metadata = page["metadata"]
            
            # 基本メタデータを作成
            metadata = {
                "source": file_path,
                "relative_path": relative_path,
                "file_name": file_name,
                "source_type": "pdf",
                "file_extension": ".pdf",
                "page_number": page_metadata["page_number"],
                "page_total": page_metadata["page_total"]
            }
            
            # PDFのメタデータを追加
            for key in ["pdf_title", "pdf_author", "pdf_subject", "pdf_creator"]:
                if page_metadata.get(key):
                    metadata[key] = page_metadata[key]
            
            # テキストを分割し、メタデータを付与
            page_chunks = splitter.split_with_metadata(page_text, metadata)
            all_chunks.extend(page_chunks)
        
        logger.info(f"Successfully processed {file_path} into {len(all_chunks)} chunks")
        return all_chunks
        
    except Exception as e:
        logger.error(f"Error processing PDF file {file_path}: {e}")
        return []
        
def process_pdf_directory(directory: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    ディレクトリ内のすべてのPDFファイルを処理する
    
    Args:
        directory: 処理するディレクトリのパス
        chunk_size: 分割するチャンクのサイズ
        chunk_overlap: チャンク間のオーバーラップ文字数
        
    Returns:
        すべてのファイルの分割されたテキストとメタデータを含む辞書のリスト
    """
    logger.info(f"Processing PDF files in directory: {directory}")
    
    all_chunks = []
    
    # ディレクトリがなければエラー
    if not os.path.exists(directory):
        logger.error(f"Directory does not exist: {directory}")
        return all_chunks
        
    # 再帰的にすべてのPDFファイルを処理
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = os.path.join(root, file)
                chunks = process_pdf_file(file_path, chunk_size, chunk_overlap)
                all_chunks.extend(chunks)
                
    logger.info(f"Processed {len(all_chunks)} total chunks from directory: {directory}")
    return all_chunks