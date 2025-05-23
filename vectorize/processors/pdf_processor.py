import os
import logging
import pdfplumber
import gc
import traceback
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
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    try:
        with pdfplumber.open(file_path) as pdf:
            # PDFのメタデータを取得（存在すれば）
            pdf_metadata = pdf.metadata or {}
            total_pages = len(pdf.pages)
            
            # ファイルサイズに基づいてバッチサイズを調整
            if file_size_mb > 100:
                batch_size = 10  # 非常に大きいPDFの場合
            elif file_size_mb > 50:
                batch_size = 20  # 大きいPDFの場合
            elif file_size_mb > 30:
                batch_size = 30  # 中程度のPDFの場合
            else:
                batch_size = 50  # 標準的なPDFの場合
                
            logger.info(f"PDF size: {file_size_mb:.1f} MB, {total_pages} pages, using batch size of {batch_size} pages")
            
            # 進捗表示の頻度を設定
            progress_interval = max(1, total_pages // 10)
            
            # メモリ消費を減らすためにページごとにメモリをクリア
            for batch_start in range(0, total_pages, batch_size):
                batch_end = min(batch_start + batch_size, total_pages)
                logger.info(f"Extracting pages {batch_start+1} to {batch_end} of {total_pages}")
                
                batch_pages = []
                for i in range(batch_start, batch_end):
                    try:
                        page = pdf.pages[i]
                        page_text = page.extract_text() or ""
                        
                        # ページ情報
                        page_info = {
                            "page_number": i + 1,
                            "page_total": total_pages,
                            "width": page.width,
                            "height": page.height
                        }
                        
                        batch_pages.append({
                            "text": page_text,
                            "metadata": {
                                **page_info,
                                "pdf_title": pdf_metadata.get("Title", ""),
                                "pdf_author": pdf_metadata.get("Author", ""),
                                "pdf_subject": pdf_metadata.get("Subject", ""),
                                "pdf_creator": pdf_metadata.get("Creator", "")
                            }
                        })
                        
                        # 進捗を定期的に表示
                        if (i + 1) % progress_interval == 0:
                            logger.info(f"Progress: {i+1}/{total_pages} pages processed")
                            
                        # 明示的にページオブジェクトを解放
                        del page
                        
                    except Exception as page_error:
                        logger.warning(f"Error extracting text from page {i+1}: {page_error}, skipping page")
                        # エラーが発生したページは空のテキストとして追加
                        batch_pages.append({
                            "text": "",
                            "metadata": {
                                "page_number": i + 1,
                                "page_total": total_pages,
                                "error": str(page_error)
                            }
                        })
                
                # バッチ処理したページをメインリストに追加
                pages.extend(batch_pages)
                
                # バッチごとにGCを実行してメモリを解放
                import gc
                gc.collect()
                
            logger.info(f"Successfully extracted text from {len(pages)} pages")
            return pages
            
    except MemoryError as me:
        logger.error(f"Memory error extracting text from PDF {file_path}: {me}")
        # 部分的に抽出できたページだけでも返す
        if pages:
            logger.info(f"Returning {len(pages)} pages that were successfully extracted before memory error")
            return pages
        return []
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {e}")
        # スタックトレースを出力
        import traceback
        logger.error(traceback.format_exc())
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
    # ファイルサイズをチェックし、大きいPDFの場合はチャンクサイズを自動調整
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > 30:  # 30MB以上のPDF
        orig_chunk_size = chunk_size
        # ファイルサイズに応じてチャンクサイズを調整
        if file_size_mb > 100:
            chunk_size = min(chunk_size, 200)  # 100MB以上の場合は最大200文字
        elif file_size_mb > 50:
            chunk_size = min(chunk_size, 300)  # 50-100MBの場合は最大300文字
        else:
            chunk_size = min(chunk_size, 400)  # 30-50MBの場合は最大400文字
        
        if chunk_size != orig_chunk_size:
            logger.info(f"Large PDF detected ({file_size_mb:.1f} MB), adjusted chunk size from {orig_chunk_size} to {chunk_size}")
    
    # より小さいオーバーラップを使用して更にメモリ使用量を削減
    chunk_overlap = min(chunk_overlap, chunk_size // 10)
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