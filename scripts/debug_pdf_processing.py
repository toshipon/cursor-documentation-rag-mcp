import os
import sys
import logging
import json
from vectorize.processors.pdf_processor import PDFProcessor
from db.vector_store import VectorStore

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def format_preview(text: str, max_length: int = 100) -> str:
    """テキストのプレビューを整形"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def log_chunk_info(chunk: dict, index: int):
    """チャンクの詳細情報をログ出力"""
    logger.info(f"\n=== Chunk {index} ===")
    logger.info(f"Type: {'Table' if chunk['metadata'].get('contains_table') else 'Text'}")
    logger.info(f"Size: {chunk['metadata'].get('chunk_size')} chars")
    logger.info(f"Confidence: {chunk['metadata'].get('confidence', 'N/A')}")
    
    # テーブルの場合は構造も表示
    if chunk['metadata'].get('contains_table'):
        logger.info("\nContent Preview:")
        logger.info("-" * 80)
        # テーブルの最初の部分を表示
        lines = chunk['content'].split('\n')
        for line in lines[:min(10, len(lines))]:
            logger.info(line)
        if len(lines) > 10:
            logger.info("...")
        logger.info("-" * 80)

def process_pdf(file_path: str):
    """PDFファイルを処理し、抽出結果を詳細に表示"""
    try:
        logger.info(f"Processing PDF: {file_path}")
        
        # PDFプロセッサの設定
        processor = PDFProcessor(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # PDFを処理してチャンクを取得
        chunks = processor.process(file_path)
        if not chunks:
            logger.error("No chunks generated from PDF")
            return

        # 処理結果の統計
        total_chunks = len(chunks)
        table_chunks = sum(1 for c in chunks if c['metadata'].get('contains_table'))
        text_chunks = total_chunks - table_chunks
        
        logger.info("\n=== Processing Summary ===")
        logger.info(f"Total chunks: {total_chunks}")
        logger.info(f"Table chunks: {table_chunks}")
        logger.info(f"Text chunks: {text_chunks}")
        
        # テーブルチャンクの詳細を表示
        if table_chunks:
            logger.info("\n=== Table Chunks Details ===")
            table_chunk_indices = [
                i for i, c in enumerate(chunks) 
                if c['metadata'].get('contains_table')
            ]
            
            for i in table_chunk_indices[:5]:  # 最初の5つのテーブルのみ表示
                log_chunk_info(chunks[i], i)
        
        # ベクトルストアの準備
        store = VectorStore("vector_store")
        
        # チャンクをベクトルストアに保存
        for chunk in chunks:
            store.add_document(chunk['content'], chunk['metadata'])
            
        logger.info("\n=== Storage Complete ===")
        logger.info(f"Successfully processed and stored {len(chunks)} chunks")

    except Exception as e:
        logger.error(f"Error processing PDF: {e}", exc_info=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_pdf_processing.py <pdf_file_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Error: File not found - {pdf_path}")
        sys.exit(1)

    process_pdf(pdf_path)