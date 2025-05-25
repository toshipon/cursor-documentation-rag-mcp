import os
import sys
import time
import logging
import queue
import threading
from typing import List, Dict, Any, Optional, Tuple

# プロジェクトのルートディレクトリをシステムパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from db.vector_store import VectorStore
from db.qdrant_store import QdrantVectorStore
from vectorize.embeddings import PLaMoEmbedder, DummyEmbedder
# Import processor classes and functions
from vectorize.processors.pdf_processor import PDFProcessor
from vectorize.processors.markdown_processor import process_markdown_file
from vectorize.processors.code_processor import process_code_file, FILE_EXTENSIONS as CODE_FILE_EXTENSIONS

# from workers.file_watcher import VectorizationEvent # No longer used, using dicts

# ロギング設定
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Wrapper classes for functional processors to make them conform to the class-based interface
class FunctionalProcessorWrapper:
    def __init__(self, process_function: callable, **kwargs):
        self.process_function = process_function
        self.kwargs = kwargs # To pass any default args like chunk_size if needed

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        # Pass through kwargs to the original function if they are part of its signature
        # This example assumes process_function takes file_path and optional config from kwargs
        # Modify as per actual function signatures if they take more relevant config
        try:
            return self.process_function(file_path, **self.kwargs)
        except Exception as e:
            logger.error(f"Error in functional processor wrapper for {file_path} with {self.process_function.__name__}: {e}", exc_info=True)
            return []

# ファイル拡張子とプロセッサクラスのマッピング
PROCESSOR_CLASSES = {
    '.pdf': PDFProcessor,
    # For markdown and code, use the wrapper for their respective functions
    '.md': lambda: FunctionalProcessorWrapper(process_markdown_file), # Example: chunk_size=400, chunk_overlap=50
    '.markdown': lambda: FunctionalProcessorWrapper(process_markdown_file),
}

# Dynamically add code processors using the wrapper
# Since process_code_file determines language from file_path and FunctionalProcessorWrapper
# doesn't need ext explicitly, the lambda can be simplified.
# This factory will create a new wrapper instance each time it's called.
common_code_wrapper_factory = lambda: FunctionalProcessorWrapper(process_code_file)
for ext in CODE_FILE_EXTENSIONS.keys():
    PROCESSOR_CLASSES[ext] = common_code_wrapper_factory

class VectorizationWorker:
    """ファイル変更を検出して自動的にベクトル化するワーカー"""
    
    def __init__(self, 
                 event_queue: queue.Queue,
                 vector_store_path: str = None,
                 use_dummy_embedder: bool = False,
                 batch_size: int = 10,
                 max_queue_size: int = 100,
                 vector_dimension: int = 512):
        """
        初期化
        
        Args:
            event_queue: ベクトル化イベントを取得するキュー
            vector_store_path: ベクトルストアのパス
            use_dummy_embedder: ダミー埋め込みモデルを使用するかどうか
            batch_size: 一度に処理するイベントの最大数
            max_queue_size: キューの最大サイズ
            vector_dimension: ベクトルの次元数
        """
        self.event_queue = event_queue
        self.vector_store_path = vector_store_path or config.VECTOR_DB_PATH
        self.use_dummy_embedder = use_dummy_embedder
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.vector_dimension = vector_dimension
        
        # ベクトルストアを初期化
        self.vector_store = None
        
        # 埋め込みモデルを初期化
        self.embedder = None
        
        # ワーカースレッド
        self.worker_thread = None
        self._stop_event = threading.Event()
        self._is_running = False
    
    def _init_resources(self) -> None:
        """リソースを初期化"""
        # ベクトルストアを初期化
        if self.vector_store is None:
            vector_store_type = os.getenv("VECTOR_STORE_TYPE", "sqlite")
            
            if vector_store_type.lower() == "qdrant":
                logger.info("Initializing Qdrant vector store")
                qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6334")
                collection_name = os.getenv("QDRANT_COLLECTION", "documents")
                self.vector_store = QdrantVectorStore(
                    url=qdrant_url,
                    collection_name=collection_name,
                    vector_dimension=self.vector_dimension
                )
            else:
                logger.info(f"Initializing SQLite vector store at {self.vector_store_path}")
                os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
                self.vector_store = VectorStore(self.vector_store_path, self.vector_dimension)
        
        # 埋め込みモデルを初期化
        if self.embedder is None:
            if self.use_dummy_embedder:
                logger.info("Using dummy embedder")
                self.embedder = DummyEmbedder(dim=self.vector_dimension)
            else:
                logger.info("Initializing PLaMo-Embedding-1B model")
                try:
                    self.embedder = PLaMoEmbedder(model_path=config.EMBEDDING_MODEL_PATH)
                except Exception as e:
                    logger.error(f"Error loading PLaMo model: {e}. Falling back to dummy embedder.")
                    self.embedder = DummyEmbedder(dim=self.vector_dimension)
    
    def _release_resources(self) -> None:
        """リソースを解放"""
        if self.vector_store:
            self.vector_store.close()
            self.vector_store = None
        
        # 埋め込みモデルは明示的な解放は不要
        self.embedder = None
    
    def _get_document_processor(self, file_path: str) -> Optional[Any]:
        """
        Factory function to get an instance of the appropriate document processor.
        
        Args:
            file_path: The path to the file to be processed.
            
        Returns:
            An instance of a document processor (e.g., PDFProcessor) or None if no
            suitable processor is found.
        """
        ext = os.path.splitext(file_path)[1].lower()
        processor_class = PROCESSOR_CLASSES.get(ext)
        
        if processor_class:
            try:
                # If processors require config or other dependencies, pass them here.
                # For PDFProcessor, it seems to be self-contained for now.
                return processor_class() 
            except Exception as e:
                logger.error(f"Error instantiating processor {processor_class.__name__} for {file_path}: {e}")
                return None
        else:
            logger.warning(f"No processor class found for extension {ext} ({file_path})")
            return None

    def _process_update_action(self, file_path: str) -> bool:
        """
        Processes a file for an 'update' action (create or modify).
        It extracts content, generates embeddings, and updates the vector store.
        
        Args:
            file_path: The path to the file to be processed.
            
        Returns:
            True if processing was successful, False otherwise.
        """
        if not os.path.exists(file_path):
            logger.error(f"File does not exist for update: {file_path}")
            return False

        # Check if file actually needs update (e.g., based on modification time or hash)
        # This logic might be better inside VectorStore or a utility if complex.
        # For now, we assume if it's in the queue, it needs processing.
        # if not self.vector_store.file_needs_update(file_path): # Assuming this method exists
        #     logger.info(f"Skipping file (no updates based on store check): {file_path}")
        #     return True

        processor = self._get_document_processor(file_path)
        if not processor:
            logger.warning(f"Unsupported file type for update: {file_path}")
            return False

        try:
            logger.info(f"Processing file for update: {file_path}")
            start_time = time.time()

            # processor.process_file is expected to return List[Dict[str, Any]]
            # where each dict has 'text' and 'metadata' (like {'source': ..., 'page_number': ...})
            chunks = processor.process_file(file_path)

            if not chunks:
                logger.warning(f"No content (chunks) extracted from {file_path}")
                # It's important to delete existing entries if the file becomes empty or unreadable
                # to avoid stale data.
                deleted_count = self.vector_store.delete_by_source(file_path)
                if deleted_count > 0:
                    logger.info(f"Deleted {deleted_count} existing documents for now empty/unreadable file: {file_path}")
                return False # Or True, depending on whether "no content" is an error or valid state

            logger.info(f"Extracted {len(chunks)} chunks from {file_path}")

            texts_to_embed = []
            valid_chunks = []
            for i, chunk_doc in enumerate(chunks):
                text = chunk_doc.get("text", "").strip()
                if not text:
                    logger.warning(f"Empty text in chunk {i} from {file_path}, skipping this chunk.")
                    continue
                texts_to_embed.append(text)
                valid_chunks.append(chunk_doc)
            
            if not texts_to_embed:
                logger.error(f"No valid text content found in any chunks from {file_path} after filtering.")
                # Again, consider deleting existing entries
                deleted_count = self.vector_store.delete_by_source(file_path)
                if deleted_count > 0:
                    logger.info(f"Deleted {deleted_count} existing documents for file with no valid text: {file_path}")
                return False

            # Generate embeddings for the valid texts
            vectors = self.embedder.embed_batch(texts_to_embed)

            # Before adding new documents, delete existing ones for this source
            # This handles updates correctly by replacing old content.
            delete_count = self.vector_store.delete_by_source(file_path)
            if delete_count > 0:
                logger.info(f"Deleted {delete_count} existing documents for {file_path} before update.")

            # Ensure each chunk's metadata includes the source (file_path)
            for chunk in valid_chunks:
                if "metadata" not in chunk:
                    chunk["metadata"] = {}
                if "source" not in chunk["metadata"]: # Only set if not already present
                    chunk["metadata"]["source"] = file_path
            
            # Add the new documents (valid_chunks with their vectors) to the vector store
            # The add_documents method needs to associate vectors with their respective chunks.
            # It should handle the 'metadata' from each chunk.
            self.vector_store.add_documents(valid_chunks, vectors)

            elapsed = time.time() - start_time
            logger.info(f"Successfully processed (updated) {file_path} in {elapsed:.2f} seconds. Added {len(valid_chunks)} chunks.")
            return True

        except Exception as e:
            logger.error(f"Error processing file {file_path} for update: {e}", exc_info=True)
            return False

    def _process_delete_action(self, file_path: str) -> bool:
        """
        Processes a 'delete' action for a file by removing its associated
        documents from the vector store.
        
        Args:
            file_path: The path of the file whose documents should be deleted.
            
        Returns:
            True if deletion was successful or if no documents needed deletion,
            False if an error occurred.
        """
        try:
            logger.info(f"Processing delete action for file: {file_path}")
            # Use delete_by_source to remove all documents originating from this file_path
            deleted_count = self.vector_store.delete_by_source(file_path)
            
            if deleted_count > 0:
                logger.info(f"Successfully deleted {deleted_count} documents from vector store for source: {file_path}")
            else:
                logger.info(f"No documents found in vector store for source: {file_path} (delete action)")
            return True # Success even if no documents were found, as the state is consistent
            
        except Exception as e:
            logger.error(f"Error deleting documents for source {file_path} from vector store: {e}", exc_info=True)
            return False
    
    def _process_events(self) -> None:
        """Continuously monitors the event queue and processes file events."""
        while not self._stop_event.is_set():
            try:
                self._init_resources() # Ensure DB and embedder are ready
                
                try:
                    event_task = self.event_queue.get(timeout=1.0) # Wait for 1 sec
                except queue.Empty:
                    continue # No event, continue loop
                
                file_path = event_task.get('file_path')
                action = event_task.get('action')

                if not file_path or not action:
                    logger.warning(f"Invalid event task received: {event_task}")
                    self.event_queue.task_done()
                    continue
                
                logger.info(f"Dequeued event: Action: {action}, File: {file_path}")
                
                success = False
                if action == 'update': # Covers 'create' and 'modify'
                    success = self._process_update_action(file_path)
                elif action == 'delete':
                    success = self._process_delete_action(file_path)
                else:
                    logger.warning(f"Unknown action '{action}' for file {file_path}")
                
                if success:
                    try:
                        stats = self.vector_store.get_stats() # Assuming this method exists
                        logger.info(f"Vector store stats: {stats.get('total_documents', 0)} documents, "
                                   f"{stats.get('files', {}).get('count', 0)} distinct files.") # Adjusted for potential stats structure
                    except Exception as e:
                        logger.warning(f"Could not retrieve vector store stats: {e}")
                
                self.event_queue.task_done()
                    
            except Exception as e: # Catch broad exceptions in the loop to keep worker alive
                logger.error(f"Unhandled error in event processing loop: {e}", exc_info=True)
                # Potentially add a small delay before retrying to prevent rapid failure loops
                time.sleep(5) 
                
        self._release_resources()
        logger.info("Vectorization worker's main event loop stopped.")
    
    def start(self) -> None:
        """ベクトル化ワーカーを開始"""
        if self._is_running:
            logger.warning("Vectorization worker is already running")
            return
        
        # ストップイベントをリセット
        self._stop_event.clear()
        
        # ワーカースレッドを作成して開始
        self.worker_thread = threading.Thread(target=self._process_events)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        
        self._is_running = True
        logger.info("Vectorization worker started")
    
    def stop(self) -> None:
        """ベクトル化ワーカーを停止"""
        if not self._is_running:
            return
        
        # ストップイベントをセット
        self._stop_event.set()
        
        # ワーカースレッドが終了するのを待つ
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
            
        self._is_running = False
        logger.info("Vectorization worker stopping")
    
    def is_running(self) -> bool:
        """ワーカーが実行中かどうかを返す"""
        return self._is_running


def create_vectorization_worker(event_queue: queue.Queue,
                               vector_store_path: str = None,
                               use_dummy_embedder: bool = False) -> VectorizationWorker:
    """
    VectorizationWorkerのインスタンスを作成
    
    Args:
        event_queue: ベクトル化イベントを取得するキュー
        vector_store_path: ベクトルストアのパス
        use_dummy_embedder: ダミー埋め込みモデルを使用するかどうか
        
    Returns:
        VectorizationWorkerのインスタンス
    """
    return VectorizationWorker(
        event_queue=event_queue,
        vector_store_path=vector_store_path,
        use_dummy_embedder=use_dummy_embedder
    )
