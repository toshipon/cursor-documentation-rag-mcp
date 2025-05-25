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
from vectorize.processors.markdown_processor import process_markdown_file
from vectorize.processors.pdf_processor import process_pdf_file
from vectorize.processors.code_processor import process_code_file
from workers.file_watcher import VectorizationEvent

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
    
    def _get_processor(self, file_path: str) -> Optional[callable]:
        """
        ファイルタイプに対応するプロセッサ関数を取得
        
        Args:
            file_path: ファイルパス
            
        Returns:
            プロセッサ関数。対応するものがなければNone
        """
        ext = os.path.splitext(file_path)[1].lower()
        return FILE_PROCESSORS.get(ext)
    
    def _process_file(self, file_path: str) -> bool:
        """
        単一ファイルを処理してベクトル化
        
        Args:
            file_path: ファイルパス
            
        Returns:
            処理が成功したかどうか
        """
        # ファイルが存在しなければエラー
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return False
        
        # 既に処理済みでかつ更新がない場合はスキップ
        if not self.vector_store.file_needs_update(file_path):
            logger.info(f"Skipping file (no updates): {file_path}")
            return True
        
        # プロセッサを取得
        processor = self._get_processor(file_path)
        if processor is None:
            logger.warning(f"Unsupported file type: {file_path}")
            return False
        
        try:
            # ファイルを処理してチャンクを取得
            logger.info(f"Processing file: {file_path}")
            start_time = time.time()
            
            docs = processor(file_path)
            
            if not docs:
                logger.warning(f"No content extracted from {file_path}")
                return False
            
            logger.info(f"Extracted {len(docs)} chunks from {file_path}")
            
            # 既に処理済みのファイルは一旦削除
            if self.vector_store.file_exists(file_path):
                self.vector_store.delete_file(file_path)
            
            # テキストをベクトル化
            texts = []
            for i, doc in enumerate(docs):
                text = doc.get("text", doc.get("content", "")).strip()
                if not text:
                    logger.warning(f"Empty text in chunk {i} from {file_path}")
                    continue
                texts.append(text)

            if not texts:
                logger.error(f"No valid text content found in any chunks from {file_path}")
                return False

            # テキストの長さをログに出力
            for i, text in enumerate(texts):
                logger.debug(f"Chunk {i} from {file_path}: {len(text)} characters")

            vectors = self.embedder.embed_batch(texts)
            
            # ベクトルストアに保存
            self.vector_store.add_documents(docs, vectors, file_path=file_path)
            
            elapsed = time.time() - start_time
            logger.info(f"Successfully processed {file_path} in {elapsed:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return False
    
    def _process_delete(self, file_path: str) -> bool:
        """
        ファイル削除イベントを処理
        
        Args:
            file_path: 削除されたファイルのパス
            
        Returns:
            処理が成功したかどうか
        """
        try:
            # ベクトルストアからファイルを削除
            logger.info(f"Deleting file from vector store: {file_path}")
            deleted_count = self.vector_store.delete_file(file_path)
            
            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} documents for {file_path}")
                return True
            else:
                logger.warning(f"No documents found for {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return False
    
    def _process_events(self) -> None:
        """イベントキューからイベントを取得して処理"""
        while not self._stop_event.is_set():
            try:
                # リソースを確保
                self._init_resources()
                
                # キューから次のイベントを取得
                try:
                    # 一定時間キューからイベントを待つ
                    event = self.event_queue.get(timeout=1.0)
                except queue.Empty:
                    # タイムアウトしたら次のループへ
                    continue
                
                try:
                    # イベントを処理
                    if event.event_type == 'modified':
                        success = self._process_file(event.file_path)
                    elif event.event_type == 'deleted':
                        success = self._process_delete(event.file_path)
                    else:
                        logger.warning(f"Unknown event type: {event.event_type}")
                        success = False
                    
                    # 統計情報をログに出力
                    if success:
                        stats = self.vector_store.get_stats()
                        logger.info(f"Vector store stats: {stats.get('total_documents', 0)} documents, "
                                   f"{stats.get('total_files', 0)} files")
                                   
                finally:
                    # キューからの取得が完了したことを通知
                    self.event_queue.task_done()
                    
            except Exception as e:
                logger.error(f"Error in event processing: {e}")
                
        # リソースを解放
        self._release_resources()
        logger.info("Vectorization worker stopped")
    
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
