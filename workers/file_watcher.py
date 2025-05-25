import os
import sys
import time
import logging
import queue
import threading
from typing import Set, Dict, Any, List, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

# プロジェクトのルートディレクトリをシステムパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# ロギング設定
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Default supported extensions if not provided by config or constructor
DEFAULT_SUPPORTED_EXTENSIONS = {
    '.md', '.markdown', '.txt', '.log', '.pdf', '.json', '.yaml', '.yml',
    '.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp', '.cs', '.go',
    '.php', '.rb', '.swift', '.kt', '.scala', '.rs', '.lua', '.pl', '.sh',
    '.html', '.htm', '.css', '.scss', '.less', '.xml', '.toml', '.ini', '.cfg', '.conf'
}

class _InternalChangeEvent:
    """内部処理用のファイル変更イベントを表すクラス (デバウンス処理用)"""
    
    def __init__(self, file_path: str, event_type: str, timestamp: float = None):
        """
        初期化
        
        Args:
            file_path: ファイルパス
            event_type: イベントタイプ ('modified', 'deleted')
            timestamp: イベント発生時刻（Unix時間）
        """
        self.file_path = file_path
        self.event_type = event_type # 'modified' (for create/update) or 'deleted'
        self.timestamp = timestamp or time.time()
    
    def __str__(self) -> str:
        return f"_InternalChangeEvent({self.event_type}, {self.file_path}, {self.timestamp})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, _InternalChangeEvent):
            return False
        return self.file_path == other.file_path and self.event_type == other.event_type
    
    def __hash__(self) -> int:
        return hash((self.file_path, self.event_type))

class DocumentEventHandler(FileSystemEventHandler):
    """ファイルシステムのイベントを処理するハンドラー"""
    
    def __init__(self, event_queue: queue.Queue, ignored_patterns: List[str] = None,
                 supported_extensions: Optional[Set[str]] = None,
                 debounce_seconds: float = 2.0):
        """
        初期化
        
        Args:
            event_queue: イベント辞書を格納するキュー
            ignored_patterns: 無視するファイルパターンのリスト
            supported_extensions: サポートするファイル拡張子のセット。Noneの場合、configから読み込む。
            debounce_seconds: イベントのデバウンス時間（秒）
        """
        self.event_queue = event_queue
        self.ignored_patterns = ignored_patterns or []
        self.debounce_seconds = debounce_seconds
        
        if supported_extensions is not None:
            self.supported_extensions = set(supported_extensions)
        else:
            cfg_extensions = getattr(config, 'SUPPORTED_EXTENSIONS', None)
            if cfg_extensions is not None:
                self.supported_extensions = set(cfg_extensions)
                logger.info("Using SUPPORTED_EXTENSIONS from config.py")
            else:
                self.supported_extensions = DEFAULT_SUPPORTED_EXTENSIONS
                logger.info("Using DEFAULT_SUPPORTED_EXTENSIONS from file_watcher.py")
        
        logger.debug(f"DocumentEventHandler initialized with extensions: {self.supported_extensions}")

        self.last_events: Dict[str, float] = {}  # ファイルパスごとの最終イベント時刻
        self.pending_events: Set[_InternalChangeEvent] = set()  # 保留中の内部イベント
        
        # デバウンスタイマー
        self.timer = None
        self.timer_lock = threading.Lock()
    
    def _is_target_file(self, file_path: str) -> bool:
        """
        ファイルが処理対象かどうかを判定 (旧 _should_ignore のロジックを反転・変更)
        
        Args:
            file_path: ファイルパス
            
        Returns:
            処理対象ならTrue、そうでなければFalse
        """
        # 隠しファイルは無視
        if os.path.basename(file_path).startswith('.'):
            logger.debug(f"Ignoring hidden file: {file_path}")
            return False
        
        # 特定の無視ディレクトリをチェック
        # configから読み込むように変更することも検討 (例: config.IGNORED_DIRECTORIES)
        ignored_dirs = getattr(config, 'WATCHER_IGNORED_DIRECTORIES', ['__pycache__', '.git', 'venv', 'env', '.venv', '.env', 'node_modules'])
        for ignored_dir_name in ignored_dirs:
            # パスセパレータを考慮してチェック
            if os.path.sep + ignored_dir_name + os.path.sep in file_path or \
               file_path.endswith(os.path.sep + ignored_dir_name):
                logger.debug(f"Ignoring file in ignored directory '{ignored_dir_name}': {file_path}")
                return False
        
        # 無視パターンをチェック
        for pattern in self.ignored_patterns:
            if pattern in file_path: # シンプルな部分文字列一致。正規表現も検討可。
                logger.debug(f"Ignoring file due to pattern '{pattern}': {file_path}")
                return False
        
        # 拡張子をチェック
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.supported_extensions:
            logger.debug(f"Ignoring file with unsupported extension '{ext}': {file_path}")
            return False
            
        return True # すべてのチェックをパスした場合、処理対象
    
    def _process_event(self, event: FileSystemEvent) -> None:
        """
        watchdogイベントを処理
        
        Args:
            event: watchdogイベント
        """
        # ディレクトリの場合は無視
        if event.is_directory:
            return

        action: Optional[str] = None
        file_path: Optional[str] = None

        if event.event_type == 'created' or event.event_type == 'modified':
            action = 'update' # 'create' or 'update' as per new requirement
            file_path = event.src_path
            if not self._is_target_file(file_path):
                return
        elif event.event_type == 'deleted':
            action = 'delete'
            file_path = event.src_path
            # For deleted files, we might still want to process them if they were previously target files.
            # The _is_target_file check might be less relevant here, or based on prior state.
            # However, for simplicity, we check its extension. If it's a hidden file, etc., it might be ignored.
            # This logic depends on whether we want to clean up non-target files from vector store.
            # For now, we'll apply similar filtering.
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in self.supported_extensions:
                 # If it's not a supported extension, we might not care about its deletion for vectorization.
                 # However, if it was moved from a supported name to an unsupported one, this could be an issue.
                 # The 'moved' event handler below is better for this.
                logger.debug(f"Ignoring deletion of non-target or filtered file: {file_path}")
                return
        elif event.event_type == 'moved':
            src_path = event.src_path
            dest_path = event.dest_path

            # Treat as deletion of source if it was a target file
            # Check if src_path itself would be a target (ignoring its current existence)
            src_ext = os.path.splitext(src_path)[1].lower()
            if src_ext in self.supported_extensions: # Basic check for relevant deletion
                self._add_internal_event(_InternalChangeEvent(src_path, 'deleted'))

            # Treat as creation/modification of destination if it is a target file
            if self._is_target_file(dest_path):
                self._add_internal_event(_InternalChangeEvent(dest_path, 'modified'))
            return # Handled by _add_internal_event
        else:
            logger.debug(f"Ignoring event type: {event.event_type} for path: {event.src_path}")
            return

        if action and file_path:
            internal_event = _InternalChangeEvent(file_path, 'modified' if action == 'update' else 'deleted')
            self._add_internal_event(internal_event)

    def _add_internal_event(self, event: _InternalChangeEvent) -> None:
        """
        内部イベントを保留リストに追加し、デバウンスタイマーを設定
        
        Args:
            event: 内部ファイル変更イベント
        """
        # 同じファイルの既存のイベントを削除して新しいイベントに置き換え
        # If a 'delete' comes after 'modified' for the same file, 'delete' should prevail.
        # If 'modified' comes after 'delete' (e.g. quick recreate), 'modified' should prevail.
        # Current logic: last event wins.
        self.pending_events = {e for e in self.pending_events if e.file_path != event.file_path}
        self.pending_events.add(event)
        
        # 最終イベント時刻を更新
        self.last_events[event.file_path] = time.time()
        
        # デバウンスタイマーをリセット
        with self.timer_lock:
            if self.timer:
                self.timer.cancel()
            self.timer = threading.Timer(self.debounce_seconds, self._flush_pending_events)
            self.timer.daemon = True
            self.timer.start()
    
    def _flush_pending_events(self) -> None:
        """デバウンス期間が終了したら、保留中のすべてのイベントをキューに追加（辞書形式で）"""
        current_time = time.time()
        events_to_queue = []
        
        processed_paths_in_batch = set()

        # デバウンス期間が終了したイベントを見つける
        for internal_event in list(self.pending_events): # Iterate over a copy
            last_event_time = self.last_events.get(internal_event.file_path, 0)
            # Process if debounce time has passed AND this path hasn't been processed in this flush cycle
            if (current_time - last_event_time >= self.debounce_seconds) and \
               (internal_event.file_path not in processed_paths_in_batch):
                
                action_for_queue = 'update' if internal_event.event_type == 'modified' else 'delete'
                
                task = {
                    'file_path': internal_event.file_path,
                    'action': action_for_queue
                }
                events_to_queue.append(task)
                self.pending_events.remove(internal_event)
                processed_paths_in_batch.add(internal_event.file_path)
        
        # イベントをキューに追加
        for task in events_to_queue:
            logger.info(f"Queueing task: {task}")
            self.event_queue.put(task)
        
        # まだ保留中のイベントがある場合は、タイマーを再設定
        # (e.g. new events came in while flushing, or some events had shorter debounce time than others)
        if self.pending_events:
            with self.timer_lock:
                # Check if timer is already running or has been cancelled
                if self.timer: # if timer was cancelled by external stop, don't restart
                    self.timer.cancel() 
                
                # Find the minimum remaining time to wait for any pending event
                min_wait_time = self.debounce_seconds
                now = time.time()
                if self.pending_events: # Check again, might have been cleared
                    try:
                        min_wait_time = min(
                            self.debounce_seconds - (now - self.last_events.get(e.file_path, now))
                            for e in self.pending_events
                        )
                        min_wait_time = max(0, min_wait_time) # Ensure non-negative
                    except ValueError: # No pending events
                        pass

                if self.pending_events: # Final check
                     self.timer = threading.Timer(min_wait_time, self._flush_pending_events)
                     self.timer.daemon = True
                     self.timer.start()
                else:
                    self.timer = None # All events flushed
    
    # イベントハンドラー（watchdogからのコールバック）
    def on_created(self, event: FileSystemEvent) -> None:
        """ファイル作成イベントのハンドラー"""
        logger.debug(f"File created: {event.src_path}")
        self._process_event(event)
    
    def on_modified(self, event: FileSystemEvent) -> None:
        """ファイル変更イベントのハンドラー"""
        logger.debug(f"File modified: {event.src_path}")
        self._process_event(event)
    
    def on_deleted(self, event: FileSystemEvent) -> None:
        """ファイル削除イベントのハンドラー"""
        logger.debug(f"File deleted: {event.src_path}")
        self._process_event(event)
    
    def on_moved(self, event: FileSystemEvent) -> None:
        """ファイル移動イベントのハンドラー"""
        logger.debug(f"File moved: from {event.src_path} to {event.dest_path}")
        self._process_event(event)

class FileWatcher:
    """ファイルシステムの変更を監視するクラス"""
    
    def __init__(self, 
                 watched_dirs: List[str], 
                 event_queue: queue.Queue,
                 ignored_patterns: Optional[List[str]] = None,
                 supported_extensions: Optional[Set[str]] = None,
                 recursive: bool = True,
                 debounce_seconds: float = 2.0):
        """
        初期化
        
        Args:
            watched_dirs: 監視するディレクトリのリスト
            event_queue: イベント辞書を格納するキュー
            ignored_patterns: 無視するファイルパターンのリスト (例: ["temp_", ".tmp"])
            supported_extensions: サポートするファイル拡張子のセット。Noneの場合、configから読み込む。
            recursive: サブディレクトリも再帰的に監視するかどうか
            debounce_seconds: イベントのデバウンス時間（秒）
        """
        self.watched_dirs = [os.path.abspath(d) for d in watched_dirs]
        self.event_queue = event_queue
        self.ignored_patterns = ignored_patterns or []
        # The supported_extensions logic is now primarily in DocumentEventHandler.
        # FileWatcher's own self.supported_extensions is mainly for record-keeping or if create_file_watcher needs it.
        # DocumentEventHandler will handle the fallback chain for its own instance of supported_extensions.
        self.supported_extensions_arg = supported_extensions # Keep the argument for clarity if needed by create_file_watcher
        self.recursive = recursive
        self.debounce_seconds = debounce_seconds
        
        self.observer = Observer()
        self.handler = DocumentEventHandler( 
            event_queue=self.event_queue,
            ignored_patterns=self.ignored_patterns,
            supported_extensions=supported_extensions, # Pass the argument directly to the handler
            debounce_seconds=self.debounce_seconds
        )
        
        self._is_running = False
    
    def start(self) -> None:
        """ファイル監視を開始"""
        if self._is_running:
            logger.warning("File watcher is already running")
            return
            
        # 各ディレクトリに対して監視を設定
        for directory in self.watched_dirs:
            if not os.path.exists(directory):
                logger.warning(f"Directory does not exist: {directory}")
                continue
                
            logger.info(f"Starting watch on directory: {directory}")
            self.observer.schedule(self.handler, directory, recursive=self.recursive)
        
        # 監視開始
        self.observer.start()
        self._is_running = True
        logger.info("File watcher started")
    
    def stop(self) -> None:
        """ファイル監視を停止"""
        if not self._is_running:
            logger.info("File watcher is not running.")
            return
        
        logger.info("Stopping file watcher...")
        self.observer.stop()
        self.observer.join() # Wait for observer thread to finish
        
        # Cancel any pending debounce timer in the handler
        with self.handler.timer_lock:
            if self.handler.timer:
                self.handler.timer.cancel()
                self.handler.timer = None
        
        self._is_running = False
        logger.info("File watcher stopped.")
    
    def is_running(self) -> bool:
        """監視が実行中かどうかを返す"""
        return self._is_running


def create_file_watcher(watched_dirs: List[str], 
                       event_queue: queue.Queue, 
                       ignored_patterns: Optional[List[str]] = None,
                       supported_extensions: Optional[Set[str]] = None,
                       recursive: bool = True,
                       debounce_seconds: float = 2.0) -> FileWatcher:
    """
    FileWatcherのインスタンスを作成
    
    Args:
        watched_dirs: 監視するディレクトリのリスト
        event_queue: イベント辞書を格納するキュー
        ignored_patterns: 無視するパターンのリスト
        supported_extensions: サポートするファイル拡張子のセット。Noneの場合、configから読み込む。
        recursive: サブディレクトリも再帰的に監視するかどうか
        debounce_seconds: イベントのデバウンス時間（秒）
        
    Returns:
        FileWatcherのインスタンス
    """
    # The supported_extensions argument is passed directly to FileWatcher,
    # which then passes it to DocumentEventHandler.
    # DocumentEventHandler now contains the robust fallback logic.
    # No need for complex fallback here in create_file_watcher itself for this specific parameter.
    return FileWatcher(
        watched_dirs=watched_dirs,
        event_queue=event_queue,
        ignored_patterns=ignored_patterns,
        supported_extensions=supported_extensions, # Pass through; handler will use its fallback
        recursive=recursive,
        debounce_seconds=debounce_seconds
    )
