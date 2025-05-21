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

# ベクトル化対象のファイル拡張子
SUPPORTED_EXTENSIONS = {
    '.md', '.markdown',  # マークダウン
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.cs', '.go', '.rb', '.php', '.swift', '.kt', '.rs',  # コード
    '.pdf'  # PDF
}

class VectorizationEvent:
    """ベクトル化イベントを表すクラス"""
    
    def __init__(self, file_path: str, event_type: str, timestamp: float = None):
        """
        初期化
        
        Args:
            file_path: ファイルパス
            event_type: イベントタイプ ('created', 'modified', 'deleted')
            timestamp: イベント発生時刻（Unix時間）
        """
        self.file_path = file_path
        self.event_type = event_type
        self.timestamp = timestamp or time.time()
    
    def __str__(self) -> str:
        return f"VectorizationEvent({self.event_type}, {self.file_path}, {self.timestamp})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, VectorizationEvent):
            return False
        return self.file_path == other.file_path and self.event_type == other.event_type
    
    def __hash__(self) -> int:
        return hash((self.file_path, self.event_type))

class FileChangeHandler(FileSystemEventHandler):
    """ファイルシステムのイベントを処理するハンドラー"""
    
    def __init__(self, event_queue: queue.Queue, ignored_patterns: List[str] = None, 
                 debounce_seconds: float = 2.0):
        """
        初期化
        
        Args:
            event_queue: ベクトル化イベントを格納するキュー
            ignored_patterns: 無視するファイルパターンのリスト
            debounce_seconds: イベントのデバウンス時間（秒）
        """
        self.event_queue = event_queue
        self.ignored_patterns = ignored_patterns or []
        self.debounce_seconds = debounce_seconds
        self.last_events: Dict[str, float] = {}  # ファイルパスごとの最終イベント時刻
        self.pending_events: Set[VectorizationEvent] = set()  # 保留中のイベント
        
        # デバウンスタイマー
        self.timer = None
        self.timer_lock = threading.Lock()
    
    def _should_ignore(self, file_path: str) -> bool:
        """
        ファイルを無視すべきかどうかを判定
        
        Args:
            file_path: ファイルパス
            
        Returns:
            無視すべきならTrue、そうでなければFalse
        """
        # 隠しファイルを無視
        if os.path.basename(file_path).startswith('.'):
            return True
        
        # 特定のディレクトリを無視
        ignored_dirs = ['__pycache__', '.git', 'venv', 'env', '.venv', '.env']
        for ignored_dir in ignored_dirs:
            if f'/{ignored_dir}/' in file_path or file_path.endswith(f'/{ignored_dir}'):
                return True
        
        # 無視パターンをチェック
        for pattern in self.ignored_patterns:
            if pattern in file_path:
                return True
        
        # 拡張子をチェック
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return True
            
        return False
    
    def _process_event(self, event: FileSystemEvent) -> None:
        """
        watchdogイベントを処理
        
        Args:
            event: watchdogイベント
        """
        # ディレクトリの場合は無視
        if event.is_directory:
            return
            
        file_path = event.src_path
        
        # 無視すべきファイルは処理しない
        if self._should_ignore(file_path):
            return
            
        # イベントタイプを変換
        if event.event_type == 'created' or event.event_type == 'modified':
            event_type = 'modified'  # 作成と変更は同じように処理
        elif event.event_type == 'deleted':
            event_type = 'deleted'
        elif event.event_type == 'moved':
            # 移動元と移動先を別々に処理
            if hasattr(event, 'dest_path'):
                # 移動元のファイルを削除として処理
                self._add_event(VectorizationEvent(file_path, 'deleted'))
                
                # 移動先のファイルを作成として処理
                dest_path = event.dest_path
                if not self._should_ignore(dest_path):
                    self._add_event(VectorizationEvent(dest_path, 'modified'))
            return
        else:
            return  # その他のイベントは無視
            
        # イベントをキューに追加
        vectorization_event = VectorizationEvent(file_path, event_type)
        self._add_event(vectorization_event)
    
    def _add_event(self, event: VectorizationEvent) -> None:
        """
        イベントを保留リストに追加し、デバウンスタイマーを設定
        
        Args:
            event: ベクトル化イベント
        """
        # 同じファイルの既存のイベントを削除して新しいイベントに置き換え
        self.pending_events = {e for e in self.pending_events if e.file_path != event.file_path}
        self.pending_events.add(event)
        
        # 最終イベント時刻を更新
        self.last_events[event.file_path] = time.time()
        
        # デバウンスタイマーをリセット
        with self.timer_lock:
            if self.timer:
                self.timer.cancel()
            self.timer = threading.Timer(self.debounce_seconds, self._flush_events)
            self.timer.daemon = True
            self.timer.start()
    
    def _flush_events(self) -> None:
        """デバウンス期間が終了したら、保留中のすべてのイベントをキューに追加"""
        current_time = time.time()
        events_to_process = []
        
        # デバウンス期間が終了したイベントを見つける
        for event in list(self.pending_events):
            last_event_time = self.last_events.get(event.file_path, 0)
            if current_time - last_event_time >= self.debounce_seconds:
                events_to_process.append(event)
                self.pending_events.remove(event)
        
        # イベントをキューに追加
        for event in events_to_process:
            logger.info(f"Queueing {event}")
            self.event_queue.put(event)
        
        # まだ保留中のイベントがある場合は、タイマーを再設定
        if self.pending_events:
            with self.timer_lock:
                self.timer = threading.Timer(self.debounce_seconds, self._flush_events)
                self.timer.daemon = True
                self.timer.start()
    
    # イベントハンドラー（watchdogからのコールバック）
    def on_created(self, event: FileSystemEvent) -> None:
        """ファイル作成イベントのハンドラー"""
        self._process_event(event)
    
    def on_modified(self, event: FileSystemEvent) -> None:
        """ファイル変更イベントのハンドラー"""
        self._process_event(event)
    
    def on_deleted(self, event: FileSystemEvent) -> None:
        """ファイル削除イベントのハンドラー"""
        self._process_event(event)
    
    def on_moved(self, event: FileSystemEvent) -> None:
        """ファイル移動イベントのハンドラー"""
        self._process_event(event)

class FileWatcher:
    """ファイルシステムの変更を監視するクラス"""
    
    def __init__(self, 
                 watched_dirs: List[str], 
                 event_queue: queue.Queue,
                 ignored_patterns: List[str] = None,
                 recursive: bool = True,
                 debounce_seconds: float = 2.0):
        """
        初期化
        
        Args:
            watched_dirs: 監視するディレクトリのリスト
            event_queue: イベントを格納するキュー
            ignored_patterns: 無視するパターンのリスト
            recursive: サブディレクトリも再帰的に監視するかどうか
            debounce_seconds: イベントのデバウンス時間（秒）
        """
        self.watched_dirs = [os.path.abspath(d) for d in watched_dirs]
        self.event_queue = event_queue
        self.ignored_patterns = ignored_patterns or []
        self.recursive = recursive
        self.debounce_seconds = debounce_seconds
        
        self.observer = Observer()
        self.handler = FileChangeHandler(
            event_queue=self.event_queue,
            ignored_patterns=self.ignored_patterns,
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
            return
            
        self.observer.stop()
        self.observer.join()
        self._is_running = False
        logger.info("File watcher stopped")
    
    def is_running(self) -> bool:
        """監視が実行中かどうかを返す"""
        return self._is_running


def create_file_watcher(watched_dirs: List[str], 
                       event_queue: queue.Queue, 
                       ignored_patterns: List[str] = None,
                       recursive: bool = True) -> FileWatcher:
    """
    FileWatcherのインスタンスを作成
    
    Args:
        watched_dirs: 監視するディレクトリのリスト
        event_queue: イベントを格納するキュー
        ignored_patterns: 無視するパターンのリスト
        recursive: サブディレクトリも再帰的に監視するかどうか
        
    Returns:
        FileWatcherのインスタンス
    """
    return FileWatcher(
        watched_dirs=watched_dirs,
        event_queue=event_queue,
        ignored_patterns=ignored_patterns,
        recursive=recursive
    )
