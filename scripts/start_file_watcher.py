import os
import sys
import argparse
import logging
import time
import queue
import signal
from typing import List

# プロジェクトのルートディレクトリをシステムパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from workers.file_watcher import FileWatcher, create_file_watcher
from workers.vectorization_worker import VectorizationWorker, create_vectorization_worker

# ロギング設定
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# グローバル変数
file_watcher = None
vectorization_worker = None

def shutdown(signum, frame):
    """シャットダウンハンドラ"""
    global file_watcher, vectorization_worker
    
    logger.info("Shutting down...")
    
    # 各コンポーネントを停止
    if file_watcher:
        file_watcher.stop()
    
    if vectorization_worker:
        vectorization_worker.stop()
    
    logger.info("Shutdown complete")
    sys.exit(0)

def main():
    """メイン実行関数"""
    global file_watcher, vectorization_worker
    
    parser = argparse.ArgumentParser(description="ファイル変更の監視と自動ベクトル化を行います")
    parser.add_argument("--watch_dirs", nargs='+', required=True, help="監視するディレクトリ（複数指定可）")
    parser.add_argument("--vector_db", default=config.VECTOR_DB_PATH, help=f"ベクトルDBのパス (デフォルト: {config.VECTOR_DB_PATH})")
    parser.add_argument("--use_dummy", action="store_true", help="ダミー埋め込みモデルを使用（テスト用）")
    parser.add_argument("--ignored_patterns", nargs='+', default=[], help="無視するファイルパターン（複数指定可）")
    parser.add_argument("--non_recursive", action="store_true", help="サブディレクトリを再帰的に監視しない")
    args = parser.parse_args()
    
    # 監視対象ディレクトリを絶対パスに変換
    watch_dirs = [os.path.abspath(d) for d in args.watch_dirs]
    
    # 各ディレクトリの存在チェック
    for directory in watch_dirs:
        if not os.path.exists(directory):
            logger.error(f"Directory does not exist: {directory}")
            return 1
    
    # ベクトルDBのディレクトリが存在しなければ作成
    os.makedirs(os.path.dirname(args.vector_db), exist_ok=True)
    
    logger.info(f"Watching directories: {', '.join(watch_dirs)}")
    logger.info(f"Vector DB path: {args.vector_db}")
    logger.info(f"Using dummy embedder: {args.use_dummy}")
    
    try:
        # イベントキューを作成
        event_queue = queue.Queue()
        
        # シグナルハンドラを設定
        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)
        
        # ファイル監視ワーカーを作成して起動
        file_watcher = create_file_watcher(
            watched_dirs=watch_dirs,
            event_queue=event_queue,
            ignored_patterns=args.ignored_patterns,
            recursive=not args.non_recursive
        )
        file_watcher.start()
        
        # ベクトル化ワーカーを作成して起動
        vectorization_worker = create_vectorization_worker(
            event_queue=event_queue,
            vector_store_path=args.vector_db,
            use_dummy_embedder=args.use_dummy
        )
        vectorization_worker.start()
        
        logger.info("File watcher and vectorization worker started")
        logger.info("Press Ctrl+C to stop")
        
        # メインスレッドは終了せずに待機
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        shutdown(None, None)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
