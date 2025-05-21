import os
import sys
import argparse
import logging
import uvicorn

# プロジェクトのルートディレクトリをシステムパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# ロギング設定
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

def main():
    """
    Memory Context Provider (MCP) サーバーを起動する
    """
    parser = argparse.ArgumentParser(description="MCP サーバーを起動します")
    parser.add_argument("--host", default="0.0.0.0", help="ホストアドレス (デフォルト: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="ポート番号 (デフォルト: 8000)")
    parser.add_argument("--reload", action="store_true", help="ファイル変更時に自動リロード（開発用）")
    parser.add_argument("--debug", action="store_true", help="デバッグモードで実行")
    args = parser.parse_args()

    # サーバー設定
    host = args.host
    port = args.port
    reload = args.reload
    log_level = "debug" if args.debug else "info"
    
    # ログメッセージ
    logger.info(f"Starting MCP server at http://{host}:{port}")
    logger.info(f"Vector DB path: {config.VECTOR_DB_PATH}")
    logger.info(f"Embedding model path: {config.EMBEDDING_MODEL_PATH}")
    
    # MCPサーバーを起動（mcp.server:appアプリケーションを指定）
    uvicorn.run(
        "mcp.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level
    )

if __name__ == "__main__":
    main()