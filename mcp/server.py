import os
import sys
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import time

# プロジェクトのルートディレクトリをシステムパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from vectorize.embeddings import PLaMoEmbedder, DummyEmbedder
from db.vector_store import VectorStore

# ロギング設定
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Memory Context Provider (MCP) API",
    description="ドキュメント検索 API",
    version="0.1.0",
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# リクエスト・レスポンスモデル
class QueryRequest(BaseModel):
    """検索クエリリクエスト"""
    query: str = Field(..., description="検索クエリテキスト")
    top_k: int = Field(5, description="返却する結果数", ge=1, le=50)
    filters: Optional[Dict[str, Any]] = Field(None, description="フィルタリング条件")

class QueryResult(BaseModel):
    """検索結果"""
    content: str = Field(..., description="テキスト内容")
    metadata: Dict[str, Any] = Field(..., description="メタデータ")
    score: float = Field(..., description="類似度スコア", ge=0.0, le=1.0)

class QueryResponse(BaseModel):
    """検索レスポンス"""
    results: List[QueryResult] = Field(..., description="検索結果リスト")
    total: int = Field(..., description="検索結果の総数")
    query_time_ms: float = Field(..., description="クエリ処理時間（ミリ秒）")

# グローバル変数
embedder = None
vector_store = None

def get_embedder():
    """埋め込みモデルの初期化"""
    global embedder
    if embedder is None:
        try:
            logger.info("Initializing PLaMo-Embedding-1B model")
            embedder = PLaMoEmbedder(model_path=config.EMBEDDING_MODEL_PATH)
        except Exception as e:
            logger.error(f"Error loading PLaMo model: {e}. Falling back to dummy embedder.")
            embedder = DummyEmbedder()
    return embedder

def get_vector_store():
    """ベクトルストアの初期化"""
    global vector_store
    if vector_store is None:
        logger.info(f"Initializing vector store at {config.VECTOR_DB_PATH}")
        os.makedirs(os.path.dirname(config.VECTOR_DB_PATH), exist_ok=True)
        vector_store = VectorStore(config.VECTOR_DB_PATH)
    return vector_store

@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の初期化"""
    logger.info("Starting MCP server")
    # 事前に埋め込みモデルとベクトルストアを初期化
    get_embedder()
    get_vector_store()

@app.on_event("shutdown")
async def shutdown_event():
    """アプリケーション終了時の処理"""
    logger.info("Shutting down MCP server")
    global vector_store
    if vector_store:
        vector_store.close()

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "name": "Memory Context Provider API",
        "version": "0.1.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    return {"status": "healthy"}

@app.get("/stats")
async def get_stats(vector_store: VectorStore = Depends(get_vector_store)):
    """統計情報取得エンドポイント"""
    stats = vector_store.get_stats()
    return stats

@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    テキスト検索エンドポイント
    
    Args:
        request: 検索クエリリクエスト
        
    Returns:
        検索結果レスポンス
    """
    start_time = time.time()
    
    try:
        # クエリテキストを埋め込みベクトルに変換
        query_vector = embedder.embed(request.query)
        
        # ベクトル検索を実行
        search_results = vector_store.similarity_search(
            query_vector=query_vector,
            top_k=request.top_k,
            filter_criteria=request.filters
        )
        
        # レスポンスを整形
        results = []
        for result in search_results:
            results.append(QueryResult(
                content=result["content"],
                metadata=result["metadata"],
                score=float(result["score"]) if "score" in result else 0.0
            ))
        
        # 処理時間を計算（ミリ秒単位）
        query_time_ms = (time.time() - start_time) * 1000
        
        return QueryResponse(
            results=results,
            total=len(results),
            query_time_ms=query_time_ms
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """グローバル例外ハンドラ"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )