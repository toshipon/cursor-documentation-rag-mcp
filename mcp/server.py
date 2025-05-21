import os
import sys
import logging
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from cachetools import TTLCache, LRUCache
import threading

# プロジェクトのルートディレクトリをシステムパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from vectorize.embeddings import PLaMoEmbedder, DummyEmbedder
from db.vector_store import VectorStore

# ロギング設定
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# バッチサイズの設定
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "32"))

# キャッシュの設定
query_cache = TTLCache(maxsize=1000, ttl=600)  # 10分間のTTLで1000件までキャッシュ
embedding_cache = LRUCache(maxsize=500)  # 500件までのLRUキャッシュ

# スレッドローカルストレージ
thread_local = threading.local()

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
    cache: bool = Field(True, description="キャッシュを使用するかどうか")

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
    cached: bool = Field(False, description="キャッシュからの結果かどうか")

class BatchQueryRequest(BaseModel):
    """バッチ検索クエリリクエスト"""
    queries: List[str] = Field(..., description="検索クエリテキストのリスト")
    top_k: int = Field(5, description="各クエリで返却する結果数", ge=1, le=50)
    filters: Optional[Dict[str, Any]] = Field(None, description="フィルタリング条件")
    cache: bool = Field(True, description="キャッシュを使用するかどうか")

class BatchQueryResponse(BaseModel):
    """バッチ検索レスポンス"""
    results: List[List[QueryResult]] = Field(..., description="クエリごとの検索結果リスト")
    total: int = Field(..., description="検索結果の総数")
    query_time_ms: float = Field(..., description="クエリ処理時間（ミリ秒）")
    cached_count: int = Field(0, description="キャッシュから取得した結果の数")

# グローバル変数
embedder = None
vector_store = None

def get_embedder():
    """埋め込みモデルの初期化（スレッドセーフ対応）"""
    global embedder
    
    # すでにグローバルに初期化されている場合はそれを返す
    if embedder is not None:
        return embedder
        
    # スレッドローカルにない場合は初期化
    if not hasattr(thread_local, 'embedder'):
        try:
            logger.info("Initializing PLaMo-Embedding-1B model")
            thread_local.embedder = PLaMoEmbedder(model_path=config.EMBEDDING_MODEL_PATH)
        except Exception as e:
            logger.error(f"Error loading PLaMo model: {e}. Falling back to dummy embedder.")
            thread_local.embedder = DummyEmbedder()
            
        # 最初のスレッドで初期化したらグローバル変数にも設定
        if embedder is None:
            embedder = thread_local.embedder
            
    return thread_local.embedder

def get_vector_store():
    """ベクトルストアの初期化（スレッドセーフ対応）"""
    global vector_store
    
    # すでにグローバルに初期化されている場合はそれを返す
    if vector_store is not None:
        return vector_store
    
    # スレッドローカルにない場合は初期化
    if not hasattr(thread_local, 'vector_store'):
        logger.info(f"Initializing vector store at {config.VECTOR_DB_PATH}")
        os.makedirs(os.path.dirname(config.VECTOR_DB_PATH), exist_ok=True)
        thread_local.vector_store = VectorStore(config.VECTOR_DB_PATH)
        
        # 最初のスレッドで初期化したらグローバル変数にも設定
        if vector_store is None:
            vector_store = thread_local.vector_store
    
    return thread_local.vector_store

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
    stats["cache_info"] = {
        "query_cache_size": len(query_cache),
        "query_cache_capacity": query_cache.maxsize,
        "embedding_cache_size": len(embedding_cache),
        "embedding_cache_capacity": embedding_cache.maxsize
    }
    return stats

@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store),
    background_tasks: BackgroundTasks = None
):
    """
    テキスト検索エンドポイント
    
    Args:
        request: 検索クエリリクエスト
        
    Returns:
        検索結果レスポンス
    """
    start_time = time.time()
    
    # キャッシュキーの生成
    cache_key = f"{request.query}:{request.top_k}:{request.filters}"
    
    # キャッシュから結果を取得
    if request.cache and cache_key in query_cache:
        logger.debug(f"Cache hit for query: {request.query}")
        cached_result = query_cache[cache_key]
        cached_result["cached"] = True
        cached_result["query_time_ms"] = (time.time() - start_time) * 1000
        return QueryResponse(**cached_result)
    
    try:
        # クエリベクトルのキャッシュをチェック
        if request.query in embedding_cache:
            query_vector = embedding_cache[request.query]
        else:
            # クエリテキストを埋め込みベクトルに変換
            query_vector = embedder.embed(request.query)
            # キャッシュに保存
            if request.cache:
                embedding_cache[request.query] = query_vector
        
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
        
        response_data = {
            "results": results,
            "total": len(results),
            "query_time_ms": query_time_ms,
            "cached": False
        }
        
        # 結果をキャッシュに保存（バックグラウンドで）
        if request.cache and background_tasks:
            background_tasks.add_task(lambda: query_cache.update({cache_key: response_data}))
        elif request.cache:
            query_cache[cache_key] = response_data
        
        return QueryResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")

@app.post("/batch_query", response_model=BatchQueryResponse)
async def batch_query(
    request: BatchQueryRequest,
    embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    バッチテキスト検索エンドポイント
    
    Args:
        request: バッチ検索クエリリクエスト
        
    Returns:
        バッチ検索結果レスポンス
    """
    start_time = time.time()
    
    if len(request.queries) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size exceeds maximum allowed size of {MAX_BATCH_SIZE}"
        )
    
    try:
        # 結果とキャッシュカウンターの初期化
        all_results = []
        cached_count = 0
        
        # バッチ内のキャッシュミスしたクエリを収集
        cache_miss_queries = []
        cache_miss_indices = []
        
        # まずキャッシュをチェック
        if request.cache:
            for i, query in enumerate(request.queries):
                cache_key = f"{query}:{request.top_k}:{request.filters}"
                if cache_key in query_cache:
                    # キャッシュヒット
                    cached_result = query_cache[cache_key]
                    all_results.append(cached_result["results"])
                    cached_count += 1
                else:
                    # キャッシュミス
                    all_results.append(None)  # プレースホルダー
                    cache_miss_queries.append(query)
                    cache_miss_indices.append(i)
        else:
            # キャッシュを使用しない場合は全クエリをキャッシュミスとして扱う
            cache_miss_queries = request.queries
            cache_miss_indices = list(range(len(request.queries)))
            all_results = [None] * len(request.queries)  # すべてプレースホルダー
        
        # キャッシュミスしたクエリがある場合は処理
        if cache_miss_queries:
            # クエリベクトルのバッチ作成
            # 既存のembedding_cacheから取得できるものは取得
            vectors_to_embed = []
            query_vectors = []
            for query in cache_miss_queries:
                if query in embedding_cache:
                    query_vectors.append(embedding_cache[query])
                else:
                    vectors_to_embed.append(query)
            
            # 必要であれば新たな埋め込みを作成
            if vectors_to_embed:
                new_vectors = embedder.embed_batch(vectors_to_embed)
                
                # 新しい埋め込みをキャッシュに追加
                if request.cache:
                    for q, v in zip(vectors_to_embed, new_vectors):
                        embedding_cache[q] = v
                
                # 全ベクトルを順番通りに組み立て
                vectors_idx = 0
                for i, query in enumerate(cache_miss_queries):
                    if query in embedding_cache:
                        query_vectors.append(embedding_cache[query])
                    else:
                        query_vectors.append(new_vectors[vectors_idx])
                        vectors_idx += 1
            
            # バッチ検索を実行
            batch_results = vector_store.batch_similarity_search(
                query_vectors=query_vectors,
                top_k=request.top_k,
                filter_criteria=request.filters
            )
            
            # 結果を整形してキャッシュに追加
            for i, (query, results) in enumerate(zip(cache_miss_queries, batch_results)):
                formatted_results = []
                for result in results:
                    formatted_results.append(QueryResult(
                        content=result["content"],
                        metadata=result["metadata"],
                        score=float(result["score"]) if "score" in result else 0.0
                    ))
                
                # 元の順序で結果を保存
                original_idx = cache_miss_indices[i]
                all_results[original_idx] = formatted_results
                
                # キャッシュに保存
                if request.cache:
                    cache_key = f"{query}:{request.top_k}:{request.filters}"
                    query_cache[cache_key] = {
                        "results": formatted_results,
                        "total": len(formatted_results),
                        "query_time_ms": 0,  # キャッシュからは処理時間は意味がないので0
                        "cached": False
                    }
        
        # 処理時間を計算（ミリ秒単位）
        query_time_ms = (time.time() - start_time) * 1000
        
        total_results = sum(len(results) for results in all_results if results is not None)
        
        return BatchQueryResponse(
            results=all_results,
            total=total_results,
            query_time_ms=query_time_ms,
            cached_count=cached_count
        )
        
    except Exception as e:
        logger.error(f"Error processing batch query: {e}")
        raise HTTPException(status_code=500, detail=f"Batch query processing error: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """グローバル例外ハンドラ"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )