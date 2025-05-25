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
from db.qdrant_store import QdrantVectorStore
from mcp.monitoring import MCPMonitor

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

# モニタリングの設定
monitor = MCPMonitor(app)

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

class HybridSearchRequest(BaseModel):
    """ハイブリッド検索リクエスト"""
    query: str = Field(..., description="検索クエリテキスト")
    top_k: int = Field(5, description="返却する結果数", ge=1, le=50)
    filters: Optional[Dict[str, Any]] = Field(None, description="フィルタリング条件")
    cache: bool = Field(True, description="キャッシュを使用するかどうか")
    vector_weight: float = Field(0.7, description="ベクトル検索の重み", ge=0.0, le=1.0)
    text_weight: float = Field(0.3, description="テキスト検索の重み", ge=0.0, le=1.0)

class KeywordSearchRequest(BaseModel):
    """キーワード検索リクエスト"""
    keyword: str = Field(..., description="検索キーワード")
    top_k: int = Field(5, description="返却する結果数", ge=1, le=50)
    filters: Optional[Dict[str, Any]] = Field(None, description="フィルタリング条件")
    cache: bool = Field(True, description="キャッシュを使用するかどうか")
    match_type: str = Field("contains", description="一致タイプ (contains, exact, starts_with, ends_with)")

# ハイブリッド検索キャッシュ
hybrid_search_cache = TTLCache(maxsize=500, ttl=600)  # 10分間のTTLで500件までキャッシュ

# キーワード検索キャッシュ
keyword_search_cache = TTLCache(maxsize=500, ttl=600)  # 10分間のTTLで500件までキャッシュ

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
            # 本番環境の場合はPLaMoEmbedderを使用
            if os.environ.get("ENV") == "test":
                logger.info("Using DummyEmbedder for testing")
                thread_local.embedder = DummyEmbedder()
            else:
                logger.info(f"Initializing PLaMoEmbedder with model at {config.EMBEDDING_MODEL_PATH}")
                thread_local.embedder = PLaMoEmbedder(
                    model_path=config.EMBEDDING_MODEL_PATH, 
                    device=os.environ.get("DEVICE", None)
                )
            
            # 最初のスレッドで初期化したらグローバル変数にも設定
            if embedder is None:
                embedder = thread_local.embedder
                
        except Exception as e:
            logger.error(f"Error initializing embedder: {e}")
            raise
    
    return thread_local.embedder

def get_vector_store():
    """ベクトルストアの初期化（スレッドセーフ対応）"""
    global vector_store
    
    # すでにグローバルに初期化されている場合はそれを返す
    if vector_store is not None:
        return vector_store
    
    # スレッドローカルにない場合は初期化
    if not hasattr(thread_local, 'vector_store'):
        try:
            vector_store_type = os.getenv("VECTOR_STORE_TYPE", "sqlite")
            
            if vector_store_type.lower() == "qdrant":
                logger.info("Initializing Qdrant vector store for MCP server")
                qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6334")
                collection_name = os.getenv("QDRANT_COLLECTION", "documents")
                
                # モデルの埋め込み次元数を取得
                embedder = get_embedder()
                vector_dimension = embedder.get_dimension() if hasattr(embedder, 'get_dimension') else 512
                
                thread_local.vector_store = QdrantVectorStore(
                    url=qdrant_url,
                    collection_name=collection_name,
                    vector_dimension=vector_dimension
                )
            else:
                logger.info(f"Initializing SQLite VectorStore with DB path: {config.VECTOR_DB_PATH}")
                # モデルの埋め込み次元数を取得
                embedder = get_embedder()
                vector_dimension = embedder.get_dimension() if hasattr(embedder, 'get_dimension') else 2048
                
                thread_local.vector_store = VectorStore(
                    db_path=config.VECTOR_DB_PATH,
                    vector_dimension=vector_dimension
                )
            
            # 最初のスレッドで初期化したらグローバル変数にも設定
            if vector_store is None:
                vector_store = thread_local.vector_store
                
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    # Set global variable
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

@app.get("/metrics")
async def get_metrics():
    """
    Prometheusメトリクス取得エンドポイント
        
    Returns:
        Prometheusメトリクス
    """
    return JSONResponse(
        content=monitor.get_metrics(),
        media_type="text/plain"
    )

@app.get("/stats")
async def get_stats(
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    統計情報取得エンドポイント
    
    Args:
        vector_store: ベクトルストア
        
    Returns:
        統計情報
    """
    stats = vector_store.get_stats()
    stats["cache_info"] = {
        "query_cache_size": len(query_cache),
        "query_cache_capacity": query_cache.maxsize,
        "embedding_cache_size": len(embedding_cache),
        "embedding_cache_capacity": embedding_cache.maxsize,
        "hybrid_search_cache_size": len(hybrid_search_cache),
        "hybrid_search_cache_capacity": hybrid_search_cache.maxsize,
        "keyword_search_cache_size": len(keyword_search_cache),
        "keyword_search_cache_capacity": keyword_search_cache.maxsize
    }
    
    # モニタリング情報を追加
    stats["resource_stats"] = monitor.get_resource_stats()
    
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
        embedder: 埋め込みモデル
        vector_store: ベクトルストア
        background_tasks: バックグラウンドタスク
        
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
        
        # モニタリングのためのメトリクス記録
        monitor.record_query("single", cached_result["query_time_ms"] / 1000, cached=True)
        
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
            # Ensure mapping aligns with VectorStore output: result["text"] to QueryResult.content
            results.append(QueryResult(
                content=result.get("text", result.get("content", "")), # Prioritize "text", fallback to "content" if old format
                metadata=result.get("metadata", {}),
                score=float(result.get("score", 0.0)) 
            ))
        
        # 処理時間を計算（ミリ秒単位）
        query_time_ms = (time.time() - start_time) * 1000
        
        response_data = {
            "results": results,
            "total": len(results),
            "query_time_ms": query_time_ms,
            "cached": False
        }
        
        # モニタリングのためのメトリクス記録
        monitor.record_query("single", query_time_ms / 1000, cached=False)
        
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
        embedder: 埋め込みモデル
        vector_store: ベクトルストア
        
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
        
        # モニタリングのためのメトリクス記録
        monitor.record_query("batch", query_time_ms / 1000, cached=(cached_count > 0))
        
        return BatchQueryResponse(
            results=all_results,
            total=total_results,
            query_time_ms=query_time_ms,
            cached_count=cached_count
        )
        
    except Exception as e:
        logger.error(f"Error processing batch query: {e}")
        raise HTTPException(status_code=500, detail=f"Batch query processing error: {str(e)}")

@app.post("/hybrid_search", response_model=QueryResponse)
async def hybrid_search(
    request: HybridSearchRequest,
    embedder = Depends(get_embedder),
    vector_store: VectorStore = Depends(get_vector_store),
    background_tasks: BackgroundTasks = None
):
    """
    ハイブリッド検索エンドポイント（ベクトル類似度検索とテキスト検索を組み合わせる）
    
    Args:
        request: ハイブリッド検索リクエスト
        embedder: 埋め込みモデル
        vector_store: ベクトルストア
        background_tasks: バックグラウンドタスク
        
    Returns:
        検索結果レスポンス
    """
    start_time = time.time()
    
    # キャッシュキーの生成
    cache_key = f"{request.query}:{request.top_k}:{request.filters}:{request.vector_weight}:{request.text_weight}"
    
    # キャッシュから結果を取得
    if request.cache and cache_key in hybrid_search_cache:
        logger.debug(f"Cache hit for hybrid search: {request.query}")
        cached_result = hybrid_search_cache[cache_key]
        cached_result["cached"] = True
        cached_result["query_time_ms"] = (time.time() - start_time) * 1000
        
        # モニタリングのためのメトリクス記録
        monitor.record_query("hybrid", cached_result["query_time_ms"] / 1000, cached=True)
        
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
        
        # ハイブリッド検索を実行
        search_results = vector_store.hybrid_search(
            query_text=request.query,
            query_vector=query_vector,
            top_k=request.top_k,
            filter_criteria=request.filters,
            vector_weight=request.vector_weight,
            text_weight=request.text_weight
        )
        
        # レスポンスを整形
        results = []
        for result in search_results:
            results.append(QueryResult(
                content=result["content"],
                metadata=result["metadata"],
                score=float(result["score"])
            ))
        
        # 処理時間を計算（ミリ秒単位）
        query_time_ms = (time.time() - start_time) * 1000
        
        response_data = {
            "results": results,
            "total": len(results),
            "query_time_ms": query_time_ms,
            "cached": False
        }
        
        # モニタリングのためのメトリクス記録
        monitor.record_query("hybrid", query_time_ms / 1000, cached=False)
        
        # 結果をキャッシュに保存（バックグラウンドで）
        if request.cache and background_tasks:
            background_tasks.add_task(lambda: hybrid_search_cache.update({cache_key: response_data}))
        elif request.cache:
            hybrid_search_cache[cache_key] = response_data
        
        return QueryResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error processing hybrid search: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid search processing error: {str(e)}")

@app.post("/keyword_search", response_model=QueryResponse)
async def keyword_search(
    request: KeywordSearchRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    background_tasks: BackgroundTasks = None
):
    """
    キーワード検索エンドポイント（ベクトル検索を使用せず、テキスト検索のみ）
    
    Args:
        request: キーワード検索リクエスト
        vector_store: ベクトルストア
        background_tasks: バックグラウンドタスク
        
    Returns:
        検索結果レスポンス
    """
    start_time = time.time()
    
    # キャッシュキーの生成
    cache_key = f"{request.keyword}:{request.top_k}:{request.filters}:{request.match_type}"
    
    # キャッシュから結果を取得
    if request.cache and cache_key in keyword_search_cache:
        logger.debug(f"Cache hit for keyword search: {request.keyword}")
        cached_result = keyword_search_cache[cache_key]
        cached_result["cached"] = True
        cached_result["query_time_ms"] = (time.time() - start_time) * 1000
        
        # モニタリングのためのメトリクス記録
        monitor.record_query("keyword", cached_result["query_time_ms"] / 1000, cached=True)
        
        return QueryResponse(**cached_result)
    
    try:
        # キーワード検索を実行
        search_results = vector_store.keyword_search(
            keyword=request.keyword,
            top_k=request.top_k,
            filter_criteria=request.filters,
            match_type=request.match_type
        )
        
        # レスポンスを整形
        results = []
        for result in search_results:
            results.append(QueryResult(
                content=result["content"],
                metadata=result["metadata"],
                score=float(result["score"])
            ))
        
        # 処理時間を計算（ミリ秒単位）
        query_time_ms = (time.time() - start_time) * 1000
        
        response_data = {
            "results": results,
            "total": len(results),
            "query_time_ms": query_time_ms,
            "cached": False
        }
        
        # モニタリングのためのメトリクス記録
        monitor.record_query("keyword", query_time_ms / 1000, cached=False)
        
        # 結果をキャッシュに保存（バックグラウンドで）
        if request.cache and background_tasks:
            background_tasks.add_task(lambda: keyword_search_cache.update({cache_key: response_data}))
        elif request.cache:
            keyword_search_cache[cache_key] = response_data
        
        return QueryResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error processing keyword search: {e}")
        raise HTTPException(status_code=500, detail=f"Keyword search processing error: {str(e)}")

@app.post("/maintenance/optimize_db")
async def optimize_database(
    vector_store: VectorStore = Depends(get_vector_store)
):
    """
    データベース最適化エンドポイント
    
    Args:
        vector_store: ベクトルストア
        
    Returns:
        操作結果
    """
    try:
        # データベース最適化を実行
        vector_store.optimize_database()
        
        return {
            "status": "success",
            "message": "Database optimization completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error optimizing database: {e}")
        raise HTTPException(status_code=500, detail=f"Database optimization error: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """グローバル例外ハンドラ"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )