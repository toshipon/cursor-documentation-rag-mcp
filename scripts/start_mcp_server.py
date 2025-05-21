from fastapi import FastAPI, Request
import uvicorn
from vectorize.embeddings import DummyEmbedder
from db.vector_store import VectorStore
import os

app = FastAPI()
DB_PATH = os.environ.get("VECTOR_DB_PATH", "vector_store/vector_store.db")
vector_store = VectorStore(DB_PATH)
embedder = DummyEmbedder()

@app.post("/query")
async def query_endpoint(request: Request):
    data = await request.json()
    query = data.get("query", "")
    top_k = data.get("top_k", 5)
    # フィルタは未実装
    q_vec = embedder.embed(query)
    results = vector_store.similarity_search(q_vec, top_k=top_k)
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)