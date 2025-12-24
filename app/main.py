"""
RAG API - Semantic Search over Creative Works
A portfolio project demonstrating embeddings, vector search, and caching.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import httpx
import json
import hashlib
import redis
from typing import Optional
import numpy as np

from app.config import get_settings, Settings


# Initialize FastAPI
app = FastAPI(
    title="RAG API",
    description="Semantic search API for creative works",
    version="0.1.0"
)

# CORS - adjust origins for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redis connection (lazy initialization)
_redis_client = None

def get_redis(settings: Settings = Depends(get_settings)) -> Optional[redis.Redis]:
    """Get Redis client, returns None if not configured."""
    global _redis_client
    
    if not settings.upstash_redis_url:
        return None
    
    if _redis_client is None:
        _redis_client = redis.from_url(
            settings.upstash_redis_url,
            password=settings.upstash_redis_token,
            decode_responses=True
        )
    return _redis_client


# Request/Response Models
class TextInput(BaseModel):
    text: str
    
class EmbeddingResponse(BaseModel):
    embedding: list[float]
    cached: bool = False
    
class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class SearchResult(BaseModel):
    text: str
    score: float
    
class SearchResponse(BaseModel):
    results: list[SearchResult]
    query_cached: bool = False


# In-memory document store
DOCUMENTS = [
    # Nok'so
    "This forsaken land was my home, and I didn't know to call it Forsaken, until someone else taught me the word.",
    "Today it is a distant memory, a fractured dream with a sunburn.",
    "Fields stitched together with rusting oil wells, jackhammers bowing up and down like tired animals. The desert didn't have weather; it had moods. Mostly bad ones.",
    "My favorite spot was beside a man-made hill— industrial waste piled high and left to calcify in the shape of a secret. Kids claim these things as kingdoms.",
    "I looked up once, straight into the sun— stupid, reckless, customary. My eyes burned, but through the white heat I saw him.",
    "The Falcon— fast enough that the world forgot its frame rate. Mysterious in the way a shadow is mysterious when it doesn't match the thing that cast it.",
    "Nok'so… The name landed like heat behind the eyes. Hard to say what swaps places with you when the desert starts humming.",
    "An old El Camino erupted over the waste-hill like a rusted comet with a bad attitude. It crashed exactly where I'd been standing.",
    "The sort of moment that should have killed me, but didn't. The sort of moment that belongs to myth long before you know you're living one.",
    "When the dust settled, the Falcon was gone. Or maybe just higher up, where children can't follow.",
    "I kept running long after my lungs quit negotiating. Some instincts stay with you. Some debts, too.",
    
    # Character portals
    "She crossed from Los Angeles to New York, mythologies trailing behind her.",
    "Victorian parlors hide secrets in their wallpaper, whispered conversations.",
    "The fox spirit moves through city streets, ancient trickster in modern skin.",
    "Seagulls circle the harbor, Furies reimagined with salt-white wings.",
    "A cavalry officer's letters home, ink fading but honor preserved.",
    "Cantonese syllables drift through 1840s air, a woman finding her voice.",
    "Poetry lives in the spaces between words, incantations for those who cross.",
]

# Cache for document embeddings
_doc_embeddings = None


async def get_embedding(text: str, settings: Settings) -> list[float]:
    """Get embedding from OpenAI API."""
    
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=500, 
            detail="OpenAI API key not configured"
        )
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "text-embedding-3-small",
                "input": text
            },
            timeout=30.0
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"OpenAI API error: {response.text}"
            )
        
        data = response.json()
        return data["data"][0]["embedding"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a_np = np.array(a)
    b_np = np.array(b)
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))


def cache_key(text: str) -> str:
    """Generate a cache key from text."""
    return f"emb:{hashlib.md5(text.encode()).hexdigest()}"


# Endpoints

# Get the directory where this file lives
BASE_DIR = Path(__file__).resolve().parent

@app.get("/")
async def root():
    """Serve the frontend."""
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api")
async def api_info():
    """API info endpoint."""
    return {
        "name": "RAG API",
        "version": "0.1.0",
        "endpoints": ["/health", "/embed", "/search", "/documents"]
    }


@app.get("/health")
async def health_check(settings: Settings = Depends(get_settings)):
    """Health check endpoint."""
    
    redis_status = "not configured"
    redis_client = None
    
    if settings.upstash_redis_url:
        try:
            redis_client = get_redis(settings)
            redis_client.ping()
            redis_status = "connected"
        except Exception as e:
            redis_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "services": {
            "redis": redis_status,
            "openai": "configured" if settings.openai_api_key else "not configured"
        }
    }


@app.get("/documents")
async def list_documents():
    """List all documents in the store."""
    return {
        "count": len(DOCUMENTS),
        "documents": DOCUMENTS
    }


@app.post("/embed", response_model=EmbeddingResponse)
async def embed_text(
    input: TextInput,
    settings: Settings = Depends(get_settings)
):
    """Generate embedding for input text with optional caching."""
    
    # Check cache first
    redis_client = get_redis(settings) if settings.upstash_redis_url else None
    key = cache_key(input.text)
    
    if redis_client:
        try:
            cached = redis_client.get(key)
            if cached:
                return EmbeddingResponse(
                    embedding=json.loads(cached),
                    cached=True
                )
        except Exception:
            pass  # Cache miss or error, continue to API
    
    # Get fresh embedding
    embedding = await get_embedding(input.text, settings)
    
    # Cache it
    if redis_client:
        try:
            redis_client.setex(key, 86400, json.dumps(embedding))  # 24hr TTL
        except Exception:
            pass  # Caching failed, not critical
    
    return EmbeddingResponse(embedding=embedding, cached=False)


@app.post("/search", response_model=SearchResponse)
async def semantic_search(
    request: SearchRequest,
    settings: Settings = Depends(get_settings)
):
    """Search documents by semantic similarity."""
    global _doc_embeddings
    
    # Build document embeddings if not cached
    if _doc_embeddings is None:
        _doc_embeddings = []
        for doc in DOCUMENTS:
            emb = await get_embedding(doc, settings)
            _doc_embeddings.append(emb)
    
    # Get query embedding (with caching)
    redis_client = get_redis(settings) if settings.upstash_redis_url else None
    key = cache_key(request.query)
    query_cached = False
    
    if redis_client:
        try:
            cached = redis_client.get(key)
            if cached:
                query_embedding = json.loads(cached)
                query_cached = True
            else:
                query_embedding = await get_embedding(request.query, settings)
                redis_client.setex(key, 86400, json.dumps(query_embedding))
        except Exception:
            query_embedding = await get_embedding(request.query, settings)
    else:
        query_embedding = await get_embedding(request.query, settings)
    
    # Calculate similarities
    scores = []
    for i, doc_emb in enumerate(_doc_embeddings):
        score = cosine_similarity(query_embedding, doc_emb)
        scores.append((DOCUMENTS[i], score))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top_k results
    results = [
        SearchResult(text=text, score=round(score, 4))
        for text, score in scores[:request.top_k]
    ]
    
    return SearchResponse(results=results, query_cached=query_cached)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
