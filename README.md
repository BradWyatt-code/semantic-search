# RAG API

A semantic search API demonstrating embeddings, vector similarity, and Redis caching. Built with FastAPI and designed for Railway deployment.

## Features

- **Semantic Search**: Find documents by meaning, not just keywords
- **Embedding Generation**: Convert text to vector representations
- **Redis Caching**: Reduce API costs by caching embeddings
- **Production Ready**: CORS, health checks, proper error handling

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│  FastAPI    │────▶│  OpenAI     │
│  (Frontend) │     │  (Railway)  │     │  Embeddings │
└─────────────┘     └──────┬──────┘     └─────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │   Upstash   │
                   │    Redis    │
                   └─────────────┘
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check with service status |
| GET | `/documents` | List all searchable documents |
| POST | `/embed` | Generate embedding for text |
| POST | `/search` | Semantic search over documents |

## Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Run locally
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## Deploy to Railway

1. Push this code to a GitHub repository
2. Go to [Railway](https://railway.app) and create a new project
3. Select "Deploy from GitHub repo"
4. Choose your repository
5. Add environment variables in Railway dashboard:
   - `OPENAI_API_KEY`
   - `UPSTASH_REDIS_URL` (optional)
   - `UPSTASH_REDIS_TOKEN` (optional)
6. Railway will auto-deploy on every push

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for embeddings |
| `UPSTASH_REDIS_URL` | No | Upstash Redis REST URL |
| `UPSTASH_REDIS_TOKEN` | No | Upstash Redis token |
| `DEBUG` | No | Enable debug mode |

## Example Usage

```bash
# Health check
curl https://your-app.railway.app/health

# Generate embedding
curl -X POST https://your-app.railway.app/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "desert landscapes and oil derricks"}'

# Semantic search
curl -X POST https://your-app.railway.app/search \
  -H "Content-Type: application/json" \
  -d '{"query": "mythology and transformation", "top_k": 3}'
```

## Tech Stack

- **FastAPI** - Modern Python web framework
- **OpenAI Embeddings** - text-embedding-3-small model
- **Upstash Redis** - Serverless Redis for caching
- **Railway** - Container deployment platform

## Next Steps

- [ ] Add Upstash Vector for persistent vector storage
- [ ] Implement document upload endpoint
- [ ] Add authentication
- [ ] Build frontend interface

## Enterprise Architecture

This project implements the semantic retrieval layer of a RAG system.

While this deployment uses OpenAI embeddings and Redis for demonstration purposes,
the architecture is designed to support fully on-prem enterprise environments
(e.g. Windows domain file shares, on-prem Llama servers, PostgreSQL + pgvector).

See [ENTERPRISE_USE_CASE.md](ENTERPRISE_USE_CASE.md) for details.
