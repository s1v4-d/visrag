# VLM RAG (Production-grade scaffold)

This project provides a production-grade scaffold for a multimodal RAG system using:

- OpenAI GPT-4o-mini for multimodal reasoning (vision + text)
- OpenAI `text-embedding-3-small` for text embeddings
- OpenCLIP for image/text embeddings in the visual retrieval track
- Qdrant as a local vector database with named vectors (image, text)
- FastAPI for a clean API surface
- Docker Compose + Makefile for local orchestration and dev experience

The design follows clean interfaces (providers/services) and aligns with patterns in the reference repo: https://github.com/s1v4-d/talk2data

## Quick start

1) Create an `.env` file via Makefile and build the stack:

```bash
make build-docker
make start-docker
```

2) Open the API at http://localhost:8080/docs

3) Set your OpenAI key in `.env` (replace placeholder) or export `OPENAI_API_KEY`.

## API overview

- GET `/api/health` – health check
- POST `/api/index/texts` – index text docs (JSON list of {text, metadata})
- POST `/api/index/images` – index multiple images (multipart)
- POST `/api/index/pdf` – index a PDF (extracts pages to images and indexes)
- POST `/api/search` – search with `mode` in {image|text|hybrid}
- POST `/api/complete` – retrieve + answer with GPT-4o-mini using top images

## Why Qdrant

Qdrant supports named vectors per collection (e.g., `image` and `text`) and rich payloads. That lets us store both image and text vector spaces for each item, and search either space or both (hybrid).

Note: pgvector can store any vector type (text or image), but it doesn’t natively provide multiple named vector spaces per record; you can add multiple vector columns, but Qdrant’s ergonomics and performance for ANN search make it a good default for multimodal.

## Dev commands

```bash
make build-docker    # build containers
make start-docker    # start api + qdrant
make logs            # follow API logs
make logs-all        # follow all logs
make lint            # run ruff check
make fmt             # run ruff format
make test            # run pytest in container
make stop-docker     # stop stack
```

## Env vars

- `OPENAI_API_KEY` – required for embeddings and GPT-4o-mini
- `QDRANT_URL` – default http://qdrant:6333
- `SERVICE_PORT` – default 8080

## Notes

- OpenCLIP and PyTorch run on CPU by default; if CUDA is available, it is used automatically.
- For large PDFs/images, consider downscaling before VLM calls to reduce cost/latency.# vlmrag
VLM based RAG pipeline
