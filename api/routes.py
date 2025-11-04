from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from src.providers.openai_client import OpenAIClient
from src.providers.openclip_embedder import OpenCLIPEmbedder
from src.providers.qdrant_store import QdrantStore
from src.providers.settings import settings
from src.services.indexer import save_pdf_pages_as_images, generate_ids
from src.services.retriever import Retriever


router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


class TextDoc(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


@router.post("/index/texts")
def index_texts(docs: List[TextDoc]) -> dict:
    store = QdrantStore()
    oai = OpenAIClient()
    ids = generate_ids(len(docs))
    text_vecs = oai.embed_texts([d.text for d in docs])
    payloads = [{"text": d.text, **d.metadata} for d in docs]
    store.upsert(ids=ids, image_vecs=None, text_vecs=text_vecs, payloads=payloads)
    return {"upserted": len(ids), "ids": ids}


@router.post("/index/images")
async def index_images(files: List[UploadFile] = File(...)) -> dict:
    store = QdrantStore()
    clip = OpenCLIPEmbedder()
    os.makedirs(settings.data_dir, exist_ok=True)
    saved_paths: List[str] = []
    for f in files:
        out = os.path.join(settings.data_dir, f"img_{uuid.uuid4()}.png")
        with open(out, "wb") as w:
            w.write(await f.read())
        saved_paths.append(out)
    image_vecs = clip.embed_images(saved_paths)
    ids = generate_ids(len(saved_paths))
    payloads = [{"image_path": p} for p in saved_paths]
    store.upsert(ids=ids, image_vecs=image_vecs.tolist(), text_vecs=None, payloads=payloads)
    return {"upserted": len(ids), "ids": ids}


@router.post("/index/pdf")
async def index_pdf(file: UploadFile = File(...)) -> dict:
    # Save PDF, render pages, embed images
    store = QdrantStore()
    clip = OpenCLIPEmbedder()
    os.makedirs(settings.data_dir, exist_ok=True)
    pdf_path = os.path.join(settings.data_dir, f"{uuid.uuid4()}.pdf")
    with open(pdf_path, "wb") as w:
        w.write(await file.read())
    page_dir = os.path.join(settings.data_dir, f"pdf_{uuid.uuid4()}_pages")
    pages = save_pdf_pages_as_images(pdf_path, page_dir)
    if not pages:
        raise HTTPException(status_code=400, detail="No pages extracted from PDF")
    image_vecs = clip.embed_images(pages)
    ids = generate_ids(len(pages))
    payloads = [{"image_path": p, "source_pdf": os.path.basename(pdf_path)} for p in pages]
    store.upsert(ids=ids, image_vecs=image_vecs.tolist(), text_vecs=None, payloads=payloads)
    return {"upserted": len(ids), "ids": ids}


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    mode: str = Field("hybrid", pattern="^(image|text|hybrid)$")


@router.post("/search")
def search(req: SearchRequest) -> dict:
    store = QdrantStore()
    clip = OpenCLIPEmbedder()
    oai = OpenAIClient()
    r = Retriever(store, clip, oai)
    if req.mode == "image":
        hits = r.search_image(req.query, req.top_k)
    elif req.mode == "text":
        hits = r.search_text(req.query, req.top_k)
    else:
        hits = r.search_hybrid(req.query, req.top_k)
    return {
        "results": [
            {
                "id": h.id,
                "score": h.score,
                "image_path": h.image_path,
                "text": h.text,
                "metadata": h.metadata,
            }
            for h in hits
        ]
    }


class CompleteRequest(BaseModel):
    query: str
    top_k: int = 3
    mode: str = Field("hybrid", pattern="^(image|text|hybrid)$")


@router.post("/complete")
def complete(req: CompleteRequest) -> dict:
    store = QdrantStore()
    clip = OpenCLIPEmbedder()
    oai = OpenAIClient()
    r = Retriever(store, clip, oai)
    hits = (
        r.search_image(req.query, req.top_k)
        if req.mode == "image"
        else r.search_text(req.query, req.top_k)
        if req.mode == "text"
        else r.search_hybrid(req.query, req.top_k)
    )
    img_paths = [h.image_path for h in hits if h.image_path][:3]
    answer = oai.vision_answer(req.query, img_paths) if img_paths else "No relevant images found."
    return {
        "answer": answer,
        "results": [
            {
                "id": h.id,
                "score": h.score,
                "image_path": h.image_path,
                "text": h.text,
                "metadata": h.metadata,
            }
            for h in hits
        ],
    }
