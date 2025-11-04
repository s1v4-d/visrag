from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.core.contracts import RetrievalResult
from src.providers.openai_client import OpenAIClient
from src.providers.openclip_embedder import OpenCLIPEmbedder
from src.providers.qdrant_store import QdrantStore


class Retriever:
    def __init__(self, store: QdrantStore, clip: OpenCLIPEmbedder, oai: OpenAIClient) -> None:
        self.store = store
        self.clip = clip
        self.oai = oai

    def search_image(self, query: str, top_k: int) -> List[RetrievalResult]:
        qv = self.clip.embed_texts_clip([query])[0].tolist()
        hits = self.store.search_named("image", qv, top_k=top_k)
        return [
            RetrievalResult(
                id=str(h.id),
                score=float(h.score or 0.0),
                image_path=(h.payload or {}).get("image_path"),
                text=(h.payload or {}).get("text"),
                metadata=h.payload or {},
            )
            for h in hits
        ]

    def search_text(self, query: str, top_k: int) -> List[RetrievalResult]:
        qv = self.oai.embed_texts([query])[0]
        hits = self.store.search_named("text", qv, top_k=top_k)
        return [
            RetrievalResult(
                id=str(h.id),
                score=float(h.score or 0.0),
                image_path=(h.payload or {}).get("image_path"),
                text=(h.payload or {}).get("text"),
                metadata=h.payload or {},
            )
            for h in hits
        ]

    def search_hybrid(self, query: str, top_k: int) -> List[RetrievalResult]:
        img = self.search_image(query, top_k)
        txt = self.search_text(query, top_k)
        # naive merge by id: normalize and sum scores
        scores: dict[str, float] = {}
        items: dict[str, RetrievalResult] = {}
        all_items = img + txt
        if not all_items:
            return []
        sc = np.array([r.score for r in all_items])
        min_s, max_s = sc.min(), sc.max()
        rng = (max_s - min_s) or 1.0
        for r in all_items:
            norm = (r.score - min_s) / rng
            scores[r.id] = scores.get(r.id, 0.0) + float(norm)
            items[r.id] = items.get(r.id, r)
        ranked = sorted(items.values(), key=lambda r: scores[r.id], reverse=True)
        return ranked[:top_k]
