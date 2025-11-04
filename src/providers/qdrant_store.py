from __future__ import annotations

from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, NamedVector, PointStruct

from .settings import settings


class QdrantStore:
    """Qdrant collection with named vectors: {"image": 512, "text": 1536}"""

    def __init__(self) -> None:
        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection = settings.qdrant_collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection in existing:
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors={
                "image": VectorParams(size=settings.image_dim, distance=Distance.COSINE),
                "text": VectorParams(size=settings.text_dim, distance=Distance.COSINE),
            },
        )

    def upsert(
        self,
        ids: List[str],
        image_vecs: List[List[float]] | None,
        text_vecs: List[List[float]] | None,
        payloads: List[Dict[str, Any]],
    ) -> None:
        vectors: List[Dict[str, List[float]]] = []
        for i in range(len(ids)):
            vec: Dict[str, List[float]] = {}
            if image_vecs is not None:
                vec["image"] = image_vecs[i]
            if text_vecs is not None:
                vec["text"] = text_vecs[i]
            vectors.append(vec)

        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))
        ]
        self.client.upsert(self.collection, points=points)

    def search_named(self, vector_name: str, query_vector: List[float], top_k: int = 5):
        return self.client.search(
            collection_name=self.collection,
            query_vector=NamedVector(name=vector_name, vector=query_vector),
            limit=top_k,
        )
