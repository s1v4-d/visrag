from __future__ import annotations

import base64
from typing import List

from openai import OpenAI

from .settings import settings


class OpenAIClient:
    def __init__(self) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        self.client = OpenAI(api_key=settings.openai_api_key)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=settings.embedding_model, input=texts)
        return [d.embedding for d in resp.data]

    def vision_answer(self, question: str, image_paths: List[str]) -> str:
        parts = [{"type": "input_text", "text": question}]
        for p in image_paths[:3]:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            parts.append({"type": "input_image", "image_url": f"data:image/png;base64,{b64}"})
        resp = self.client.responses.create(
            model=settings.openai_vision_model,
            input=[{"role": "user", "content": parts}],
        )
        # SDK exposes output_text convenience
        return getattr(resp, "output_text", "") or ""
