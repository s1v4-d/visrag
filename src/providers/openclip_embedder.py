from __future__ import annotations

from typing import List

import numpy as np
import torch
from PIL import Image
import open_clip


class OpenCLIPEmbedder:
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device).eval()

    def embed_images(self, image_paths: List[str]) -> np.ndarray:
        vecs: List[np.ndarray] = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            img_t = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                v = self.model.encode_image(img_t)
                v = v / v.norm(dim=-1, keepdim=True)
            vecs.append(v.squeeze(0).cpu().numpy())
        return np.vstack(vecs).astype("float32") if vecs else np.zeros((0, 512), dtype="float32")

    def embed_texts_clip(self, texts: List[str]) -> np.ndarray:
        toks = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            v = self.model.encode_text(toks)
            v = v / v.norm(dim=-1, keepdim=True)
        return v.cpu().numpy().astype("float32")
