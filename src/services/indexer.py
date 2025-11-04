from __future__ import annotations

import os
import uuid
from typing import Iterable, List, Optional

import fitz  # PyMuPDF


def save_pdf_pages_as_images(pdf_path: str, out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    saved: List[str] = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=144)
        out = os.path.join(out_dir, f"page_{i+1:04d}.png")
        pix.save(out)
        saved.append(out)
    doc.close()
    return saved


def generate_ids(n: int) -> List[str]:
    return [str(uuid.uuid4()) for _ in range(n)]
