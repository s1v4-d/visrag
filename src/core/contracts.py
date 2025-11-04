from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalResult:
    id: str
    score: float
    image_path: Optional[str] = None
    text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Query:
    text: str
    top_k: int = 5
    mode: str = "hybrid"  # image|text|hybrid
    filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    text: str
    citations: List[RetrievalResult] = field(default_factory=list)
