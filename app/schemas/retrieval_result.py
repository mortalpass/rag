from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RetrievalResult:

    chunk_id: str

    content: str

    metadata: Dict[str, Any]

    dense_score: float = 0.0

    bm25_score: float = 0.0

    fusion_score: float = 0.0

    rerank_score: float = 0.0