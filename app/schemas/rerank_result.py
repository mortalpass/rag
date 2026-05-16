from dataclasses import dataclass

from app.chunking.chunk_models import (
    Chunk
)


@dataclass
class RerankResult:

    chunk: Chunk

    retrieval_score: float

    rerank_score: float