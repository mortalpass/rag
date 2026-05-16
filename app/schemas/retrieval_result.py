from dataclasses import dataclass

from app.chunking.chunk_models import (
    Chunk
)


@dataclass
class RetrievalResult:

    chunk: Chunk

    score: float

    source: str