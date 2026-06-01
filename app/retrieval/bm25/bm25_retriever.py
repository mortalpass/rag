# app/retrieval/bm25/bm25_retriever.py

from app.schemas.retrieval_result import (
    RetrievalResult
)
from app.chunking.tokenizer_utils import tokenize


class BM25Retriever:

    def __init__(
        self,
        index
    ):

        self.index = index

    def search(self, query: str, top_k: int = 10):
        tokenized_query = tokenize(query)
        scores = self.index.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(self.index.chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for chunk, score in ranked[:top_k]:

            # flatten metadata
            metadata = {
                "title": getattr(chunk, "title", ""),
                "path": getattr(chunk, "path", ""),
                "chunk_type": getattr(chunk, "chunk_type", "")
            }

            # 如果 chunk.metadata 是 Pydantic 模型，可以 flatten
            if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "model_dump"):
                metadata.update(chunk.metadata.model_dump())

            results.append(
                RetrievalResult(
                    chunk_id=getattr(chunk, "chunk_id", ""),
                    content=getattr(chunk, "content", ""),
                    metadata=metadata,
                    bm25_score=float(score)
                )
            )

        return results