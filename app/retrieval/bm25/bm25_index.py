from rank_bm25 import BM25Okapi
import jieba
from app.schemas.retrieval_result import (
    RetrievalResult
)


class BM25Index:

    def __init__(self, chunks):

        self.chunks = chunks

        self.corpus = [
            list(jieba.cut(c.content))
            for c in chunks
        ]

        self.bm25 = BM25Okapi(
            self.corpus
        )

    def search(
        self,
        query,
        top_k=5
    ):

        tokenized_query = list(
            jieba.cut(query)
        )

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return [
            RetrievalResult(
                chunk=self.chunks[i],
                score=float(score),
                source="bm25"
            )
            for i, score in ranked
        ]
