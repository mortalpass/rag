from collections import defaultdict

from app.schemas.retrieval_result import (
    RetrievalResult
)


class RRF:

    def __init__(self, k=60):

        self.k = k

    def fuse(
        self,
        dense_results,
        bm25_results
    ):

        score_map = defaultdict(float)

        chunk_map = {}

        def add(results):

            for rank, item in enumerate(results):

                cid = item.chunk.chunk_id

                score_map[cid] += (
                    1 / (self.k + rank)
                )

                chunk_map[cid] = item.chunk

        add(dense_results)

        add(bm25_results)

        fused = [
            RetrievalResult(
                chunk=chunk_map[cid],
                score=score,
                source="rrf"
            )
            for cid, score in score_map.items()
        ]

        return sorted(
            fused,
            key=lambda x: x.score,
            reverse=True
        )