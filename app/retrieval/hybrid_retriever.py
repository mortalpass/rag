from app.retrieval.bm25.bm25_index import (
    BM25Index
)

from app.retrieval.dense.dense_index import (
    DenseIndex
)

from app.retrieval.fusion.rrf import (
    RRF
)

from app.retrieval.rerank.bge_reranker import (
    BGEReranker
)


class HybridRetriever:

    def __init__(self, chunks):

        self.bm25 = BM25Index(chunks)

        self.dense = DenseIndex(chunks)

        self.rrf = RRF()

        self.reranker = BGEReranker()

    def search(
        self,
        query,
        top_k=5
    ):

        dense_results = self.dense.search(
            query,
            top_k
        )

        bm25_results = self.bm25.search(
            query,
            top_k
        )

        fused_results = self.rrf.fuse(
            dense_results,
            bm25_results
        )

        reranked_results = (
            self.reranker.rerank(
                query,
                fused_results
            )
        )

        return reranked_results