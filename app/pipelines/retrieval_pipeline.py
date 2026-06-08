from app.retrieval.dense.dense_retriever import (
    DenseRetriever
)

from app.retrieval.bm25.bm25_retriever import (
    BM25Retriever
)

from app.retrieval.hybrid_retriever import (
    HybridRetriever
)


class RetrievalPipeline:

    def __init__(
        self,
        embedding_provider,
        vector_store,
        bm25_index
    ):
        dense_retriever = (
            DenseRetriever(
                embedding_provider,
                vector_store
            )
        )

        bm25_retriever = (
            BM25Retriever(
                bm25_index
            )
        )

        self.retriever = (
            HybridRetriever(
                dense_retriever,
                bm25_retriever
            )
        )

    def search(
        self,
        query,
        top_k=5
    ):
        return self.retriever.search(
            query,
            top_k
        )