# app/pipelines/retrieval_pipeline.py

from app.embedding.providers.factory import (
    EmbeddingFactory
)

from app.vectorstores.milvus_store import (
    MilvusStore
)

from app.config.settings import (
    Settings
)

from app.retrieval.dense.dense_retriever import (
    DenseRetriever
)

from app.retrieval.bm25.bm25_index import (
    BM25Index
)

from app.retrieval.bm25.bm25_retriever import (
    BM25Retriever
)

from app.retrieval.hybrid_retriever import (
    HybridRetriever
)


class RetrievalPipeline:

    def __init__(self):

        # shared dependencies
        embedding_provider = (
            EmbeddingFactory.create(
                Settings.EMBEDDING_PROVIDER
            )
        )

        vector_store = (
            MilvusStore(
                db_path="./milvus.db",
                collection_name="rag_chunks",
                dim=384
            )
        )

        # dense retriever
        dense_retriever = (
            DenseRetriever(
                embedding_provider=embedding_provider,
                vector_store=vector_store
            )
        )

        # bm25 index
        bm25_index = BM25Index()

        bm25_index.load()

        # bm25 retriever
        bm25_retriever = (
            BM25Retriever(
                bm25_index
            )
        )

        # hybrid retriever
        self.retriever = (
            HybridRetriever(
                dense_retriever=dense_retriever,
                bm25_retriever=bm25_retriever
            )
        )

    def search(
        self,
        query: str,
        top_k: int = 5
    ):

        return self.retriever.search(
            query=query,
            top_k=top_k
        )