from app.embedding.providers.factory import (
    EmbeddingFactory
)

from app.vectorstores.milvus_store import (
    MilvusStore
)

from app.config.settings import (
    Settings
)


class RetrievalPipeline:

    def __init__(self):

        self.embedding_provider = (
            EmbeddingFactory.create(
                Settings.EMBEDDING_PROVIDER
            )
        )

        self.vector_store = MilvusStore(
            db_path="./milvus.db",
            collection_name="rag_chunks",
            dim=384
        )

    def search(
        self,
        query: str,
        top_k: int = 5
    ):

        query_vector = (
            self.embedding_provider.embed_text(
                query
            )
        )

        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k
        )

        return results
