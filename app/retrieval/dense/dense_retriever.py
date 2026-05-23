from app.vectorstores.base import BaseVectorStore


class DenseRetriever:

    def __init__(
        self,
        embedding_provider,
        vector_store: BaseVectorStore
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ):

        query_vector = self.embedding_provider.embed_text(query)

        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k
        )

        return results