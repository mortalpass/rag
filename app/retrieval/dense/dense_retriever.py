# app/retrieval/dense/dense_retriever.py
from app.schemas.retrieval_result import RetrievalResult


class DenseRetriever:

    def __init__(self, embedding_provider, vector_store):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def search(self, query: str, top_k: int = 10):
        # BGE query instruction
        query_vector = self.embedding_provider.embed_text(
            "Represent this sentence for searching relevant passages: " + query
        )

        raw_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k
        )

        results = []

        for hit in raw_results:
            entity = hit["entity"]
            similarity = float(hit["distance"])  # COSINE metric

            results.append(
                RetrievalResult(
                    chunk_id=hit["id"],
                    content=entity.get("content", ""),
                    metadata=entity.get("metadata", {}),
                    dense_score=similarity
                )
            )

        return results
