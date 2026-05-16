import numpy as np

from app.embedding.bge_embedding import (
    BGEEmbedding
)

from app.schemas.retrieval_result import (
    RetrievalResult
)


class DenseIndex:

    def __init__(self, chunks):

        self.chunks = chunks

        self.embedding_model = BGEEmbedding()

        self.embeddings = (
            self.embedding_model.embed_texts(
                [c.content for c in chunks]
            )
        )

    def search(
        self,
        query,
        top_k=5
    ):

        q_emb = self.embedding_model.embed_text(
            query
        )

        scores = np.dot(
            self.embeddings,
            q_emb
        )

        ranked_indices = np.argsort(
            scores
        )[::-1][:top_k]

        return [
            RetrievalResult(
                chunk=self.chunks[i],
                score=float(scores[i]),
                source="dense"
            )
            for i in ranked_indices
        ]