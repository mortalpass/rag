from sentence_transformers import (
    SentenceTransformer
)

from app.models.model_manager import (
    ModelManager
)


class BGEEmbedding:

    def __init__(self):

        model_path = (
            ModelManager
            .get_embedding_model_path()
        )

        self.model = SentenceTransformer(
            model_path
        )

    def embed_text(
        self,
        text: str
    ) -> list[float]:

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()

    def embed_batch(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings.tolist()
