from sentence_transformers import (
    SentenceTransformer
)

from app.embedding.embedding_provider import (
    EmbeddingProvider
)

from app.models.model_manager import (
    ModelManager
)


class BGEEmbedding(
    EmbeddingProvider
):

    def __init__(self):

        model_path = (
            ModelManager
            .get_embedding_model_path()
        )

        self.model = SentenceTransformer(
            model_path
        )

    def embed_text(self, text):

        return self.model.encode(
            text,
            normalize_embeddings=True
        )

    def embed_texts(self, texts):

        return self.model.encode(
            texts,
            normalize_embeddings=True
        )