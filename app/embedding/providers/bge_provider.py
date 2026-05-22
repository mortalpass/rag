from app.embedding.providers.base import EmbeddingProvider

from app.embedding.models.bge_embedding import BGEEmbedding


class BGEProvider(EmbeddingProvider):

    def __init__(
        self,
        model: BGEEmbedding
    ):
        self.model = model

    def embed_text(self, text: str) -> list[float]:
        return self.model.embed_text(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self.model.embed_batch(texts)
