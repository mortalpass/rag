from openai import OpenAI

from app.embedding.providers.base import EmbeddingProvider


class OpenAIProvider(EmbeddingProvider):

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small"
    ):

        self.client = OpenAI(api_key=api_key)

        self.model = model

    def embed_text(
        self,
        text: str
    ) -> list[float]:

        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )

        return response.data[0].embedding

    def embed_texts(
        self,
        texts: list[str]
    ) -> list[list[float]]:

        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )

        return [
            item.embedding
            for item in response.data
        ]