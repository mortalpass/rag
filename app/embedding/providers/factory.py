from app.embedding.models.bge_embedding import BGEEmbedding

from app.embedding.providers.base import EmbeddingProvider

from app.embedding.providers.bge_provider import BGEProvider

from app.embedding.providers.openai_provider import OpenAIProvider


class EmbeddingFactory:

    @staticmethod
    def create(
        provider_name: str
    ) -> EmbeddingProvider:

        if provider_name == "bge":

            model = BGEEmbedding()

            return BGEProvider(model)

        elif provider_name == "openai":

            return OpenAIProvider(
                api_key="YOUR_API_KEY"
            )

        else:

            raise ValueError(
                f"Unsupported provider: {provider_name}"
            )
