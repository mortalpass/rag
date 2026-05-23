from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def embed_text(
            self,
            text: str
    ) -> list[float]:
        pass

    @abstractmethod
    def embed_texts(
            self,
            texts: list[str]
    ) -> list[list[float]]:
        pass
