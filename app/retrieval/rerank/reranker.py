from abc import ABC, abstractmethod

from app.schemas.retrieval_result import (
    RetrievalResult
)

from app.schemas.rerank_result import (
    RerankResult
)


class Reranker(ABC):

    @abstractmethod
    def rerank(
        self,
        query: str,
        retrieval_results: list[
            RetrievalResult
        ]
    ) -> list[RerankResult]:

        pass