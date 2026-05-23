from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseVectorStore(ABC):

    @abstractmethod
    def create_collection(self):
        pass

    @abstractmethod
    def insert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]]
    ):
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5
    ):
        pass