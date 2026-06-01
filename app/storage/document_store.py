import json
from typing import List


class DocumentStore:

    def __init__(self, path: str = "/Users/sw-team/Desktop/rag/"
                                   "data/processed/chunks.json"):
        self.path = path

    def load_chunks(self) -> List[dict]:
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_chunks(self, chunks: List[dict]):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
