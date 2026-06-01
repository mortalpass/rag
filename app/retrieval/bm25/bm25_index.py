import pickle
from rank_bm25 import BM25Okapi

from app.chunking.tokenizer_utils import tokenize


class BM25Index:

    def __init__(self, path: str = "bm25.index"):
        self.path = path
        self.bm25 = None
        self.chunks = []

    def build(self, chunks, incremental: bool = False):
        """
        chunks: 新增 chunks 列表
        incremental: 是否增量更新
        """
        if incremental and self.chunks:
            self.chunks.extend(chunks)  # 保留已有chunks
        else:
            self.chunks = chunks  # 全量覆盖

        tokenized = [
            tokenize(chunk.content)
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(tokenized)

    def save(self):

        with open(self.path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "chunks": self.chunks
            }, f)

    def load(self):

        with open(self.path, "rb") as f:
            data = pickle.load(f)

        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
