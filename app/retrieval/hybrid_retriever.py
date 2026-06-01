# app/retrieval/hybrid_retriever.py
from app.retrieval.fusion.rrf import RRF
from app.retrieval.rerank.bge_reranker import BGEReranker

class HybridRetriever:

    def __init__(self, dense_retriever, bm25_retriever):
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.rrf = RRF()
        self.reranker = BGEReranker()

    def search(self, query: str, top_k: int = 5):

        # 1️⃣ recall
        dense_results = self.dense.search(query, top_k=30)
        bm25_results = self.bm25.search(query, top_k=30)

        # 2️⃣ fusion
        fused_results = self.rrf.fuse(dense_results, bm25_results)

        # 3️⃣ rerank
        reranked_results = self.reranker.rerank(query, fused_results, top_k=top_k)

        return reranked_results