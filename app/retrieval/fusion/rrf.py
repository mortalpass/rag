# app/retrieval/fusion/rrf.py
from app.schemas.retrieval_result import RetrievalResult

class RRF:

    def fuse(self, dense_results, bm25_results, k: int = 60):

        scores = {}
        all_docs = {}

        # Dense 结果
        for rank, doc in enumerate(dense_results):
            doc_id = doc.chunk_id

            # 更新 RRF 分数
            scores.setdefault(doc_id, 0)
            scores[doc_id] += 1 / (k + rank + 1)

            # 如果是新文档，直接存储
            if doc_id not in all_docs:
                all_docs[doc_id] = doc
            else:
                # 已存在对象，更新 dense_score
                existing = all_docs[doc_id]
                existing.dense_score = doc.dense_score  # Dense一定有分数

        # BM25 结果
        for rank, doc in enumerate(bm25_results):
            doc_id = doc.chunk_id

            # 更新 RRF 分数
            scores.setdefault(doc_id, 0)
            scores[doc_id] += 1 / (k + rank + 1)

            if doc_id not in all_docs:
                all_docs[doc_id] = doc
            else:
                # 已存在对象，更新 bm25_score
                existing = all_docs[doc_id]
                existing.bm25_score = doc.bm25_score  # BM25一定有分数

        # 按融合分数排序
        reranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fused_results = []
        for doc_id, fusion_score in reranked:
            doc = all_docs[doc_id]
            doc.fusion_score = fusion_score
            fused_results.append(doc)

        return fused_results