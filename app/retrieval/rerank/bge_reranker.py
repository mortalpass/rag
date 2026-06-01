# app/retrieval/rerank/bge_reranker.py
from app.models.model_manager import ModelManager
from sentence_transformers import CrossEncoder

class BGEReranker:

    def __init__(self):
        model_path = ModelManager.get_reranker_model_path()
        self.model = CrossEncoder(model_path)

    def rerank(self, query, documents, top_k=5):
        pairs = [[query, doc.content] for doc in documents]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        results = []
        for doc, score in ranked[:top_k]:
            doc.rerank_score = float(score)
            results.append(doc)

        return results