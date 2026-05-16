import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from app.config.settings import (
    Settings
)

from app.models.model_manager import (
    ModelManager
)

from app.retrieval.rerank.reranker import (
    Reranker
)

from app.schemas.rerank_result import (
    RerankResult
)


class BGEReranker(
    Reranker
):

    def __init__(self):

        model_dir = (
            ModelManager
            .get_reranker_model_path()
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_dir
            )
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(model_dir)
        )

        self.model.eval()

    @torch.no_grad()
    def rerank(
        self,
        query,
        retrieval_results
    ):

        pairs = [
            (
                query,
                item.chunk.content
            )
            for item in retrieval_results
        ]

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )

        scores = (
            self.model(**inputs)
            .logits
            .view(-1,)
            .float()
        )

        reranked = []

        for item, score in zip(
            retrieval_results,
            scores
        ):

            reranked.append(
                RerankResult(
                    chunk=item.chunk,
                    retrieval_score=item.score,
                    rerank_score=float(score)
                )
            )

        reranked.sort(
            key=lambda x: x.rerank_score,
            reverse=True
        )

        return reranked[
            :Settings.RERANK_TOP_K
        ]