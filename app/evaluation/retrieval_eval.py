# app/evaluation/retrieval_eval.py

class RetrievalEvaluator:

    @staticmethod
    def print_results(results):

        print("\nReranked Results:\n")

        for idx, r in enumerate(results):

            print("=" * 80)

            print(
                f"Rank: {idx + 1}"
            )

            print(
                "Dense Score:",
                round(r.dense_score, 4)
            )

            print(
                "BM25 Score:",
                round(r.bm25_score, 4)
            )

            print(
                "Fusion Score:",
                round(r.fusion_score, 4)
            )

            print(
                "Rerank Score:",
                round(r.rerank_score, 4)
            )

            print(
                "Title:",
                r.metadata.get(
                    "title",
                    ""
                )
            )

            print(
                "Path:",
                " > ".join(
                    r.metadata.get(
                        "path",
                        []
                    )
                )
            )

            print("-" * 80)

            print(
                r.content[:500]
            )