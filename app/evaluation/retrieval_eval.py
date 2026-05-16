class RetrievalEvaluator:

    @staticmethod
    def print_results(results):

        print("\nReranked Results:\n")

        for r in results:

            chunk = r.chunk

            print("=" * 80)

            print(
                "Rerank Score:",
                round(
                    r.rerank_score,
                    4
                )
            )

            print(
                "Retrieval Score:",
                round(
                    r.retrieval_score,
                    4
                )
            )

            print(
                "Title:",
                chunk.title
            )

            print(
                "Path:",
                " > ".join(
                    chunk.path
                )
            )

            print("-" * 80)

            print(
                chunk.content[:500]
            )