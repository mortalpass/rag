# app/cli/serve.py

from app.pipelines.retrieval_pipeline import (
    RetrievalPipeline
)

from app.evaluation.retrieval_eval import (
    RetrievalEvaluator
)


def main():

    pipeline = RetrievalPipeline()

    print("\nRAG Ready\n")

    while True:

        query = input(
            "\nQuery > "
        )

        if query == "exit":
            break

        results = pipeline.search(
            query=query,
            top_k=5
        )

        RetrievalEvaluator.print_results(
            results
        )


if __name__ == "__main__":
    main()