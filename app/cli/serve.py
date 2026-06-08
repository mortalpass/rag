# app/cli/serve.py
from app.evaluation.retrieval_eval import (
    RetrievalEvaluator
)
from app.core.container import Container


def main():
    pipeline = (
        Container.get_retrieval_pipeline()
    )

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