from app.pipelines.retrieval_pipeline import (
    RetrievalPipeline
)


def main():

    pipeline = RetrievalPipeline()

    print("\nRAG Ready\n")

    while True:

        query = input("\nQuery > ")

        if query == "exit":
            break

        results = pipeline.search(
            query=query,
            top_k=5
        )

        for result in results:

            entity = result["entity"]

            print("\n================")
            print(entity["content"])

            metadata = entity.get(
                "metadata",
                {}
            )

            print(metadata)

            print(
                f"score: {result['distance']}"
            )


if __name__ == "__main__":
    main()