from app.pipelines.ingestion_pipeline import (
    IngestionPipeline
)


def main():

    pipeline = IngestionPipeline()

    pipeline.ingest_markdown(
        "/Users/sw-team/Desktop/rag/data/raw/test.md"
    )


if __name__ == "__main__":
    main()
