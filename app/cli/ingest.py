from app.pipelines.ingestion_pipeline import (
    IngestionPipeline
)

from app.storage.chunk_exporter import (
    ChunkExporter
)


def main():
    pipeline = IngestionPipeline()

    chunks = pipeline.ingest_markdown(
        "/Users/sw-team/Desktop/rag/"
        "data/raw/test.md"
    )

    ChunkExporter.export_to_json(
        chunks=chunks,
        output_path=(
            "/Users/sw-team/Desktop/rag/"
            "data/processed/chunks.json"
        )
    )


if __name__ == "__main__":
    main()
