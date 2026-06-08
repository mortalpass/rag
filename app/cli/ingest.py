from app.core.container import Container

from app.storage.chunk_exporter import (
    ChunkExporter
)


def main():

    pipeline = (
        Container.get_ingestion_pipeline()
    )

    chunks = pipeline.ingest_directory(
        "/Users/sw-team/Desktop/rag/data/raw"
    )

    ChunkExporter.export_to_json(
        chunks=chunks,
        output_path=(
            "/Users/sw-team/Desktop/rag/"
            "data/processed/chunks.json"
        )
    )

    print(
        f"Ingested {len(chunks)} chunks"
    )


if __name__ == "__main__":
    main()