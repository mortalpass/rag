from pathlib import Path

from app.chunking.ast_parser import (
    MarkdownASTParser
)

from app.chunking.semantic_chunker import (
    SemanticChunker
)

from app.storage.chunk_exporter import (
    ChunkExporter
)

from app.retrieval.hybrid_retriever import (
    HybridRetriever
)

from app.evaluation.retrieval_eval import (
    RetrievalEvaluator
)


def build_chunks():

    markdown_text = Path(
        "data/raw/test.md"
    ).read_text(
        encoding="utf-8"
    )

    parser = MarkdownASTParser()

    root = parser.parse(markdown_text)

    chunker = SemanticChunker()

    chunks = []

    for child in root.children:

        chunks.extend(
            chunker.chunk_section(child)
        )

    ChunkExporter.export_to_json(
        chunks,
        "data/processed/chunks.json"
    )

    return chunks


def main():

    chunks = build_chunks()

    retriever = HybridRetriever(
        chunks
    )

    print("\nHybrid RAG Ready\n")

    while True:

        query = input("\nQuery > ")

        if query == "exit":
            break

        results = retriever.search(
            query
        )

        RetrievalEvaluator.print_results(
            results
        )


if __name__ == "__main__":

    main()
