from pathlib import Path

from ast_parser import MarkdownASTParser
from chunk_exporter import ChunkExporter

from semantic_chunker import SemanticChunker


def print_chunk(chunk):

    print("=" * 100)

    print("TITLE:", chunk.title)

    print("PATH:", " > ".join(chunk.path))

    print("TOKENS:", chunk.token_count)

    print("TYPE:", chunk.chunk_type)

    print("-" * 100)

    print(chunk.content[:800])

    print("\n")


def main():

    markdown_text = Path(
        "data/test.md"
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
        "output/output_chunks.json"
    )


if __name__ == "__main__":

    main()