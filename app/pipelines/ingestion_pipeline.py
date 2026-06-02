import os
from hashlib import md5
from pathlib import Path

from app.chunking.ast_parser import (
    MarkdownASTParser
)

from app.chunking.semantic_chunker import (
    SemanticChunker
)

from app.embedding.providers.factory import (
    EmbeddingFactory
)

from app.retrieval.bm25.bm25_index import (
    BM25Index
)

from app.vectorstores.milvus_store import (
    MilvusStore
)

from app.config.settings import (
    Settings
)


class IngestionPipeline:

    def __init__(self):

        self.parser = MarkdownASTParser()

        self.embedding_provider = (
            EmbeddingFactory.create(
                Settings.EMBEDDING_PROVIDER
            )
        )

        self.chunker = SemanticChunker(
            self.embedding_provider
        )

        self.vector_store = MilvusStore(
            db_path="./milvus.db",
            collection_name="rag_chunks",
            dim=384
        )

        self.vector_store.create_collection()

    def parse_and_chunk(
        self,
        file_path: str
    ):

        markdown_text = Path(
            file_path
        ).read_text(
            encoding="utf-8"
        )

        root = self.parser.parse(
            markdown_text
        )

        path = Path(file_path)

        source_file = str(
            path.resolve()
        )

        doc_id = md5(
            source_file.encode("utf-8")
        ).hexdigest()

        updated_at = str(
            path.stat().st_mtime
        )

        self.parser.attach_document_metadata(
            root,

            doc_id=doc_id,

            source=path.name,

            source_file=source_file,

            root_title=path.stem,

            updated_at=updated_at
        )

        chunks = []

        for child in root.children:

            chunks.extend(
                self.chunker.chunk_section(
                    child
                )
            )

        return chunks

    def ingest_chunks(
        self,
        chunks
    ):

        if not chunks:
            return

        texts = [
            chunk.content
            for chunk in chunks
        ]

        vectors = (
            self.embedding_provider
            .embed_texts(texts)
        )

        ids = []

        payloads = []

        for chunk in chunks:

            ids.append(
                chunk.chunk_id
            )

            payloads.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "title": chunk.title,
                    "path": chunk.path,
                    "chunk_type": chunk.chunk_type,

                    # flatten metadata
                    **chunk.metadata.model_dump()
                }
            )

        self.vector_store.upsert(
            ids=ids,
            vectors=vectors,
            payloads=payloads
        )

        print(
            f"Ingested {len(chunks)} chunks"
        )

    def ingest_markdown(
        self,
        file_path: str
    ):

        chunks = self.parse_and_chunk(
            file_path
        )

        self.ingest_chunks(
            chunks
        )

        bm25_index = BM25Index()

        if os.path.exists(
            bm25_index.path
        ):
            bm25_index.load()

            bm25_index.build(
                chunks,
                incremental=True
            )

        else:

            bm25_index.build(
                chunks,
                incremental=False
            )

        bm25_index.save()

        return chunks

    def ingest_directory(
        self,
        directory: str,
        recursive: bool = True
    ):
        """
        Ingest all markdown files under a directory.
        """

        root = Path(directory)

        if not root.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        pattern = "**/*.md" if recursive else "*.md"

        md_files = sorted(
            root.glob(pattern)
        )

        if not md_files:
            print(
                f"No markdown files found in {directory}"
            )
            return []

        print(
            f"Found {len(md_files)} markdown files"
        )

        bm25_index = BM25Index()

        if os.path.exists(
            bm25_index.path
        ):
            bm25_index.load()
            incremental = True
        else:
            incremental = False

        all_chunks = []

        for md_file in md_files:

            print(
                f"Processing: {md_file}"
            )

            try:

                chunks = self.parse_and_chunk(
                    str(md_file)
                )

                self.ingest_chunks(
                    chunks
                )

                bm25_index.build(
                    chunks,
                    incremental=incremental
                )

                incremental = True

                all_chunks.extend(
                    chunks
                )

                print(
                    f"✓ {md_file.name}: {len(chunks)} chunks"
                )

            except Exception as e:

                print(
                    f"✗ Failed: {md_file}"
                )

                print(e)

        bm25_index.save()

        print(
            f"\nFinished"
        )

        print(
            f"Files: {len(md_files)}"
        )

        print(
            f"Chunks: {len(all_chunks)}"
        )

        return all_chunks