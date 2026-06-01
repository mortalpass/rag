import os
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
from app.retrieval.bm25.bm25_index import BM25Index
from app.storage.ast_exporter import print_tree

from app.vectorstores.milvus_store import (
    MilvusStore
)

from app.config.settings import Settings


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
            ids.append(chunk.chunk_id)

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

        self.vector_store.insert(
            ids=ids,
            vectors=vectors,
            payloads=payloads
        )

        print(
            f"Ingested {len(chunks)} chunks"
        )

    def ingest_markdown(self, file_path: str):

        chunks = self.parse_and_chunk(file_path)

        self.ingest_chunks(chunks)

        bm25_index = BM25Index()

        if os.path.exists(bm25_index.path):
            bm25_index.load()
            bm25_index.build(chunks, incremental=True)
        else:
            bm25_index.build(chunks, incremental=False)

        bm25_index.save()

        return chunks
