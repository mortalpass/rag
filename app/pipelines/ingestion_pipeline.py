from pathlib import Path
from uuid import uuid4

from app.chunking.ast_parser import MarkdownASTParser
from app.chunking.semantic_chunker import SemanticChunker

from app.embedding.providers.factory import EmbeddingFactory

from app.vectorstores.milvus_store import MilvusStore

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

    def ingest_markdown(
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

        texts = [
            chunk.content
            for chunk in chunks
        ]

        vectors = self.embedding_provider.embed_texts(texts)

        ids = []

        payloads = []

        for chunk in chunks:

            chunk_id = str(uuid4())

            ids.append(chunk_id)

            payloads.append(
                {
                    "content": chunk.content,
                    "metadata": {
                        "header_path": getattr(
                            chunk,
                            "header_path",
                            ""
                        )
                    }
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