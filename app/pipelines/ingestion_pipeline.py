import datetime

from hashlib import md5
from pathlib import Path

from app.chunking.ast_parser import MarkdownASTParser
from app.chunking.semantic_chunker import SemanticChunker

from app.retrieval.bm25.bm25_index import BM25Index
from app.storage.document_record import DocumentRecord
from app.storage.document_registry import DocumentRegistry


class IngestionPipeline:

    def __init__(
        self,
        embedding_provider,
        vector_store,
        bm25_index: BM25Index,
        document_registry: DocumentRegistry
    ):

        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.document_registry = document_registry

        self.parser = MarkdownASTParser()

        self.chunker = SemanticChunker(
            self.embedding_provider
        )

    # =========================
    # Document Load
    # =========================

    def load_document(self, file_path: str):

        path = Path(file_path)

        markdown_text = path.read_text(
            encoding="utf-8"
        )

        source_file = str(path.resolve())

        doc_id = md5(
            source_file.encode("utf-8")
        ).hexdigest()

        content_hash = md5(
            markdown_text.encode("utf-8")
        ).hexdigest()

        updated_at = str(path.stat().st_mtime)

        return {
            "path": path,
            "markdown_text": markdown_text,
            "source_file": source_file,
            "doc_id": doc_id,
            "content_hash": content_hash,
            "updated_at": updated_at
        }

    # =========================
    # Change Detection
    # =========================

    def should_skip(self, doc_id: str, content_hash: str) -> bool:

        record = self.document_registry.get(doc_id)

        if record is None:
            return False

        return record.content_hash == content_hash

    # =========================
    # Delete Old Index
    # =========================

    def remove_document_chunks(self, doc_id: str):

        chunk_ids = self.document_registry.get_chunk_ids(doc_id)

        if chunk_ids:
            self.vector_store.delete(chunk_ids)

        self.bm25_index.delete_by_doc_id(doc_id)

    # =========================
    # Parse + Chunk
    # =========================

    def parse_and_chunk(self, doc_ctx):

        root = self.parser.parse(
            doc_ctx["markdown_text"]
        )

        self.parser.attach_document_metadata(
            root,
            doc_id=doc_ctx["doc_id"],
            source=doc_ctx["path"].name,
            source_file=doc_ctx["source_file"],
            root_title=doc_ctx["path"].stem,
            updated_at=doc_ctx["updated_at"]
        )

        chunks = []

        for child in root.children:
            chunks.extend(
                self.chunker.chunk_section(child)
            )

        return chunks

    # =========================
    # Vector Store Write
    # =========================

    def ingest_chunks(self, chunks):

        if not chunks:
            return

        texts = [c.content for c in chunks]

        vectors = self.embedding_provider.embed_texts(texts)

        ids = []
        payloads = []

        for c in chunks:

            ids.append(c.chunk_id)

            payloads.append({
                "chunk_id": c.chunk_id,
                "content": c.content,
                "title": c.title,
                "path": c.path,
                "chunk_type": c.chunk_type,
                **c.metadata.model_dump()
            })

        self.vector_store.upsert(
            ids=ids,
            vectors=vectors,
            payloads=payloads
        )

    # =========================
    # Registry Update
    # =========================

    def save_document_record(self, doc_ctx, chunks):

        doc_id = doc_ctx["doc_id"]

        old = self.document_registry.get(doc_id)

        now = datetime.datetime.utcnow().isoformat()

        version = old.version + 1 if old else 1
        created_at = old.created_at if old else now

        record = DocumentRecord(
            doc_id=doc_id,
            source_file=doc_ctx["source_file"],
            title=doc_ctx["path"].stem,
            content_hash=doc_ctx["content_hash"],
            chunk_count=len(chunks),
            version=version,
            created_at=created_at,
            updated_at=now
        )

        self.document_registry.upsert(record)

        self.document_registry.upsert_chunk_mapping(
            doc_id,
            [c.chunk_id for c in chunks]
        )

    # =========================
    # Single File Ingestion
    # =========================

    def ingest_markdown(self, file_path: str):

        doc_ctx = self.load_document(file_path)

        doc_id = doc_ctx["doc_id"]
        content_hash = doc_ctx["content_hash"]

        # ===== skip unchanged =====
        if self.should_skip(doc_id, content_hash):
            print(f"[SKIP] {file_path}")
            return []

        # ===== remove old index if exists =====
        if self.document_registry.exists(doc_id):
            self.remove_document_chunks(doc_id)

        # ===== parse + chunk =====
        chunks = self.parse_and_chunk(doc_ctx)

        # ===== vector store =====
        self.ingest_chunks(chunks)

        # ===== bm25 =====
        self.bm25_index.build(chunks, incremental=True)
        self.bm25_index.save()

        # ===== registry =====
        self.save_document_record(doc_ctx, chunks)

        print(f"[INGEST] {file_path} -> {len(chunks)} chunks")

        return chunks

    # =========================
    # Directory Ingestion
    # =========================

    def ingest_directory(
        self,
        directory: str,
        recursive: bool = True
    ):

        root = Path(directory)

        if not root.exists():
            raise FileNotFoundError(directory)

        pattern = "**/*.md" if recursive else "*.md"

        files = sorted(root.glob(pattern))

        all_chunks = []

        for f in files:

            try:
                chunks = self.ingest_markdown(str(f))
                all_chunks.extend(chunks)

            except Exception as e:
                print(f"[FAILED] {f}")
                print(e)

        return all_chunks