import os

from app.config.settings import (
    Settings
)

from app.embedding.providers.factory import (
    EmbeddingFactory
)

from app.vectorstores.milvus_store import (
    MilvusStore
)

from app.retrieval.bm25.bm25_index import (
    BM25Index
)

from app.pipelines.ingestion_pipeline import (
    IngestionPipeline
)

from app.pipelines.retrieval_pipeline import (
    RetrievalPipeline
)

from app.storage.document_registry import (
    DocumentRegistry
)


class Container:

    initialized = False

    document_registry = None

    embedding_provider = None

    vector_store = None

    bm25_index = None

    ingestion_pipeline = None

    retrieval_pipeline = None

    @classmethod
    def initialize(cls):

        if cls.initialized:
            return

        print("Loading Embedding Model...")

        cls.embedding_provider = (
            EmbeddingFactory.create(
                Settings.EMBEDDING_PROVIDER
            )
        )

        print("Connecting Milvus...")

        cls.vector_store = MilvusStore(
            db_path=Settings.MILVUS_DB_PATH,
            collection_name=Settings.MILVUS_COLLECTION_NAME,
            dim=Settings.EMBEDDING_DIM
        )

        print("Loading BM25...")

        cls.bm25_index = BM25Index(
            path=Settings.BM25_INDEX_PATH
        )

        if os.path.exists(
            Settings.BM25_INDEX_PATH
        ):
            try:
                cls.bm25_index.load()

                print(
                    f"Loaded BM25 index: "
                    f"{len(cls.bm25_index.chunks)} chunks"
                )

            except Exception as e:

                print(
                    f"Failed loading BM25: {e}"
                )

        print("Loading Document Registry...")

        cls.document_registry = (
            DocumentRegistry(
                db_path=Settings.DOCUMENT_DB_PATH
            )
        )

        print("Building Ingestion Pipeline...")

        cls.ingestion_pipeline = (
            IngestionPipeline(
                embedding_provider=cls.embedding_provider,
                vector_store=cls.vector_store,
                bm25_index=cls.bm25_index,
                document_registry=cls.document_registry
            )
        )

        print("Building Retrieval Pipeline...")

        cls.retrieval_pipeline = (
            RetrievalPipeline(
                embedding_provider=cls.embedding_provider,
                vector_store=cls.vector_store,
                bm25_index=cls.bm25_index
            )
        )

        cls.initialized = True

        print("System Ready")

    @classmethod
    def get_embedding_provider(cls):

        cls.initialize()

        return cls.embedding_provider

    @classmethod
    def get_vector_store(cls):

        cls.initialize()

        return cls.vector_store

    @classmethod
    def get_bm25_index(cls):

        cls.initialize()

        return cls.bm25_index

    @classmethod
    def get_document_registry(cls):

        cls.initialize()

        return cls.document_registry

    @classmethod
    def get_ingestion_pipeline(cls):

        cls.initialize()

        return cls.ingestion_pipeline

    @classmethod
    def get_retrieval_pipeline(cls):

        cls.initialize()

        return cls.retrieval_pipeline

    @classmethod
    def shutdown(cls):

        if cls.document_registry:
            try:
                cls.document_registry.close()
            except Exception:
                pass

        cls.initialized = False

        cls.document_registry = None
        cls.embedding_provider = None
        cls.vector_store = None
        cls.bm25_index = None
        cls.ingestion_pipeline = None
        cls.retrieval_pipeline = None