class Settings:
    EMBEDDING_MODEL_NAME = (
        "Xorbits/bge-small-en-v1.5"
    )

    RERANKER_MODEL_NAME = (
        "Xorbits/bge-reranker-base"
    )

    EMBEDDING_PROVIDER = "bge"

    RERANK_TOP_K = 3

    MAX_TOKENS = 1000

    MIN_TOKENS = 120

    TOPIC_SHIFT_THRESHOLD = 0.72

    RETRIEVAL_TOP_K = 5
