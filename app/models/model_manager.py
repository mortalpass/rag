from functools import lru_cache

from modelscope import snapshot_download

from app.config.settings import (
    Settings
)


class ModelManager:

    @staticmethod
    @lru_cache(maxsize=None)
    def get_embedding_model_path():

        return snapshot_download(
            Settings.EMBEDDING_MODEL_NAME
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def get_reranker_model_path():

        return snapshot_download(
            Settings.RERANKER_MODEL_NAME
        )