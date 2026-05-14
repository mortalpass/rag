from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
import numpy as np
from modelscope import snapshot_download

model_dir = snapshot_download(
    'AI-ModelScope/bge-small-en-v1.5'
)

model = SentenceTransformer(model_dir)
# model = SentenceTransformer(
#     "BAAI/bge-small-en-v1.5"
# )


def embed_text(text: str):

    return model.encode(
        text,
        normalize_embeddings=True
    )


def cosine_similarity(v1, v2):

    return np.dot(v1, v2) / (
        norm(v1) * norm(v2)
    )