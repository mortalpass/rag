from uuid import uuid4

import numpy as np

from app.chunking.chunk_models import Chunk
from app.chunking.tokenizer_utils import (
    count_tokens
)

from app.config.settings import Settings

from app.embedding.providers.base import (
    EmbeddingProvider
)


class SemanticChunker:

    def __init__(
        self,
        embedding_provider: EmbeddingProvider
    ):
        self.embedding_provider = (
            embedding_provider
        )

    def cosine_similarity(
        self,
        v1,
        v2
    ) -> float:

        v1 = np.asarray(v1)
        v2 = np.asarray(v2)

        assert v1.ndim == 1
        assert v2.ndim == 1

        return float(np.dot(v1, v2))

    def chunk_section(
        self,
        section
    ):

        chunks = []

        regions = self.detect_semantic_regions(
            section.blocks
        )

        for idx, region in enumerate(regions):

            text = "\n\n".join(
                [
                    block["content"]
                    for block in region
                ]
            )

            chunk = Chunk(
                chunk_id=str(uuid4()),

                chunk_type="semantic_region",

                title=section.title,

                path=section.path,

                content=text,

                token_count=count_tokens(text),

                metadata={
                    "section_level": section.level,
                    "region_index": idx
                }
            )

            chunks.append(chunk)

        for child in section.children:

            chunks.extend(
                self.chunk_section(child)
            )

        return chunks

    def detect_semantic_regions(
        self,
        blocks
    ):

        if not blocks:
            return []

        contents = [
            block["content"]
            for block in blocks
        ]

        embeddings = (
            self.embedding_provider
            .embed_texts(contents)
        )

        embeddings = np.asarray(
            embeddings
        )

        assert embeddings.ndim == 2

        regions = []

        current_region = [blocks[0]]

        current_tokens = count_tokens(
            blocks[0]["content"]
        )

        for i in range(1, len(blocks)):

            prev_emb = embeddings[i - 1]

            curr_emb = embeddings[i]

            similarity = (
                self.cosine_similarity(
                    prev_emb,
                    curr_emb
                )
            )

            block = blocks[i]

            content = block["content"]

            tokens = count_tokens(
                content
            )

            should_split = False

            if (
                similarity
                < Settings.TOPIC_SHIFT_THRESHOLD
            ):
                should_split = True

            if (
                current_tokens + tokens
                > Settings.MAX_TOKENS
            ):
                should_split = True

            if should_split:

                regions.append(
                    current_region
                )

                current_region = [block]

                current_tokens = tokens

            else:

                current_region.append(
                    block
                )

                current_tokens += tokens

        if current_region:

            regions.append(
                current_region
            )

        return regions
