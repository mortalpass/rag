from uuid import uuid4
import numpy as np

from app.chunking.chunk_models import Chunk

from app.chunking.tokenizer_utils import (
    count_tokens
)

from app.config.settings import Settings

from app.embedding.bge_embedding import (
    BGEEmbedding
)


class SemanticChunker:

    def __init__(self):

        self.embedding_model = BGEEmbedding()

    def cosine_similarity(
        self,
        v1,
        v2
    ):

        return np.dot(v1, v2)

    def chunk_section(self, section):

        chunks = []

        regions = self.detect_semantic_regions(
            section.blocks
        )

        for idx, region in enumerate(regions):

            text = "\n\n".join([
                b["content"]
                for b in region
            ])

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

        regions = []

        current_region = [blocks[0]]

        prev_emb = self.embedding_model.embed_text(
            blocks[0]["content"]
        )

        current_tokens = count_tokens(
            blocks[0]["content"]
        )

        for block in blocks[1:]:

            content = block["content"]

            tokens = count_tokens(content)

            emb = self.embedding_model.embed_text(
                content
            )

            similarity = self.cosine_similarity(
                prev_emb,
                emb
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

                regions.append(current_region)

                current_region = [block]

                current_tokens = tokens

            else:

                current_region.append(block)

                current_tokens += tokens

            prev_emb = emb

        if current_region:
            regions.append(current_region)

        return regions
