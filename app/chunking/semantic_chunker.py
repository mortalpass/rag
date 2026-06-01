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
from datetime import datetime

from hashlib import sha256

from app.schemas.chunk_metadata import (
    ChunkMetadata
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

        denominator = (
                np.linalg.norm(v1)
                * np.linalg.norm(v2)
        )

        if denominator == 0:
            return 0.0

        return float(
            np.dot(v1, v2) / denominator
        )

    def build_metadata(
            self,
            *,
            section,
            text,
            chunk_index,
            start_block,
            end_block
    ):

        return ChunkMetadata(

            # =====================
            # document
            # =====================

            doc_id=getattr(
                section,
                "doc_id",
                "unknown_doc"
            ),

            source=getattr(
                section,
                "source",
                ""
            ),

            source_type="markdown",

            document_title=getattr(
                section,
                "root_title",
                section.title
            ),

            # =====================
            # hierarchy
            # =====================

            section_title=section.title,

            section_level=section.level,

            section_path=section.path,

            parent_section=(
                section.path[-2]
                if len(section.path) > 1
                else None
            ),

            # =====================
            # chunk
            # =====================

            chunk_index=chunk_index,

            chunk_type="semantic_region",

            # =====================
            # positioning
            # =====================

            start_block=start_block,

            end_block=end_block,

            # =====================
            # statistics
            # =====================

            token_count=count_tokens(text),

            char_count=len(text),

            # =====================
            # ast metadata
            # =====================

            ast_type="paragraph",

            has_code="```" in text,

            has_table="|" in text,

            has_list="- " in text,

            has_link="http" in text,

            # =====================
            # code metadata
            # =====================

            programming_language=(
                "python"
                if "def " in text
                else None
            ),

            symbols=[],

            imports=[],

            frameworks=[],

            # =====================
            # embedding
            # =====================

            embedding_model=(
                Settings.EMBEDDING_MODEL_NAME
            ),

            embedding_dimension=384,

            # =====================
            # indexing
            # =====================

            content_hash=sha256(
                text.encode("utf-8")
            ).hexdigest(),

            # =====================
            # timestamps
            # =====================

            created_at=(
                datetime.utcnow()
                .isoformat()
            )
        )

    def enrich_content(
            self,
            section,
            text
    ):

        header_path = " > ".join(
            section.path
        )

        return f"""
    Document: {section.title}

    Section: {header_path}

    Content:
    {text}
    """.strip()

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

            metadata = self.build_metadata(

                section=section,

                text=text,

                chunk_index=idx,

                start_block=0,

                end_block=len(region) - 1
            )

            enriched_text = self.enrich_content(
                section,
                text
            )

            chunk = Chunk(
                chunk_id=str(uuid4()),

                chunk_type="semantic_region",

                title=section.title,

                path=section.path,

                content=enriched_text,

                token_count=count_tokens(text),

                metadata=metadata
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

        regions = []

        current_region = []

        current_tokens = 0

        for block in blocks:

            content = block["content"]

            tokens = count_tokens(content)

            is_code = (
                    "```" in content
            )

            # code block 单独 chunk
            if is_code:

                if current_region:
                    regions.append(
                        current_region
                    )

                regions.append([block])

                current_region = []

                current_tokens = 0

                continue

            # token overflow
            if (
                    current_tokens + tokens
                    > Settings.MAX_TOKENS
            ):

                if current_region:
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
            regions.append(current_region)

        return regions
