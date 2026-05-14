from uuid import uuid4

from embedding_utils import (
    embed_text,
    cosine_similarity
)

from tokenizer_utils import count_tokens

from chunk_models import Chunk


MAX_TOKENS = 1000

MIN_TOKENS = 120

TOPIC_SHIFT_THRESHOLD = 0.72


class SemanticChunker:

    def chunk_section(self, section):

        chunks = []

        semantic_regions = self.detect_semantic_regions(
            section.blocks
        )

        semantic_regions = self.merge_small_regions(
            semantic_regions
        )

        for idx, region in enumerate(semantic_regions):

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

        # children recursion

        for child in section.children:

            chunks.extend(
                self.chunk_section(child)
            )

        return chunks

    # ---------------------------------
    # semantic region detection
    # ---------------------------------

    def detect_semantic_regions(self, blocks):

        if not blocks:
            return []

        regions = []

        current_region = [blocks[0]]

        prev_embedding = embed_text(
            blocks[0]["content"]
        )

        current_tokens = count_tokens(
            blocks[0]["content"]
        )

        for block in blocks[1:]:

            content = block["content"]

            tokens = count_tokens(content)

            # -------------------------
            # code/table 强制独立
            # -------------------------

            if block["type"] in [
                "code",
                "table"
            ]:

                if current_region:
                    regions.append(current_region)

                regions.append([block])

                current_region = []

                current_tokens = 0

                continue

            embedding = embed_text(content)

            similarity = cosine_similarity(
                prev_embedding,
                embedding
            )

            # -------------------------
            # topic shift
            # -------------------------

            should_split = False

            if similarity < TOPIC_SHIFT_THRESHOLD:
                should_split = True

            if (
                current_tokens + tokens
                > MAX_TOKENS
            ):
                should_split = True

            if should_split:

                regions.append(current_region)

                current_region = [block]

                current_tokens = tokens

            else:

                current_region.append(block)

                current_tokens += tokens

            prev_embedding = embedding

        if current_region:
            regions.append(current_region)

        return regions

    # ---------------------------------
    # merge tiny regions
    # ---------------------------------

    def merge_small_regions(
        self,
        regions
    ):

        merged = []

        current = []

        current_tokens = 0

        for region in regions:

            text = "\n\n".join([
                b["content"]
                for b in region
            ])

            tokens = count_tokens(text)

            if (
                current_tokens + tokens
                < MIN_TOKENS
            ):

                current.extend(region)

                current_tokens += tokens

            else:

                if current:
                    merged.append(current)

                current = region

                current_tokens = tokens

        if current:
            merged.append(current)

        return merged