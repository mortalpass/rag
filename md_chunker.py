import json

from markdown_it import MarkdownIt
from markdown_it.token import Token

from pathlib import Path
from typing import List, Dict

import uuid
import re


class MarkdownKnowledgeChunker:

    def __init__(
        self,
        max_chunk_size: int = 1500,
        overlap: int = 200,
    ):

        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

        self.md = MarkdownIt("commonmark")

    # =====================================================
    # Parse Markdown
    # =====================================================

    def parse_markdown(
        self,
        markdown_text: str,
    ):

        return self.md.parse(markdown_text)

    # =====================================================
    # Semantic Type Detection
    # =====================================================

    def detect_semantic_type(
        self,
        title_path: List[str],
        content: str,
    ):

        full_text = " ".join(title_path).lower()

        if "quick start" in full_text:
            return "quickstart"

        if "installation" in full_text:
            return "installation"

        if "contributing" in full_text:
            return "contribution"

        if "license" in full_text:
            return "license"

        if "api" in full_text:
            return "api_reference"

        if "example" in full_text:
            return "example"

        return "general"

    # =====================================================
    # Token Count Estimate
    # =====================================================

    def estimate_tokens(
        self,
        text: str,
    ):

        return int(len(text) / 4)

    # =====================================================
    # Embedding Text Builder
    # =====================================================

    def build_embedding_text(
        self,
        title_path,
        content_type,
        text,
    ):

        return f"""
Section:
{' > '.join(title_path)}

Content Type:
{content_type}

Content:
{text}
""".strip()

    # =====================================================
    # Long Text Split
    # =====================================================

    def split_large_text(
        self,
        text: str,
    ):

        if len(text) <= self.max_chunk_size:
            return [text]

        chunks = []

        start = 0

        while start < len(text):

            end = start + self.max_chunk_size

            chunk = text[start:end]

            chunks.append(chunk)

            start = end - self.overlap

        return chunks

    # =====================================================
    # Create Knowledge Object
    # =====================================================

    def create_knowledge_object(
        self,
        source: str,
        title_path: List[str],
        heading_level: int,
        content_type: str,
        text: str,
    ):

        semantic_type = self.detect_semantic_type(
            title_path,
            text,
        )

        embedding_text = self.build_embedding_text(
            title_path=title_path,
            content_type=content_type,
            text=text,
        )

        knowledge_object = {

            # ==========================================
            # IDs
            # ==========================================
            "doc_id": source,
            "chunk_id": str(uuid.uuid4()),

            # ==========================================
            # Source
            # ==========================================
            "source": source,
            "source_type": "markdown",

            # ==========================================
            # Structure
            # ==========================================
            "title_path": title_path.copy(),

            "title": (
                title_path[-1]
                if title_path
                else "ROOT"
            ),

            "heading_level": heading_level,

            # ==========================================
            # Content
            # ==========================================
            "content_type": content_type,

            "semantic_type": semantic_type,

            "text": text.strip(),

            "embedding_text": embedding_text,

            # ==========================================
            # Metrics
            # ==========================================
            "char_count": len(text),

            "token_count_estimate": self.estimate_tokens(
                text
            ),

            # ==========================================
            # Retrieval Metadata
            # ==========================================
            "metadata": {

                "is_code": (
                    content_type == "code"
                ),

                "is_table": (
                    content_type == "table"
                ),

                "has_headings": (
                    len(title_path) > 0
                ),

                "section_depth": len(title_path),
            }
        }

        return knowledge_object

    # =====================================================
    # Build Knowledge Objects
    # =====================================================

    def build_knowledge_objects(
        self,
        tokens: List[Token],
        source: str,
    ):

        knowledge_objects = []

        title_stack = []

        current_text = []
        current_size = 0

        current_heading_level = 0

        i = 0

        # ==========================================
        # Flush Current Text Chunk
        # ==========================================

        def flush_text_chunk():

            nonlocal current_text
            nonlocal current_size

            if not current_text:
                return

            text = "\n".join(current_text).strip()

            if not text:
                return

            split_chunks = self.split_large_text(
                text
            )

            for split_text in split_chunks:

                obj = self.create_knowledge_object(
                    source=source,
                    title_path=title_stack,
                    heading_level=current_heading_level,
                    content_type="text",
                    text=split_text,
                )

                knowledge_objects.append(obj)

            current_text = []
            current_size = 0

        # ==========================================
        # Main Loop
        # ==========================================

        while i < len(tokens):

            token = tokens[i]

            # ======================================
            # Heading
            # ======================================

            if token.type == "heading_open":

                flush_text_chunk()

                level = int(token.tag[1])

                inline_token = tokens[i + 1]

                title = inline_token.content.strip()

                title_stack[:] = title_stack[:level - 1]

                title_stack.append(title)

                current_heading_level = level

                # TITLE本身也是知识对象
                title_obj = self.create_knowledge_object(
                    source=source,
                    title_path=title_stack,
                    heading_level=level,
                    content_type="title",
                    text=title,
                )

                knowledge_objects.append(title_obj)

                i += 3
                continue

            # ======================================
            # Code Fence
            # ======================================

            if token.type == "fence":

                flush_text_chunk()

                code_text = token.content.strip()

                code_chunks = self.split_large_text(
                    code_text
                )

                for code in code_chunks:

                    obj = self.create_knowledge_object(
                        source=source,
                        title_path=title_stack,
                        heading_level=current_heading_level,
                        content_type="code",
                        text=code,
                    )

                    knowledge_objects.append(obj)

                i += 1
                continue

            # ======================================
            # Paragraph
            # ======================================

            if token.type == "inline":

                text = token.content.strip()

                if text:

                    if (
                        current_size + len(text)
                        > self.max_chunk_size
                    ):

                        flush_text_chunk()

                    current_text.append(text)

                    current_size += len(text)

            i += 1

        flush_text_chunk()

        return knowledge_objects

    # =====================================================
    # Process File
    # =====================================================

    def process(
        self,
        file_path: str,
    ):

        markdown_text = Path(
            file_path
        ).read_text(
            encoding="utf-8"
        )

        tokens = self.parse_markdown(
            markdown_text
        )

        objects = self.build_knowledge_objects(
            tokens=tokens,
            source=Path(file_path).name,
        )

        return objects

    def save_chunks_to_json(self, chunks, output_path="debug_chunks.json"):
        """
        保存 chunk 到 json 文件
        """

        # 写入 json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                chunks,
                f,

                # 保留中文
                ensure_ascii=False,

                # 美化缩进
                indent=2
            )

        print(f"Chunk 已保存到: {output_path}")


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    chunker = MarkdownKnowledgeChunker(
        max_chunk_size=1500,
        overlap=200,
    )

    objects = chunker.process("data/test.md")

    # 保存 json debug 文件
    chunker.save_chunks_to_json(
        objects,
        "output/md_chunks.json"
    )

    print("=" * 100)
    print(f"TOTAL OBJECTS: {len(objects)}")
    print("=" * 100)

    for i, obj in enumerate(objects[:10]):

        print(f"\nOBJECT {i+1}")
        print("-" * 100)

        print("TITLE PATH:")
        print(" > ".join(obj["title_path"]))

        print("\nCONTENT TYPE:")
        print(obj["content_type"])

        print("\nSEMANTIC TYPE:")
        print(obj["semantic_type"])

        print("\nTOKEN ESTIMATE:")
        print(obj["token_count_estimate"])

        print("\nTEXT:")
        print(obj["text"][:500])

        print("\n")