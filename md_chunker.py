from markdown_it import MarkdownIt
from markdown_it.token import Token

from typing import List, Dict
from pathlib import Path
import uuid


class MarkdownASTChunker:

    def __init__(
        self,
        max_chunk_size: int = 1500,
        overlap: int = 200,
    ):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

        self.md = MarkdownIt("commonmark", {
            "html": False,
            "linkify": True,
            "typographer": False,
        })

    def parse_markdown(
        self,
        markdown_text: str,
    ) -> List[Token]:

        tokens = self.md.parse(markdown_text)

        return tokens

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

    def create_chunk(
        self,
        text: str,
        title_path: List[str],
        content_type: str,
        source: str,
    ):

        return {
            "chunk_id": str(uuid.uuid4()),
            "source": source,
            "title_path": title_path.copy(),
            "content_type": content_type,
            "text": text.strip(),
            "char_count": len(text),
        }

    def build_chunks(
        self,
        tokens: List[Token],
        source: str,
    ):

        chunks = []

        title_stack = []

        current_text = []
        current_size = 0

        i = 0

        def flush_text_chunk():

            nonlocal current_text
            nonlocal current_size

            if not current_text:
                return

            text = "\n".join(current_text).strip()

            if not text:
                return

            split_chunks = self.split_large_text(text)

            for split_text in split_chunks:

                chunks.append(
                    self.create_chunk(
                        text=split_text,
                        title_path=title_stack,
                        content_type="text",
                        source=source,
                    )
                )

            current_text = []
            current_size = 0

        while i < len(tokens):

            token = tokens[i]

            # ==================================
            # Heading
            # ==================================
            if token.type == "heading_open":

                flush_text_chunk()

                level = int(token.tag[1])

                inline_token = tokens[i + 1]

                title = inline_token.content.strip()

                # 修正标题层级
                title_stack[:] = title_stack[:level - 1]

                title_stack.append(title)

                i += 3
                continue

            # ==================================
            # Fence Code Block
            # ==================================
            if token.type == "fence":

                flush_text_chunk()

                code_text = token.content.strip()

                code_chunks = self.split_large_text(code_text)

                for code in code_chunks:

                    chunks.append(
                        self.create_chunk(
                            text=code,
                            title_path=title_stack,
                            content_type="code",
                            source=source,
                        )
                    )

                i += 1
                continue

            # ==================================
            # Table
            # ==================================
            if token.type == "table_open":

                flush_text_chunk()

                table_buffer = []

                while i < len(tokens):

                    t = tokens[i]

                    if t.type == "inline":
                        table_buffer.append(t.content)

                    if t.type == "table_close":
                        break

                    i += 1

                table_text = "\n".join(table_buffer)

                chunks.append(
                    self.create_chunk(
                        text=table_text,
                        title_path=title_stack,
                        content_type="table",
                        source=source,
                    )
                )

                i += 1
                continue

            # ==================================
            # Paragraph / Inline
            # ==================================
            if token.type == "inline":

                text = token.content.strip()

                if text:

                    if current_size + len(text) > self.max_chunk_size:

                        flush_text_chunk()

                    current_text.append(text)

                    current_size += len(text)

            i += 1

        flush_text_chunk()

        return chunks

    def process(
        self,
        file_path: str,
    ):

        markdown_text = Path(file_path).read_text(
            encoding="utf-8"
        )

        tokens = self.parse_markdown(markdown_text)

        chunks = self.build_chunks(
            tokens=tokens,
            source=Path(file_path).name,
        )

        return chunks


if __name__ == "__main__":

    chunker = MarkdownASTChunker(
        max_chunk_size=1500,
        overlap=200,
    )

    chunks = chunker.process("test.md")

    print("=" * 100)
    print(f"TOTAL CHUNKS: {len(chunks)}")
    print("=" * 100)

    for i, chunk in enumerate(chunks):

        print(f"\nCHUNK {i+1}")
        print("-" * 100)

        print("TITLE PATH:")
        print(" > ".join(chunk["title_path"]))

        print("\nCONTENT TYPE:")
        print(chunk["content_type"])

        print("\nCHAR COUNT:")
        print(chunk["char_count"])

        print("\nTEXT:")
        print(chunk["text"][:1000])

        print("\n")