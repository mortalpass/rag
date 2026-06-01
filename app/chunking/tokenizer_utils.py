# app/chunking/tokenizer_utils.py

import re
import tiktoken

enc = tiktoken.get_encoding(
    "cl100k_base"
)


def count_tokens(text: str):

    return len(
        enc.encode(text)
    )


def split_camel_case(text):

    return re.sub(
        r"([a-z])([A-Z])",
        r"\1 \2",
        text
    )


def tokenize(text: str):

    text = text.lower()

    tokens = set()

    # =========================
    # preserve original
    # =========================

    raw_tokens = re.findall(
        r"[a-zA-Z0-9_\-./]+",
        text
    )

    for token in raw_tokens:

        tokens.add(token)

        # =========================
        # path split
        # =========================

        path_parts = re.split(
            r"[/.]",
            token
        )

        for p in path_parts:

            if p:
                tokens.add(p)

        # =========================
        # kebab-case
        # =========================

        kebab_parts = token.split("-")

        for p in kebab_parts:

            if p:
                tokens.add(p)

        # =========================
        # snake_case
        # =========================

        snake_parts = token.split("_")

        for p in snake_parts:

            if p:
                tokens.add(p)

        # =========================
        # camelCase
        # =========================

        camel = split_camel_case(
            token
        )

        camel_parts = camel.split()

        for p in camel_parts:

            if p:
                tokens.add(
                    p.lower()
                )

    # remove short noise
    tokens = [
        t
        for t in tokens
        if len(t) > 1
    ]

    return list(tokens)