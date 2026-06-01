from typing import Optional

from pydantic import BaseModel


class ChunkMetadata(BaseModel):

    # =====================
    # document
    # =====================

    doc_id: str

    source: str

    source_type: str

    document_title: str

    # =====================
    # hierarchy
    # =====================

    section_title: str

    section_level: int

    section_path: list[str]

    parent_section: Optional[str] = None

    # =====================
    # chunk
    # =====================

    chunk_index: int

    chunk_type: str

    # =====================
    # positioning
    # =====================

    start_block: int

    end_block: int

    # =====================
    # statistics
    # =====================

    token_count: int

    char_count: int

    # =====================
    # ast metadata
    # =====================

    ast_type: Optional[str] = None

    heading_level: Optional[int] = None

    has_code: bool = False

    has_table: bool = False

    has_list: bool = False

    has_link: bool = False

    semantic_role: Optional[str] = None

    # =====================
    # code metadata
    # =====================

    programming_language: Optional[str] = None

    symbols: list[str] = []

    imports: list[str] = []

    code_entity_type: Optional[str] = None

    function_name: Optional[str] = None

    class_name: Optional[str] = None

    has_docstring: bool = False

    frameworks: list[str] = []

    # =====================
    # embedding
    # =====================

    embedding_model: str

    embedding_dimension: int

    # =====================
    # indexing
    # =====================

    content_hash: str

    parent_chunk_id: Optional[str] = None

    # =====================
    # timestamps
    # =====================

    created_at: str