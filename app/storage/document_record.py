from dataclasses import dataclass

@dataclass
class DocumentRecord:
    doc_id: str
    source_file: str
    title: str
    content_hash: str
    chunk_count: int
    version: int
    created_at: str
    updated_at: str