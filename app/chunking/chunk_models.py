from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Chunk:

    chunk_id: str

    chunk_type: str

    title: str

    path: List[str]

    content: str

    token_count: int

    metadata: Dict = field(default_factory=dict)

    def to_dict(self):

        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type,
            "title": self.title,
            "path": self.path,
            "content": self.content,
            "token_count": self.token_count,
            "metadata": self.metadata
        }
