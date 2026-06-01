from dataclasses import dataclass

from app.schemas.chunk_metadata import ChunkMetadata


@dataclass
class Chunk:

    chunk_id: str

    chunk_type: str

    title: str

    path: list

    content: str

    token_count: int

    metadata: ChunkMetadata

    def to_dict(self):

        return {

            "chunk_id": self.chunk_id,

            "chunk_type": self.chunk_type,

            "title": self.title,

            "path": self.path,

            "content": self.content,

            "token_count": self.token_count,

            "metadata": (
                self.metadata.model_dump()
            )
        }