import json
from pathlib import Path


class ChunkExporter:

    @staticmethod
    def export_to_json(
        chunks,
        output_path="chunks.json"
    ):

        data = [
            chunk.to_dict()
            for chunk in chunks
        ]

        Path(output_path).write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2
            ),
            encoding="utf-8"
        )

        print(
            f"\nSaved {len(chunks)} chunks "
            f"to {output_path}"
        )