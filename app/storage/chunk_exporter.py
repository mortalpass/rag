# app/storage/chunk_exporter.py

import json

from pathlib import Path


class ChunkExporter:

    @staticmethod
    def export_to_json(
        chunks,
        output_path
    ):

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        data = []

        for chunk in chunks:

            item = chunk.to_dict()

            # =====================
            # debug preview
            # =====================

            item["preview"] = (
                chunk.content[:200]
            )

            # =====================
            # debug stats
            # =====================

            item["content_length"] = len(
                chunk.content
            )

            data.append(item)

        output_path.write_text(

            json.dumps(
                data,
                ensure_ascii=False,
                indent=2
            ),

            encoding="utf-8"
        )

        print(
            f"\nSaved {len(chunks)} chunks"
        )

        print(
            f"Output: {output_path}"
        )
