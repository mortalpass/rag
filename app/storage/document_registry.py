import sqlite3

from app.storage.document_record import (
    DocumentRecord
)


class DocumentRegistry:

    def __init__(
        self,
        db_path: str = "./documents.db"
    ):

        self.db_path = db_path

        self.conn = sqlite3.connect(
            db_path,
            check_same_thread=False
        )

        self.conn.execute(
            "PRAGMA foreign_keys = ON"
        )

        self._create_table()

    def _create_table(self):

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (

                doc_id TEXT PRIMARY KEY,

                source_file TEXT NOT NULL UNIQUE,

                title TEXT NOT NULL,

                content_hash TEXT NOT NULL,

                chunk_count INTEGER NOT NULL,

                version INTEGER NOT NULL,

                created_at TEXT NOT NULL,

                updated_at TEXT NOT NULL
            )
            """
        )

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (

                chunk_id TEXT PRIMARY KEY,

                doc_id TEXT NOT NULL,

                FOREIGN KEY(doc_id)
                REFERENCES documents(doc_id)
                ON DELETE CASCADE
            )
            """
        )

        self.conn.commit()

    # =========================
    # Document CRUD
    # =========================

    def upsert(
        self,
        record: DocumentRecord
    ):

        self.conn.execute(
            """
            INSERT OR REPLACE INTO documents (

                doc_id,
                source_file,
                title,
                content_hash,
                chunk_count,
                version,
                created_at,
                updated_at

            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.doc_id,
                record.source_file,
                record.title,
                record.content_hash,
                record.chunk_count,
                record.version,
                record.created_at,
                record.updated_at
            )
        )

        self.conn.commit()

    def get(
        self,
        doc_id: str
    ):

        cursor = self.conn.execute(
            """
            SELECT *

            FROM documents

            WHERE doc_id = ?
            """,
            (doc_id,)
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_record(row)

    def get_by_path(
        self,
        source_file: str
    ):

        cursor = self.conn.execute(
            """
            SELECT *

            FROM documents

            WHERE source_file = ?
            """,
            (source_file,)
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_record(row)

    def list_documents(self):

        cursor = self.conn.execute(
            """
            SELECT *

            FROM documents

            ORDER BY updated_at DESC
            """
        )

        rows = cursor.fetchall()

        return [
            self._row_to_record(row)
            for row in rows
        ]

    def delete(
        self,
        doc_id: str
    ):

        self.conn.execute(
            """
            DELETE FROM documents

            WHERE doc_id = ?
            """,
            (doc_id,)
        )

        self.conn.commit()

    def exists(
        self,
        doc_id: str
    ) -> bool:

        cursor = self.conn.execute(
            """
            SELECT 1

            FROM documents

            WHERE doc_id = ?
            """,
            (doc_id,)
        )

        return cursor.fetchone() is not None

    def count(self):

        cursor = self.conn.execute(
            """
            SELECT COUNT(*)

            FROM documents
            """
        )

        return cursor.fetchone()[0]

    # =========================
    # Chunk Mapping
    # =========================

    def upsert_chunk_mapping(
        self,
        doc_id: str,
        chunk_ids: list[str]
    ):

        self.conn.execute(
            """
            DELETE FROM document_chunks

            WHERE doc_id = ?
            """,
            (doc_id,)
        )

        self.conn.executemany(
            """
            INSERT INTO document_chunks(
                chunk_id,
                doc_id
            )
            VALUES (?, ?)
            """,
            [
                (
                    chunk_id,
                    doc_id
                )
                for chunk_id in chunk_ids
            ]
        )

        self.conn.commit()

    def get_chunk_ids(
        self,
        doc_id: str
    ) -> list[str]:

        cursor = self.conn.execute(
            """
            SELECT chunk_id

            FROM document_chunks

            WHERE doc_id = ?
            """,
            (doc_id,)
        )

        rows = cursor.fetchall()

        return [
            row[0]
            for row in rows
        ]

    def delete_chunk_mapping(
        self,
        doc_id: str
    ):

        self.conn.execute(
            """
            DELETE FROM document_chunks

            WHERE doc_id = ?
            """,
            (doc_id,)
        )

        self.conn.commit()

    # =========================
    # Versioning
    # =========================

    def next_version(
        self,
        doc_id: str
    ) -> int:

        record = self.get(doc_id)

        if record is None:
            return 1

        return record.version + 1

    # =========================
    # Internal Helpers
    # =========================

    @staticmethod
    def _row_to_record(
        row
    ) -> DocumentRecord:

        return DocumentRecord(
            doc_id=row[0],
            source_file=row[1],
            title=row[2],
            content_hash=row[3],
            chunk_count=row[4],
            version=row[5],
            created_at=row[6],
            updated_at=row[7]
        )

    def close(self):

        self.conn.close()