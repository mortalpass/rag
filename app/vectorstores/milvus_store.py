import json

from typing import List, Dict, Any

from pymilvus import (
    MilvusClient,
    DataType
)

from app.vectorstores.base import BaseVectorStore


class MilvusStore(BaseVectorStore):

    def __init__(
        self,
        db_path: str = "./milvus.db",
        collection_name: str = "rag_chunks",
        dim: int = 384
    ):

        self.client = MilvusClient(uri=db_path)

        self.collection_name = collection_name

        self.dim = dim

        self.create_collection()

    def create_collection(self):

        if self.client.has_collection(
                self.collection_name
        ):
            return

        schema = self.client.create_schema(
            auto_id=False,
            enable_dynamic_field=False
        )

        # primary key
        schema.add_field(
            field_name="id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=128
        )

        # embedding vector
        schema.add_field(
            field_name="vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.dim
        )

        # chunk text
        schema.add_field(
            field_name="content",
            datatype=DataType.VARCHAR,
            max_length=65535
        )

        # metadata json string
        schema.add_field(
            field_name="metadata",
            datatype=DataType.VARCHAR,
            max_length=65535
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema
        )

        index_params = (
            self.client.prepare_index_params()
        )

        index_params.add_index(
            field_name="vector",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128}
        )

        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params
        )

    def insert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]]
    ):

        data = []

        for idx, vector, payload in zip(
                ids,
                vectors,
                payloads
        ):

            # 提取 content
            content = payload.get(
                "content",
                ""
            )

            # 剩余字段全部归 metadata
            metadata = {
                k: v
                for k, v in payload.items()
                if k != "content"
            }

            row = {

                "id": idx,

                "vector": vector,

                "content": content,

                # metadata -> json
                "metadata": json.dumps(
                    metadata,
                    ensure_ascii=False
                )
            }

            data.append(row)

        self.client.insert(
            collection_name=self.collection_name,
            data=data
        )

    def search(
        self,
        query_vector,
        top_k=5
    ):

        results = self.client.search(

            collection_name=self.collection_name,

            data=[query_vector],

            limit=top_k,

            search_params={
                "metric_type": "COSINE"
            },

            output_fields=[
                "content",
                "metadata"
            ]
        )

        parsed_results = []

        for hit in results[0]:

            entity = hit["entity"]

            raw_metadata = entity.get(
                "metadata",
                "{}"
            )

            try:
                metadata = json.loads(
                    raw_metadata
                )

            except Exception:
                metadata = {}

            parsed_results.append({

                "id": hit["id"],

                "distance": hit["distance"],

                "entity": {

                    "content": entity.get(
                        "content",
                        ""
                    ),

                    "metadata": metadata
                }
            })

        return parsed_results