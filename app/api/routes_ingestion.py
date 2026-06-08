from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil

from app.pipelines.ingestion_pipeline import IngestionPipeline

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


pipeline: IngestionPipeline = None


def init_pipeline(p: IngestionPipeline):
    global pipeline
    pipeline = p


# =========================
# 上传并入库
# =========================
@router.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):

    if not file.filename.endswith(".md"):
        raise HTTPException(
            status_code=400,
            detail="only .md supported"
        )

    save_path = Path("data/raw") / file.filename

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks = pipeline.ingest_markdown(str(save_path))

    return {
        "file": file.filename,
        "chunks": len(chunks)
    }


# =========================
# 删除文档（按文件路径）
# =========================
@router.delete("/delete")
def delete_file(file_path: str):

    result = pipeline.delete_markdown(file_path)

    return {
        "file": file_path,
        "deleted": result
    }


# =========================
# 删除文档（按 doc_id）
# =========================
@router.delete("/delete/{doc_id}")
def delete_doc(doc_id: str):

    result = pipeline.delete_document(doc_id)

    return {
        "doc_id": doc_id,
        "deleted": result
    }


# =========================
# 重新构建某个文件
# =========================
@router.post("/reindex")
def reindex(file_path: str):

    pipeline.delete_markdown(file_path)
    chunks = pipeline.ingest_markdown(file_path)

    return {
        "file": file_path,
        "chunks": len(chunks)
    }