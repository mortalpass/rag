# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **RAG (Retrieval-Augmented Generation) system** built in Python that implements a hybrid retrieval pipeline. The architecture processes Markdown documents through:

1. **Parsing**: Markdown AST parsing with section hierarchy
2. **Chunking**: Semantic chunking using BGE embeddings to detect topic shifts
3. **Indexing**: Dual-index (BM25 + dense embeddings) for hybrid retrieval
4. **Retrieval**: BM25 + dense retrieval → RRF fusion → BGE cross-encoder reranking
5. **Evaluation**: Result display and scoring visualization

## Architecture

### Core Pipeline Flow

```
Markdown Input
    ↓
MarkdownASTParser (Section Tree)
    ↓
SemanticChunker (Topic-shift detection + Token control)
    ↓
Chunk Export (JSON)
    ↓
HybridRetriever
    ├─ BM25Index (jieba tokenization)
    ├─ DenseIndex (BGE embeddings)
    ├─ RRF fusion
    └─ BGEReranker (cross-encoder)
    ↓
Top-K Results
```

### Key Components

**Chunking** (`app/chunking/`):
- `ast_parser.py`: Markdown-it based parser creating hierarchical sections
- `semantic_chunker.py`: Embedding-based topic shift detection with token control (120-1000 tokens, threshold 0.72)
- `chunk_models.py`: `Chunk` dataclass with path, content, metadata

**Retrieval** (`app/retrieval/`):
- `hybrid_retriever.py`: Orchestrates BM25 + dense retrieval, RRF fusion, reranking
- `bm25/`: BM25Okapi with jieba tokenization for Chinese text
- `dense/`: Dense index using BGE embeddings (Xorbits/bge-small-en-v1.5)
- `fusion/rrf.py`: Reciprocal Rank Fusion (k=60) for result merging
- `rerank/bge_reranker.py`: Cross-encoder reranker (Xorbits/bge-reranker-base)

**Embedding & Models** (`app/embedding/`, `app/models/`):
- `bge_embedding.py`: SentenceTransformer wrapper
- `model_manager.py`: ModelScope snapshot download with LRU caching

**Storage & Evaluation**:
- `storage/chunk_exporter.py`: JSON export of chunks
- `evaluation/retrieval_eval.py`: Formatted result display with scores

### Configuration

`app/config/settings.py`:
- `EMBEDDING_MODEL_NAME`: "Xorbits/bge-small-en-v1.5"
- `RERANKER_MODEL_NAME`: "Xorbits/bge-reranker-base"
- `TOPIC_SHIFT_THRESHOLD`: 0.72 (cosine similarity)
- `MAX_TOKENS`: 1000, `MIN_TOKENS`: 120
- `RETRIEVAL_TOP_K`: 5, `RERANK_TOP_K`: 3

## Common Commands

### Setup
```bash
pip install -r requirements.txt
```

### Run
```bash
python main.py
```
This builds chunks from `data/raw/test.md`, indexes them, and starts an interactive query loop.

### Build Chunks Only
```bash
python -c "from main import build_chunks; build_chunks()"
```

## Data Flow

- **Input**: `data/raw/test.md` (Markdown documents)
- **Output**: `data/processed/chunks.json` (structured chunks with metadata)
- **Query Interface**: Interactive CLI via `main.py`

## Dependencies

Key libraries:
- `markdown-it-py`: Markdown parsing
- `sentence-transformers`: BGE embeddings
- `transformers`: BGE reranker
- `rank-bm25`: BM25 indexing
- `jieba`: Chinese tokenization
- `modelscope`: Model downloads
