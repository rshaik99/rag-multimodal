"""
Single source of truth for the RAG lab.

Rule of thumb learned the hard way: the embedding model MUST be identical at
index time and query time. Pinning it here (instead of inline in each script)
is the cheapest way to guarantee that.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------- paths
DATA_DIR = ROOT / "data"          # put your PDFs here
ASSETS_DIR = ROOT / "assets"      # extracted figures (Phase 2)
NODES_CACHE = ROOT / "storage" / "nodes.jsonl"   # Step A output, inspectable

for _p in (DATA_DIR, ASSETS_DIR, NODES_CACHE.parent):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------- models
# Current as of 2026-07. Verify against provider docs before a production run.
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072          # Matryoshka: set 1024 to cut storage ~3x for ~1-2% recall
SPARSE_MODEL = "Qdrant/bm25"   # FastEmbed ONNX. Alt: "prithivida/Splade_PP_en_v1"

LLM_MODEL = "gpt-4.1"
ROUTER_MODEL = "gpt-4.1-mini"     # cheap classifier for Phase 2 routing
VISION_MODEL = "gpt-4.1"          # or "claude-sonnet-5" via Anthropic

# rerank-v3.5 was deprecated 2026-07-01; requests auto-route to rerank-4-fast
# from 2026-08-01. Use the v4 names explicitly.
COHERE_RERANK_MODEL = "rerank-v4.0-fast"      # "rerank-v4.0-pro" for max precision
LOCAL_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# ---------------------------------------------------------------- infra
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_lab")

# ---------------------------------------------------------------- knobs
CHUNK_SIZE = 1024          # tokens. 256-512 = higher precision, more fragmentation
CHUNK_OVERLAP = 128
EMBED_BATCH = 100          # too high -> OpenAI rate limits -> "hung" indexing

RETRIEVE_TOP_K = 20        # per retrieval leg (dense / sparse)
FUSED_TOP_N = 40           # candidates handed to the reranker
FINAL_TOP_K = 5            # chunks that reach the LLM
RERANK_SCORE_FLOOR = 0.30  # below this -> drop. This is your "I don't know" guard.

TEMPERATURE = 0.0          # no upside to sampling during synthesis


def require_openai() -> None:
    if not OPENAI_API_KEY:
        raise SystemExit(
            "OPENAI_API_KEY is not set.\n"
            "  cp .env.example .env  &&  edit .env"
        )


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}")
