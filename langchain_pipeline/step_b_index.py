"""
STEP B -- Embedding & hybrid indexing into Qdrant  (LangChain track)

    python -m langchain_pipeline.step_b_index [--recreate]

RetrievalMode.HYBRID requires BOTH an `embedding` (dense) and a
`sparse_embedding` (FastEmbedSparse). The collection must be created in HYBRID
mode -- you cannot bolt sparse vectors onto an existing dense-only collection.
Once created in HYBRID you may query in DENSE, SPARSE or HYBRID mode freely.

Note: this track uses its own collection name (`<COLLECTION>_lc`) so the two
tracks can coexist without clobbering each other.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

from common import config

LC_COLLECTION = f"{config.COLLECTION}_lc"


def load_documents() -> list[Document]:
    if not config.NODES_CACHE.exists():
        raise SystemExit("Run langchain_pipeline.step_a_ingest first.")
    docs = []
    for line in config.NODES_CACHE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        docs.append(Document(page_content=d["text"], metadata=d["metadata"]))
    return docs


@lru_cache(maxsize=None)
def get_embeddings() -> OpenAIEmbeddings:
    config.require_openai()
    return OpenAIEmbeddings(model=config.EMBED_MODEL,
                            api_key=config.OPENAI_API_KEY,
                            chunk_size=config.EMBED_BATCH)


@lru_cache(maxsize=None)
def get_sparse() -> FastEmbedSparse:
    return FastEmbedSparse(model_name=config.SPARSE_MODEL)


@lru_cache(maxsize=None)
def get_store(recreate: bool = False) -> QdrantVectorStore:
    """Attach to an existing hybrid collection (query path).

    Cached per process: rebuilding this means reloading the local sparse
    ONNX model and reconnecting to Qdrant, which is wasted work across
    repeated calls in the same process (e.g. one per chatbot turn).
    """
    client = QdrantClient(url=config.QDRANT_URL)
    if recreate and client.collection_exists(LC_COLLECTION):
        print(f"  dropping '{LC_COLLECTION}'")
        client.delete_collection(LC_COLLECTION)
    return QdrantVectorStore.from_existing_collection(
        collection_name=LC_COLLECTION,
        embedding=get_embeddings(),
        sparse_embedding=get_sparse(),
        retrieval_mode=RetrievalMode.HYBRID,
        url=config.QDRANT_URL,
        vector_name="dense",
        sparse_vector_name="sparse",
    )


def main(recreate: bool = False) -> QdrantVectorStore:
    config.banner("STEP B (LangChain) -- embed & index (dense + sparse)")
    docs = load_documents()
    print(f"  {len(docs)} documents to index")

    client = QdrantClient(url=config.QDRANT_URL)
    if recreate and client.collection_exists(LC_COLLECTION):
        print(f"  dropping '{LC_COLLECTION}'")
        client.delete_collection(LC_COLLECTION)

    store = QdrantVectorStore.from_documents(
        docs,
        ids=[d.metadata["chunk_id"] for d in docs],   # deterministic -> upsert, not duplicate
        embedding=get_embeddings(),
        sparse_embedding=get_sparse(),
        retrieval_mode=RetrievalMode.HYBRID,       # <-- must be set at CREATE time
        url=config.QDRANT_URL,
        collection_name=LC_COLLECTION,
        vector_name="dense",
        sparse_vector_name="sparse",
        batch_size=64,
    )

    info = client.get_collection(LC_COLLECTION)
    print(f"\n  collection '{LC_COLLECTION}': {info.points_count} points")
    print(f"  sparse vectors: {list((info.config.params.sparse_vectors or {}).keys())}")
    if not (info.config.params.sparse_vectors or {}):
        print("  ! NO SPARSE VECTORS -- re-run with --recreate.")
    return store


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true")
    main(**vars(ap.parse_args()))
