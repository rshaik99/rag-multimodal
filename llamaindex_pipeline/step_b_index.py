"""
STEP B -- Embedding & hybrid indexing into Qdrant  (LlamaIndex track)

    python -m llamaindex_pipeline.step_b_index [--recreate]

CRITICAL: `enable_hybrid=True` must be set when the collection is CREATED.
Qdrant cannot add a sparse vector to existing points without re-upserting them.
If you index dense-only and later flip the flag, hybrid search silently degrades
to dense-only. Drop and re-create -- that is what --recreate is for.
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache

import qdrant_client
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from common import config


def load_nodes() -> list[TextNode]:
    if not config.NODES_CACHE.exists():
        raise SystemExit("Run step_a_ingest first.")
    nodes = []
    for line in config.NODES_CACHE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        nodes.append(TextNode(
            text=d["text"], id_=d["id"], metadata=d["metadata"],
            excluded_embed_metadata_keys=["file_path", "doc_type", "chunk_key"],
            excluded_llm_metadata_keys=["file_path", "chunk_key"],
        ))
    return nodes


@lru_cache(maxsize=None)
def get_embed_model() -> OpenAIEmbedding:
    config.require_openai()
    return OpenAIEmbedding(
        model=config.EMBED_MODEL,
        embed_batch_size=config.EMBED_BATCH,   # rate-limit safety
        api_key=config.OPENAI_API_KEY,
        # dimensions=1024,   # uncomment for Matryoshka truncation (~3x cheaper storage)
    )


@lru_cache(maxsize=None)
def get_vector_store(recreate: bool = False) -> QdrantVectorStore:
    """Cached per process: this also loads the local sparse ONNX model and
    opens Qdrant client connections, wasted work across repeated calls in
    the same process (e.g. one per chatbot turn)."""
    client = qdrant_client.QdrantClient(url=config.QDRANT_URL)
    aclient = qdrant_client.AsyncQdrantClient(url=config.QDRANT_URL)

    if recreate and client.collection_exists(config.COLLECTION):
        print(f"  dropping existing collection '{config.COLLECTION}'")
        client.delete_collection(config.COLLECTION)

    return QdrantVectorStore(
        collection_name=config.COLLECTION,
        client=client,
        aclient=aclient,
        enable_hybrid=True,                            # <-- must be set at CREATE time
        fastembed_sparse_model=config.SPARSE_MODEL,    # local ONNX; first run downloads it
        batch_size=64,
    )


def main(recreate: bool = False) -> None:
    config.banner("STEP B -- embed & index (dense + sparse)")
    nodes = load_nodes()
    print(f"  {len(nodes)} nodes to index")

    vector_store = get_vector_store(recreate=recreate)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=get_embed_model(),
        show_progress=True,
    )

    client = qdrant_client.QdrantClient(url=config.QDRANT_URL)
    info = client.get_collection(config.COLLECTION)
    print(f"\n  collection '{config.COLLECTION}': {info.points_count} points")
    print(f"  dense vectors : {list((info.config.params.vectors or {}).keys())}")
    print(f"  sparse vectors: {list((info.config.params.sparse_vectors or {}).keys())}")
    print(f"\n  dashboard: {config.QDRANT_URL}/dashboard#/collections/{config.COLLECTION}")
    if not (info.config.params.sparse_vectors or {}):
        print("  ! NO SPARSE VECTORS -- hybrid search will silently be dense-only.")
        print("    Re-run with --recreate.")
    return index


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true",
                    help="drop and rebuild the collection (required if you "
                         "previously indexed without sparse vectors)")
    main(**vars(ap.parse_args()))
