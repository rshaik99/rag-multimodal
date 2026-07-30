"""
STEP C -- Hybrid retrieval + cross-encoder reranking  (LlamaIndex track)

    python -m llamaindex_pipeline.step_c_retrieve_rerank "what caused ERR-4417?"

Prints a 3-way comparison so you can SEE the effect of each layer:
    (1) dense only        (2) hybrid dense+sparse       (3) hybrid + rerank

Try an exact-identifier query (e.g. "ERR-4417"). Dense-only will usually miss
it; that single observation is the entire argument for hybrid search.
"""
from __future__ import annotations

import sys
import time
from functools import lru_cache

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore

from common import config
from llamaindex_pipeline.step_b_index import get_embed_model, get_vector_store


# ------------------------------------------------------------------ rerankers
@lru_cache(maxsize=None)
def get_reranker(top_n: int = config.FINAL_TOP_K):
    """
    Cohere API if a key is present, else local BGE cross-encoder, else None.
    Reranking is the single highest-ROI component in this pipeline -- retrieving
    40 and reranking to 5 typically moves nDCG@5 by 10-25 points.

    Cached per process: the local cross-encoder is ~500M params loaded from
    disk (plus an HF Hub check) on every call otherwise -- expensive to pay
    once per chatbot turn instead of once per session.
    """
    if config.COHERE_API_KEY:
        try:
            from llama_index.postprocessor.cohere_rerank import CohereRerank
            print(f"  reranker: Cohere {config.COHERE_RERANK_MODEL}")
            return CohereRerank(api_key=config.COHERE_API_KEY,
                                model=config.COHERE_RERANK_MODEL, top_n=top_n)
        except ImportError:
            print("  ! llama-index-postprocessor-cohere-rerank not installed")
    try:
        from llama_index.core.postprocessor import SentenceTransformerRerank
        print(f"  reranker: local {config.LOCAL_RERANK_MODEL}")
        return SentenceTransformerRerank(model=config.LOCAL_RERANK_MODEL, top_n=top_n)
    except Exception:                                          # noqa: BLE001
        print("  ! no reranker available (pip install sentence-transformers, "
              "or set COHERE_API_KEY). Falling back to fusion order.")
        return None


@lru_cache(maxsize=None)
def get_index() -> VectorStoreIndex:
    return VectorStoreIndex.from_vector_store(
        vector_store=get_vector_store(),
        embed_model=get_embed_model(),
    )


# ------------------------------------------------------------------ retrievers
def dense_retriever(index: VectorStoreIndex, k: int = config.FINAL_TOP_K):
    from llama_index.core.vector_stores.types import VectorStoreQueryMode
    return index.as_retriever(similarity_top_k=k,
                              vector_store_query_mode=VectorStoreQueryMode.DEFAULT)


def hybrid_retriever(index: VectorStoreIndex,
                     k: int = config.RETRIEVE_TOP_K,
                     sparse_k: int = config.FUSED_TOP_N):
    """
    Qdrant fuses the dense and sparse legs SERVER-SIDE with RRF
    (score = sum 1/(60 + rank)), so no score normalisation is needed here.
    `sparse_top_k` controls the size of the candidate pool handed to the reranker.
    """
    return index.as_retriever(
        vector_store_query_mode="hybrid",
        similarity_top_k=k,
        sparse_top_k=sparse_k,
    )


def retrieve_and_rerank(index: VectorStoreIndex, query: str,
                        top_k: int = config.FINAL_TOP_K) -> list[NodeWithScore]:
    """The production path: hybrid -> RRF -> cross-encoder -> score floor."""
    nodes = hybrid_retriever(index).retrieve(query)

    reranker = get_reranker(top_n=top_k)
    if reranker is not None:
        nodes = reranker.postprocess_nodes(nodes, query_str=query)
        # Score floor => out-of-scope questions return NOTHING rather than the
        # least-bad chunk. A large part of hallucination prevention.
        nodes = SimilarityPostprocessor(
            similarity_cutoff=config.RERANK_SCORE_FLOOR
        ).postprocess_nodes(nodes)
    return nodes[:top_k]


# ------------------------------------------------------------------ display
def _show(label: str, nodes: list[NodeWithScore], elapsed: float) -> None:
    print(f"\n--- {label}  ({elapsed * 1000:.0f} ms, {len(nodes)} hits) ---")
    if not nodes:
        print("  (nothing above the score floor -- correct behaviour for an "
              "out-of-scope question)")
    for i, n in enumerate(nodes, 1):
        m = n.node.metadata
        tag = m.get("content_type", "text")
        snippet = n.node.get_content()[:150].replace("\n", " ")
        print(f"  [{i}] {n.score:.4f}  {m.get('file_name')} p.{m.get('page')} "
              f"({tag})\n      {snippet}...")


def compare(query: str) -> None:
    config.banner(f"STEP C -- retrieval comparison\n  query: {query!r}")
    index = get_index()

    t = time.perf_counter()
    d = dense_retriever(index).retrieve(query)
    _show("1. DENSE ONLY", d, time.perf_counter() - t)

    t = time.perf_counter()
    h = hybrid_retriever(index, k=config.FINAL_TOP_K, sparse_k=config.FINAL_TOP_K).retrieve(query)
    _show("2. HYBRID (dense + sparse, RRF)", h, time.perf_counter() - t)

    t = time.perf_counter()
    r = retrieve_and_rerank(index, query)
    _show("3. HYBRID + CROSS-ENCODER RERANK", r, time.perf_counter() - t)

    print("\n  Read the top-1 of each. If (3) is not obviously better than (1), "
          "your reranker is not wired up.")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What caused incident ERR-4417 and how was it fixed?"
    compare(q)
