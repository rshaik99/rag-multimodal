"""
STEP C -- Hybrid retrieval + reranking  (LangChain track)

    python -m langchain_pipeline.step_c_retrieve_rerank "what caused ERR-4417?"

Uses ContextualCompressionRetriever as the rerank wrapper: base retriever pulls
FUSED_TOP_N candidates, the compressor (cross-encoder) squeezes them to
FINAL_TOP_K. Same shape as the LlamaIndex node-postprocessor pattern.
"""
from __future__ import annotations

import sys
import time
from functools import lru_cache

from langchain_core.cross_encoders import BaseCrossEncoder
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_qdrant import RetrievalMode
from pydantic import ConfigDict

from common import config
from langchain_pipeline.step_b_index import get_store


class _ScoringCrossEncoderReranker(BaseDocumentCompressor):
    """
    Same job as langchain's stock CrossEncoderReranker, except it stamps the
    score onto `doc.metadata["relevance_score"]` before slicing to top_n.

    GOTCHA: the stock CrossEncoderReranker.compress_documents() sorts by score
    then discards it -- callers can see the new order but never the number,
    so nothing downstream (this file's _show(), step_d's SOURCES listing) can
    display or threshold on it. Re-implementing the ~5-line method here is
    cheaper than monkeypatching a third-party class.
    """

    model: BaseCrossEncoder
    top_n: int = 3
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    def compress_documents(self, documents, query, callbacks=None):
        documents = list(documents)
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        for doc, score in zip(documents, scores):
            doc.metadata["relevance_score"] = float(score)
        ranked = sorted(documents, key=lambda d: d.metadata["relevance_score"], reverse=True)
        # Score floor => out-of-scope questions return NOTHING rather than the
        # least-bad chunk. Mirrors the LlamaIndex track's SimilarityPostprocessor.
        ranked = [d for d in ranked if d.metadata["relevance_score"] >= config.RERANK_SCORE_FLOOR]
        return ranked[: self.top_n]


@lru_cache(maxsize=None)
def get_compressor(top_n: int = config.FINAL_TOP_K):
    """Cohere API -> local BGE cross-encoder -> None, in that order.

    Cached per process: the local cross-encoder is ~500M params loaded from
    disk (plus an HF Hub check) on every call otherwise -- expensive to pay
    once per chatbot turn instead of once per session.
    """
    if config.COHERE_API_KEY:
        try:
            from langchain_cohere import CohereRerank
            print(f"  reranker: Cohere {config.COHERE_RERANK_MODEL}")
            return CohereRerank(cohere_api_key=config.COHERE_API_KEY,
                                model=config.COHERE_RERANK_MODEL, top_n=top_n)
        except ImportError:
            print("  ! langchain-cohere not installed")
    try:
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
        print(f"  reranker: local {config.LOCAL_RERANK_MODEL}")
        return _ScoringCrossEncoderReranker(
            model=HuggingFaceCrossEncoder(model_name=config.LOCAL_RERANK_MODEL),
            top_n=top_n,
        )
    except Exception:                                          # noqa: BLE001
        print("  ! no reranker available; using fusion order")
        return None


def build_retriever(top_k: int = config.FINAL_TOP_K,
                    candidates: int = config.FUSED_TOP_N,
                    mode: RetrievalMode = RetrievalMode.HYBRID,
                    metadata_filter: dict | None = None):
    """
    Returns a retriever that does: hybrid search -> RRF (server-side) -> rerank.

    metadata_filter example (the cheapest precision win in the stack):
        {"must": [{"key": "metadata.doc_type", "match": {"value": "pdf"}}]}
    """
    store = get_store()
    search_kwargs: dict = {"k": candidates}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    base = store.as_retriever(search_kwargs=search_kwargs)
    base.vectorstore.retrieval_mode = mode

    compressor = get_compressor(top_n=top_k)
    if compressor is None:
        base.search_kwargs["k"] = top_k
        return base

    try:
        from langchain_classic.retrievers import ContextualCompressionRetriever
    except ImportError:
        from langchain.retrievers import ContextualCompressionRetriever
    return ContextualCompressionRetriever(base_compressor=compressor,
                                          base_retriever=base)


def _show(label: str, docs: list[Document], elapsed: float) -> None:
    print(f"\n--- {label}  ({elapsed * 1000:.0f} ms, {len(docs)} hits) ---")
    for i, d in enumerate(docs, 1):
        m = d.metadata
        score = m.get("relevance_score", m.get("_score"))
        s = f"{score:.4f}" if isinstance(score, (int, float)) else "  -   "
        print(f"  [{i}] {s}  {m.get('file_name')} p.{m.get('page')} "
              f"({m.get('content_type', 'text')})\n"
              f"      {d.page_content[:150]}...")


def _search(store, query: str, mode: RetrievalMode, k: int) -> list[Document]:
    """
    GOTCHA: `retrieval_mode` is read from the STORE attribute, not accepted as a
    kwarg by similarity_search(). Passing it as a kwarg leaks into the Qdrant
    query_points() call and raises. Toggle the attribute instead.
    """
    previous = store.retrieval_mode
    try:
        store.retrieval_mode = mode
        return store.similarity_search(query, k=k)
    finally:
        store.retrieval_mode = previous


def compare(query: str) -> None:
    config.banner(f"STEP C (LangChain) -- retrieval comparison\n  query: {query!r}")
    store = get_store()

    t = time.perf_counter()
    dense = _search(store, query, RetrievalMode.DENSE, config.FINAL_TOP_K)
    _show("1. DENSE ONLY", dense, time.perf_counter() - t)

    t = time.perf_counter()
    hyb = _search(store, query, RetrievalMode.HYBRID, config.FINAL_TOP_K)
    _show("2. HYBRID (dense + sparse, RRF)", hyb, time.perf_counter() - t)

    t = time.perf_counter()
    rer = build_retriever().invoke(query)
    _show("3. HYBRID + CROSS-ENCODER RERANK", rer, time.perf_counter() - t)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What caused incident ERR-4417 and how was it fixed?"
    compare(q)
