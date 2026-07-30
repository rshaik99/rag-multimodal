# rag-multimodal

A grounded, citation-verified RAG pipeline implemented **twice** — once in
LlamaIndex, once in LangChain — so the same design decisions (hybrid search,
reranking, hallucination guards, multimodal routing) can be compared
side by side instead of taken on faith from a single framework's docs.

Plus a Phase 2 extension: a LangGraph agent that retrieves across text,
tables, *and* extracted figures, and only pays for a vision-LLM call when
the evidence actually needs it.

## Why two tracks

Most RAG tutorials pick a framework and show you its happy path. This repo
implements the identical four-stage pipeline in both, so you can see where
the frameworks genuinely differ (retriever composition, structured-output
ergonomics, reranker wiring) versus where the RAG *design* choices are
what actually matter, independent of framework:

| Stage | LlamaIndex | LangChain |
|---|---|---|
| A — ingest | `llamaindex_pipeline/step_a_ingest.py` | `langchain_pipeline/step_a_ingest.py` |
| B — hybrid index (dense + sparse, Qdrant) | `step_b_index.py` | `step_b_index.py` |
| C — retrieve + rerank + score floor | `step_c_retrieve_rerank.py` | `step_c_retrieve_rerank.py` |
| D — cited, grounded answer | `step_d_cited_answer.py` (regex-verified `[n]` citations) | `step_d_cited_answer.py` (structured `citations: list[int]`, programmatically checked) |

Both tracks enforce the same non-negotiables:
- **Hybrid retrieval** (dense + BM25 sparse, server-side RRF fusion in Qdrant) — dense-only embeddings reliably miss exact identifiers like error codes.
- **Cross-encoder reranking with a hard score floor** — the single biggest lever against hallucination. Below `RERANK_SCORE_FLOOR`, the pipeline returns *nothing* rather than the least-bad chunk, and the LLM is instructed to refuse when sources are insufficient.
- **Citation verification, not citation trust** — every answer's `[n]` references are checked against the actual retrieved set. A citation pointing at a source that doesn't exist is a caught hallucination, in code.

## Phase 2 — multimodal agentic routing

`langchain_pipeline/graph_router.py` is a LangGraph agent that retrieves
across all modalities at once, then routes on **evidence, not intent**:

```
retrieve (hybrid, all modalities)
    -> route: did an image node survive rerank + the score floor?
         strong signal (visual keyword AND image evidence agree) -> vision LLM directly
         weak/no signal -> cheap text LLM first
                              -> grade: is the answer grounded?
                                   grounded -> done
                                   ungrounded + image evidence exists -> escalate to vision LLM
```

The expensive vision hop is paid only when the cheap path can't ground the
answer. `phase2_multimodal/ingest_images.py` extracts figures from PDFs and
indexes a **composite node** (VLM description + transcribed data points +
surrounding page text) so charts become findable by the sparse/keyword leg,
not just by embedding similarity.

## Chatbot

`chatbot.py` is a multi-turn CLI over all three pipelines (`langchain`,
`llamaindex`, `graph`), with follow-up questions condensed against
conversation history before retrieval:

```
python -m chatbot                      # LangChain track
python -m chatbot --pipeline llamaindex
python -m chatbot --pipeline graph     # multimodal agentic router
```

At the `You:` prompt: `/prompts` lists sample questions grounded in whatever's
actually indexed (type a number to run one), `/pipeline <name>` switches
tracks, `/reset` clears history, `exit` quits.

## Quickstart

```bash
git clone <this-repo>
cd rag-multimodal
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

cp .env.example .env      # set OPENAI_API_KEY at minimum
docker compose up -d      # Qdrant on localhost:6333

pip install reportlab && python make_sample_docs.py   # generates a sample PDF in ./data

python -m langchain_pipeline.run_all --recreate        # ingest + index + 3 demo queries
python -m llamaindex_pipeline.run_all --recreate

python -m chatbot
```

## Repo structure

```
common/                shared config (models, chunking, score floor) + colors.py
llamaindex_pipeline/   Step A-D, LlamaIndex track
langchain_pipeline/    Step A-D, LangChain track, + graph_router.py (Phase 2)
phase2_multimodal/     figure extraction/indexing + a framework-free routing reference
chatbot.py             multi-turn CLI over all three pipelines
make_sample_docs.py    generates a zero-setup sample PDF (table + exact-ID stress test)
docker-compose.yml     local Qdrant
```

## Notable engineering findings

Things that weren't obvious going in, kept here because they're the kind of
detail a tutorial usually skips:

- **The reranker score floor is the actual hallucination guard**, not the
  system prompt. An LLM told "refuse if insufficient" will still try to be
  helpful with borderline-relevant context; dropping that context before it
  ever reaches the prompt is what makes refusal reliable.
- **Retrieval-driven routing can silently make its own escalation path
  unreachable.** The original Phase 2 router sent every query with *any*
  retrieved image straight to the vision LLM — which meant the
  ungrounded-text -> escalate-to-vision edge could never fire, since it
  re-checked the same condition `route()` had already used to skip text
  entirely. Fixed by only taking the vision shortcut on a strong signal
  (keyword *and* image both present) and letting weaker cases try text
  first, with grounding failure as the real trigger.
- **Two local cross-encoder stacks (LangChain's and LlamaIndex's) in one
  process reliably segfault on Windows** — a native torch/sentence-
  transformers ABI conflict, not fixed by `KMP_DUPLICATE_LIB_OK`. The
  chatbot avoids it by never importing both tracks into the same process.
- **`os.execv` is not a true exec on Windows.** There's no such syscall, so
  CPython emulates it by spawning a child and *blocking* the parent until
  the child exits — for an interactive loop, the "old" process never
  actually goes away. `subprocess.Popen` (detached) + `sys.exit()` is the
  correct way to hand off to a fresh process on Windows.
