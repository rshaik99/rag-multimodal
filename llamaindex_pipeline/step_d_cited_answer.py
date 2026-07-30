"""
STEP D -- Grounded synthesis with source attribution  (LlamaIndex track)

    python -m llamaindex_pipeline.step_d_cited_answer "what caused ERR-4417?"

Three layers of hallucination defence:
  1. Numbered, tagged context   -- the model can only cite what it can identify.
  2. Constrained prompt         -- answer ONLY from sources; admit ignorance.
  3. Return the receipts        -- source nodes come back with the answer so a
                                   human (or a test) can verify every claim.
"""
from __future__ import annotations

import re
import sys

from llama_index.core.schema import NodeWithScore
from llama_index.llms.openai import OpenAI

from common import colors, config
from llamaindex_pipeline.step_c_retrieve_rerank import get_index, retrieve_and_rerank

SYSTEM_PROMPT = """You are a precise research assistant answering from a document corpus.

RULES — follow exactly:
1. Answer ONLY using the numbered sources below. Never use outside knowledge.
2. Cite the source number in square brackets after every factual claim, e.g. [1] or [2][3].
3. If the sources do not contain the answer, reply exactly:
   "The provided sources do not contain enough information to answer this."
   Do not guess, and do not pad the answer with general knowledge.
4. When a source is a table, preserve its structure in your answer (markdown table
   or a clean list) and quote the exact figures.
5. Be concise. No preamble, no restating the question."""

USER_TEMPLATE = """{sources}

---
QUESTION: {query}

Answer using only the sources above, with [n] citations."""


def format_sources(nodes: list[NodeWithScore]) -> str:
    """Numbered + tagged. Tables get fenced so the pipes survive the prompt."""
    parts = []
    for i, n in enumerate(nodes, 1):
        m = n.node.metadata
        header = (f"Source [{i}] | file: {m.get('file_name')} "
                  f"| page: {m.get('page')} | type: {m.get('content_type', 'text')}")
        if m.get("section"):
            header += f" | section: {m['section']}"
        body = n.node.get_content()
        if m.get("content_type") == "table":
            body = f"```\n{body}\n```"
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


def verify_citations(answer: str, n_sources: int) -> tuple[list[int], list[int]]:
    """Return (valid, invalid) cited indices. An invalid one is a caught hallucination."""
    cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})
    return ([c for c in cited if 1 <= c <= n_sources],
            [c for c in cited if not 1 <= c <= n_sources])


def answer(query: str) -> dict:
    config.banner(f"STEP D -- grounded answer\n  query: {query!r}")
    config.require_openai()

    nodes = retrieve_and_rerank(get_index(), query)
    if not nodes:
        text = "The provided sources do not contain enough information to answer this."
        print(colors.answer(f"\n{text}\n"))
        print("-" * 72)
        print("SOURCES\n  (none scored above RERANK_SCORE_FLOOR -- correct refusal)")
        return {"answer": text, "sources": [], "invalid_citations": []}

    llm = OpenAI(model=config.LLM_MODEL, temperature=config.TEMPERATURE,
                 api_key=config.OPENAI_API_KEY, system_prompt=SYSTEM_PROMPT)
    resp = llm.complete(USER_TEMPLATE.format(sources=format_sources(nodes), query=query))
    text = str(resp).strip()

    valid, invalid = verify_citations(text, len(nodes))

    print(colors.answer(f"\n{text}\n"))
    print("-" * 72)
    print("SOURCES")
    for i, n in enumerate(nodes, 1):
        m = n.node.metadata
        used = "*" if i in valid else " "
        print(f" {used}[{i}] score={n.score:.4f}  {m.get('file_name')} "
              f"p.{m.get('page')} ({m.get('content_type', 'text')}) "
              f"{m.get('section', '')}")
    if invalid:
        print(f"\n  ! HALLUCINATED CITATIONS {invalid} -- these source numbers "
              "do not exist. Tighten the prompt or reduce top_k.")

    return {"answer": text,
            "sources": [{"n": i, "score": n.score, **n.node.metadata}
                        for i, n in enumerate(nodes, 1)],
            "invalid_citations": invalid}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What caused incident ERR-4417 and how was it fixed?"
    answer(q)
