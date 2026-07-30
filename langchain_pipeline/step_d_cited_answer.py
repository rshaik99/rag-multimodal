"""
STEP D -- Grounded synthesis with VERIFIED citations  (LangChain track)

    python -m langchain_pipeline.step_d_cited_answer "what caused ERR-4417?"

This track goes one step beyond the LlamaIndex version: citations are forced
out as STRUCTURED DATA (Pydantic `citations: list[int]`) rather than parsed out
of prose. That means every claim can be programmatically checked against the
retrieved set -- a citation pointing at a source index that does not exist is a
hallucination you have caught automatically, in code, without a human reading
the answer.
"""
from __future__ import annotations

import sys

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from common import colors, config
from langchain_pipeline.step_c_retrieve_rerank import build_retriever


class Claim(BaseModel):
    """One atomic factual statement plus the sources that support it."""
    statement: str = Field(description="A single factual claim answering part of the question.")
    citations: list[int] = Field(
        description="1-based source numbers that directly support this statement.")


class GroundedAnswer(BaseModel):
    answerable: bool = Field(
        description="False if the sources do not contain enough information.")
    answer: str = Field(
        description="The full answer in markdown with inline [n] citations. "
                    "Empty string if answerable is False.")
    claims: list[Claim] = Field(default_factory=list,
                                description="Claim-level attribution for auditing.")


SYSTEM = """You are a precise research assistant answering from a document corpus.

RULES — follow exactly:
1. Use ONLY the numbered sources provided. Never use outside knowledge.
2. Every factual claim must carry an inline [n] citation matching a source number.
3. If the sources are insufficient, set answerable=false and leave answer empty.
   Do not guess and do not pad with general knowledge.
4. When a source is a table, preserve its structure and quote exact figures.
5. Populate `claims` so each statement maps to the source numbers supporting it.
6. Be concise. No preamble."""

HUMAN = """{sources}

---
QUESTION: {query}"""


def format_sources(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        m = d.metadata
        header = (f"Source [{i}] | file: {m.get('file_name')} | page: {m.get('page')} "
                  f"| type: {m.get('content_type', 'text')}")
        if m.get("section"):
            header += f" | section: {m['section']}"
        body = d.page_content
        if m.get("content_type") == "table":
            body = f"```\n{body}\n```"
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


def answer(query: str) -> dict:
    config.banner(f"STEP D (LangChain) -- grounded answer\n  query: {query!r}")
    config.require_openai()

    docs = build_retriever().invoke(query)
    if not docs:
        print("\n  No sources survived the retrieval + score floor -> refusing.")
        return {"answerable": False, "answer": "", "sources": []}

    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=config.TEMPERATURE,
                     api_key=config.OPENAI_API_KEY)
    chain = (ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])
             | llm.with_structured_output(GroundedAnswer))

    result: GroundedAnswer = chain.invoke(
        {"sources": format_sources(docs), "query": query})

    # ---- programmatic citation verification -------------------------------
    n = len(docs)
    bad = sorted({c for claim in result.claims for c in claim.citations
                  if not 1 <= c <= n})
    uncited = [claim.statement for claim in result.claims if not claim.citations]

    if not result.answerable:
        print(colors.answer("\n  The provided sources do not contain enough information to "
                            "answer this.  (model refused — correct behaviour)"))
    else:
        print(colors.answer(f"\n{result.answer}\n"))

    print("-" * 72)
    print("SOURCES")
    for i, d in enumerate(docs, 1):
        m = d.metadata
        sc = m.get("relevance_score")
        sc = f"{sc:.4f}" if isinstance(sc, (int, float)) else "  -   "
        print(f"  [{i}] {sc}  {m.get('file_name')} p.{m.get('page')} "
              f"({m.get('content_type', 'text')})")

    if result.claims:
        print("\nCLAIM-LEVEL ATTRIBUTION")
        for c in result.claims:
            print(f"  {c.citations} <- {c.statement[:100]}")

    if bad:
        print(f"\n  ! HALLUCINATED CITATIONS {bad}: no such source. "
              "Reject this answer in production.")
    if uncited:
        print(f"\n  ! {len(uncited)} claim(s) with NO citation — treat as unverified.")
    if not bad and not uncited and result.answerable:
        print("\n  OK: every claim is attributed to a real retrieved source")

    return {"answerable": result.answerable, "answer": result.answer,
            "claims": [c.model_dump() for c in result.claims],
            "sources": [{"n": i, **d.metadata} for i, d in enumerate(docs, 1)],
            "invalid_citations": bad}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What caused incident ERR-4417 and how was it fixed?"
    answer(q)
