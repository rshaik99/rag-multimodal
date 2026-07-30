"""
PHASE 2 -- standalone query router (framework-free reference implementation).

    python -m phase2_multimodal.router "what does the revenue chart show?"

Three routing strategies, cheapest first. In production use TIER 2.

  TIER 1  Heuristic        ~0 ms, ~0 cost, ~70% accurate. Good pre-filter.
  TIER 2  Retrieval-driven ~0 extra cost, ~90%+ accurate. RECOMMENDED.
          Retrieve first, then look at what actually came back. Evidence beats
          intent-guessing, and it costs nothing because you retrieve anyway.
  TIER 3  LLM classifier   ~50-150 ms, small cost. Use only when you must route
          BEFORE retrieval (e.g. picking between separate indexes / tenants).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

from common import config

VISUAL_KEYWORDS = {
    "chart", "charts", "figure", "figures", "diagram", "diagrams", "graph",
    "graphs", "image", "images", "picture", "screenshot", "plot", "visual",
    "shown in", "illustrated", "depicted", "flowchart", "architecture diagram",
}
TABULAR_KEYWORDS = {"table", "row", "column", "cell", "breakdown", "by segment",
                    "by quarter", "per year", "totals"}


@dataclass
class Route:
    modality: str          # "text" | "table" | "visual"
    model: str             # which LLM to synthesise with
    filters: dict | None   # metadata filter to apply at retrieval time
    why: str


# ------------------------------------------------------------------ tier 1
def heuristic_route(query: str) -> Route:
    q = query.lower()
    if any(k in q for k in VISUAL_KEYWORDS):
        return Route("visual", config.VISION_MODEL, None, "visual keyword in query")
    if any(k in q for k in TABULAR_KEYWORDS):
        return Route("table", config.LLM_MODEL,
                     {"content_type": ["table", "text"]}, "tabular keyword in query")
    return Route("text", config.LLM_MODEL, None, "no modality signal")


# ------------------------------------------------------------------ tier 2
def retrieval_driven_route(query: str, retrieved_metadata: list[dict]) -> Route:
    """
    THE ONE TO USE. Decide after retrieval, based on what survived reranking.

    retrieved_metadata: [{"content_type": "...", "image_path": "..."}, ...]
    """
    types = {m.get("content_type", "text") for m in retrieved_metadata}
    if "image" in types:
        return Route("visual", config.VISION_MODEL, None,
                     "an image node survived retrieval + reranking")
    if "table" in types:
        return Route("table", config.LLM_MODEL, None,
                     "a table node survived retrieval + reranking")
    return Route("text", config.LLM_MODEL, None, "text-only evidence")


# ------------------------------------------------------------------ tier 3
def llm_route(query: str) -> Route:
    """Pre-retrieval classifier. Only worth it when you must pick an index first."""
    from openai import OpenAI
    from pydantic import BaseModel

    class Decision(BaseModel):
        needs_visual: bool
        needs_tabular: bool
        reasoning: str

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.beta.chat.completions.parse(
        model=config.ROUTER_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content":
                "Classify whether answering this question about a document corpus "
                "requires looking at figures/charts/diagrams (needs_visual) and/or "
                "structured tables (needs_tabular). Be conservative: only set "
                "needs_visual when the answer plausibly lives in a picture."},
            {"role": "user", "content": query},
        ],
        response_format=Decision,
    )
    d = resp.choices[0].message.parsed
    if d.needs_visual:
        return Route("visual", config.VISION_MODEL, None, d.reasoning)
    if d.needs_tabular:
        return Route("table", config.LLM_MODEL,
                     {"content_type": ["table", "text"]}, d.reasoning)
    return Route("text", config.LLM_MODEL, None, d.reasoning)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What does the revenue chart show for Q3?"
    config.banner(f"routing: {q!r}")
    r1 = heuristic_route(q)
    print(f"  TIER 1 heuristic       -> {r1.modality:6s} via {r1.model}  ({r1.why})")
    fake = [{"content_type": "table"}, {"content_type": "text"}]
    r2 = retrieval_driven_route(q, fake)
    print(f"  TIER 2 retrieval-driven-> {r2.modality:6s} via {r2.model}  ({r2.why})")
    print("          (using a stub retrieval result; wire to step_c in real use)")
    if config.OPENAI_API_KEY:
        r3 = llm_route(q)
        print(f"  TIER 3 llm classifier  -> {r3.modality:6s} via {r3.model}  ({r3.why})")
