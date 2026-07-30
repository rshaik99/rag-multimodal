"""
PHASE 2 SKELETON -- LangGraph agentic RAG with retrieval-driven routing.

    python -m langchain_pipeline.graph_router "show me the revenue chart trend"

Graph:

    START
      |
      v
  [retrieve]  hybrid + rerank over ALL modalities
      |
      v
  [route]     retrieval-driven: does the surviving top-K contain an image node?
      |                                     (this beats intent classification --
      |                                      it decides on evidence, not a guess)
      +--- text ------> [answer_text]   gpt-4.1, cheap
      |                       |
      |                       v
      |                  [grade]  is the answer grounded in the sources?
      |                       |
      |            grounded --+-- ungrounded & image candidates exist
      |               |                        |
      |               |                        v
      +--- visual ----+-------------> [answer_vision]  claude-sonnet-5 / gpt-4.1
                      |                        |
                      v                        v
                     END <---------------------+

Why this shape: the expensive vision hop is paid only on queries that need it,
and the escalation is triggered by a groundedness failure rather than by
guessing intent up front.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from common import colors, config
from langchain_pipeline.step_c_retrieve_rerank import build_retriever
from langchain_pipeline.step_d_cited_answer import format_sources

VISUAL_KEYWORDS = {"chart", "figure", "diagram", "graph", "image", "picture",
                   "screenshot", "plot", "visual", "shown in", "illustrated"}


# ------------------------------------------------------------------ state
class RAGState(TypedDict, total=False):
    query: str
    docs: list[Document]
    route: Literal["text", "visual"]
    answer: str
    grounded: bool
    escalated: bool


class Grade(BaseModel):
    grounded: bool = Field(description="True if every claim in the answer is "
                                       "supported by the provided sources.")
    reason: str = Field(description="One sentence explaining the verdict.")


def _llm(model: str | None = None, **kw) -> ChatOpenAI:
    return ChatOpenAI(model=model or config.LLM_MODEL,
                      temperature=config.TEMPERATURE,
                      api_key=config.OPENAI_API_KEY, **kw)


# ------------------------------------------------------------------ nodes
def retrieve(state: RAGState) -> RAGState:
    docs = build_retriever().invoke(state["query"])
    print(f"  [retrieve] {len(docs)} docs "
          f"({sum(1 for d in docs if d.metadata.get('content_type') == 'image')} image)")
    return {"docs": docs}


def route(state: RAGState) -> RAGState:
    """
    RETRIEVAL-DRIVEN ROUTING.

    Only take the direct-to-vision shortcut on a STRONG signal: explicit
    visual language in the query AND an image node surviving retrieval both
    agreeing. Anything weaker (an image happens to be in the top-K but the
    query never asked for a chart) goes through text first -- the cheap
    path -- with `grade` + `after_grade` as the real escalation trigger.

    This matters structurally, not just for cost: if routing sent every
    image-bearing retrieval straight to vision, `answer_text`/`grade` would
    never run when an image is present, which makes the ungrounded ->
    escalate-to-vision edge in `after_grade` unreachable (its own
    `has_images` check computes the same condition `route` already used to
    avoid the text path). Escalation being reachable is the whole point of
    this graph shape -- see the module docstring.
    """
    q = state["query"].lower()
    keyword_hit = any(k in q for k in VISUAL_KEYWORDS)
    image_hit = any(d.metadata.get("content_type") == "image" for d in state["docs"])
    route_to = "visual" if (keyword_hit and image_hit) else "text"
    print(f"  [route] keyword={keyword_hit} image_in_topk={image_hit} -> {route_to}")
    return {"route": route_to}


def answer_text(state: RAGState) -> RAGState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer ONLY from the numbered sources. Cite [n] after every "
                   "claim. If insufficient, say so explicitly."),
        ("human", "{sources}\n\n---\nQUESTION: {query}"),
    ])
    out = (prompt | _llm()).invoke(
        {"sources": format_sources(state["docs"]), "query": state["query"]})
    print("  [answer_text] done")
    return {"answer": out.content}


def grade(state: RAGState) -> RAGState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict groundedness grader. Given SOURCES and an "
                   "ANSWER, decide whether every claim in the ANSWER is supported "
                   "by the SOURCES."),
        ("human", "SOURCES:\n{sources}\n\nANSWER:\n{answer}"),
    ])
    g: Grade = (prompt | _llm(config.ROUTER_MODEL).with_structured_output(Grade)).invoke(
        {"sources": format_sources(state["docs"]), "answer": state["answer"]})
    print(f"  [grade] grounded={g.grounded} :: {g.reason}")
    return {"grounded": g.grounded}


def answer_vision(state: RAGState) -> RAGState:
    """
    Vision synthesis: text chunks + the ORIGINAL image bytes.

    Never answer a chart question from the text summary alone -- the summary was
    written for RETRIEVAL. For SYNTHESIS, give the model the real pixels.
    """
    import base64
    from pathlib import Path

    content: list[dict] = [{
        "type": "text",
        "text": ("Answer ONLY from the sources and images below. Cite [n] after "
                 f"every claim.\n\n{format_sources(state['docs'])}\n\n---\n"
                 f"QUESTION: {state['query']}"),
    }]

    for d in state["docs"]:
        p = d.metadata.get("image_path")
        if p and Path(p).exists():
            b64 = base64.b64encode(Path(p).read_bytes()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}",
                                          "detail": "high"}})

    out = _llm(config.VISION_MODEL).invoke([HumanMessage(content=content)])
    print(f"  [answer_vision] attached {len(content) - 1} image(s)")
    return {"answer": out.content, "escalated": True}


# ------------------------------------------------------------------ edges
def after_route(state: RAGState) -> str:
    return "answer_vision" if state["route"] == "visual" else "answer_text"


def after_grade(state: RAGState) -> str:
    """Escalate to vision only if ungrounded AND image evidence exists."""
    if state.get("grounded"):
        return END
    has_images = any(d.metadata.get("content_type") == "image" for d in state["docs"])
    if has_images and not state.get("escalated"):
        print("  [escalate] ungrounded + image candidates -> retry with vision LLM")
        return "answer_vision"
    return END


@lru_cache(maxsize=None)
def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("route", route)
    g.add_node("answer_text", answer_text)
    g.add_node("grade", grade)
    g.add_node("answer_vision", answer_vision)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "route")
    g.add_conditional_edges("route", after_route,
                            {"answer_text": "answer_text",
                             "answer_vision": "answer_vision"})
    g.add_edge("answer_text", "grade")
    g.add_conditional_edges("grade", after_grade,
                            {"answer_vision": "answer_vision", END: END})
    g.add_edge("answer_vision", END)
    return g.compile()


def answer(query: str) -> dict:
    """Uniform (query) -> {"answer": ..., "route": ..., ...} entry point,
    matching step_d_cited_answer.answer() so callers (e.g. chatbot.py) can
    treat this graph as just another pipeline. Prints the final answer as a
    side effect, same as the step_d modules, so it's visible regardless of
    whether this is run standalone or driven by chatbot.py."""
    result = build_graph().invoke({"query": query})
    print("\n" + "=" * 72)
    print(colors.answer(result.get("answer", "")))
    sys.stdout.flush()
    return result


if __name__ == "__main__":
    config.require_openai()
    q = " ".join(sys.argv[1:]) or "How did Cloud Services revenue trend in FY2025?"
    config.banner(f"LangGraph agentic RAG\n  query: {q!r}")
    answer(q)
