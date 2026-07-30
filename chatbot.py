"""
Interactive multi-turn chatbot over the RAG pipelines.

    python -m chatbot
    python -m chatbot --pipeline llamaindex
    python -m chatbot --pipeline graph

Each turn is condensed against prior turns into a standalone query (so
"what about part-time employees?" resolves against the previous topic)
before being handed to the selected pipeline's grounded-answer step. That
step still does its own retrieval + rerank + score-floor + citation
verification and prints the full SOURCES breakdown -- this file only adds
the conversational loop on top.

Three pipelines:
    langchain   text/table grounded answer, LangChain track
    llamaindex  text/table grounded answer, LlamaIndex track
    graph       LangGraph agentic router: retrieval-driven text/visual routing
                with vision-LLM escalation on ungrounded answers (phase2_multimodal)

Commands (typed at the prompt):
    /pipeline langchain|llamaindex|graph   switch pipelines (clears history)
    /prompts                               list sample questions for the current pipeline
    <number>                               run a prompt from the last /prompts list
    /reset                                 clear conversation history
    /exit  /quit                           leave

NOTE on /pipeline: "langchain" and "graph" share the same native
torch/sentence-transformers cross-encoder (graph_router.py calls straight
into langchain_pipeline's retriever), so switching between those two reuses
the already-warmed cache in-process -- no restart needed. Switching to/from
"llamaindex" loads a SEPARATE native cross-encoder stack, and loading both
stacks into one live process reliably segfaults (confirmed both directions on
Windows, not fixed by KMP_DUPLICATE_LIB_OK). So that switch hands off to a
brand-new detached process (subprocess.Popen) and exits this one -- the
terminal briefly clears/restarts, that's a fresh `python -m chatbot`, not a
bug. (Earlier this used os.execv, which on Windows is emulated by spawning
the child and BLOCKING until it exits -- for an interactive loop that meant
the old process never actually went away, silently doubling memory per
switch until enough stale processes piled up to make new model loads fail
with ERROR_ACCESS_DENIED. Popen + sys.exit() actually terminates this
process immediately instead.)
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

from openai import OpenAI

from common import colors, config

PIPELINES = {
    "langchain": "langchain_pipeline.step_d_cited_answer",
    "llamaindex": "llamaindex_pipeline.step_d_cited_answer",
    "graph": "langchain_pipeline.graph_router",
}

# Pipelines that share a native cross-encoder stack can switch in-process;
# crossing families needs a re-exec (see module docstring).
FAMILY = {"langchain": "lc", "graph": "lc", "llamaindex": "li"}

CONDENSE_SYSTEM = """Rewrite the LATEST user question as a standalone question that
makes sense without the chat history. Preserve the original intent and any
entities/topics implied by the history. If the latest question is already
standalone, return it unchanged. Reply with ONLY the rewritten question --
no preamble, no quotes."""

REFUSAL = "The provided sources do not contain enough information to answer this."

# Grounded in what's actually indexed right now: a 2-page mock Acme FY2025
# report (revenue table, operating margin, an incident log, a headcount
# table). Covers in-scope single-fact lookups, a table read, a cross-table
# comparison, a multi-turn follow-up chain, a deliberate text-vs-table
# inconsistency (text says "$1,048M" total revenue, the table sums to
# $2,613M -- good for checking which source the model trusts / whether it
# flags the conflict), and out-of-scope refusals.
TEXT_PROMPTS = [
    "What was Acme's total revenue in FY2025?",
    "Which segment had the highest Q4 FY2025 revenue?",
    "What was the consolidated operating margin in FY2025, and how did it compare between Cloud Services and Hardware?",
    "What caused incident ERR-4417 and how was it fixed?",
    "What was the mean time to recovery for ERR-4417?",
    "Which issue affected billing reconciliation for 1,204 accounts?",
    "How did Engineering headcount change from FY2024 to FY2025?",
    "Which function had a headcount decrease, and by how much?",
    "The report gives a $1,048M total revenue figure in the text but the segment table sums to a different number -- which is correct?",
    "What is the company's policy on remote work?",
    "Who is Acme's CEO?",
]

# For the "graph" pipeline: exercises retrieval-driven text/visual routing
# against the synthetic Cloud Services revenue chart indexed into rag_lab_lc
# (see phase2_multimodal/ingest_images.py). Covers: should-route-visual,
# should-route-text (regression check), a text/chart reconciliation prompt
# that's a good candidate for the ungrounded -> vision escalation path, and
# a refusal check for a chart that was never indexed (no pie/headcount chart
# exists -- only the one revenue bar chart).
MULTIMODAL_PROMPTS = [
    "What does the cloud services revenue chart show?",
    "Which quarter had the biggest jump in cloud revenue, according to the chart?",
    "What was Q4 cloud services revenue according to the chart?",
    "Describe the trend in the FY2025 revenue chart.",
    "What caused incident ERR-4417 and how was it fixed?",
    "What was the consolidated operating margin in FY2025?",
    "How did Engineering headcount change from FY2024 to FY2025?",
    "Does the Q3 cloud revenue in the chart match the \"$1,048M total revenue\" figure mentioned in the text?",
    "Is the growth shown in the chart consistent with the 21% year-over-year growth mentioned in the report text?",
    "What does the headcount breakdown pie chart show?",
    "What trend does the operating margin chart show over the four quarters?",
]

PROMPT_SETS = {"langchain": TEXT_PROMPTS, "llamaindex": TEXT_PROMPTS, "graph": MULTIMODAL_PROMPTS}


def _load_answer_fn(pipeline: str):
    module = importlib.import_module(PIPELINES[pipeline])
    return module.answer


def _answer_text(result: dict) -> str:
    return result.get("answer") or REFUSAL


def condense(client: OpenAI, history: list[tuple[str, str]], question: str) -> str:
    if not history:
        return question
    transcript = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history)
    resp = client.chat.completions.create(
        model=config.ROUTER_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": CONDENSE_SYSTEM},
            {"role": "user", "content": f"CHAT HISTORY:\n{transcript}\n\nLATEST QUESTION: {question}"},
        ],
    )
    return resp.choices[0].message.content.strip()


def chat(pipeline: str) -> None:
    config.require_openai()
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    answer_fn = _load_answer_fn(pipeline)

    history: list[tuple[str, str]] = []
    print(colors.status(f"RAG chatbot -- pipeline: {pipeline}"))
    print(colors.status(f"Commands: /pipeline {'|'.join(PIPELINES)}, /prompts, reset, exit\n"))
    print(colors.status(f"  (family: {FAMILY[pipeline]}; switching within the same family reuses "
                        f"the warmed cache, crossing families restarts the process)\n"))

    while True:
        try:
            question = input(colors.question("You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("/exit", "/quit", "exit", "quit"):
            break
        if question.lower() in ("/reset", "reset"):
            history.clear()
            print(colors.status("  (history cleared)"))
            continue
        if question.startswith("/pipeline"):
            parts = question.split()
            if len(parts) == 2 and parts[1] in PIPELINES and parts[1] != pipeline:
                if FAMILY[parts[1]] != FAMILY[pipeline]:
                    # Different native cross-encoder stack -- hand off to a
                    # fresh process rather than importing it in-place (see
                    # module docstring: loading both stacks segfaults).
                    #
                    # NOTE: os.execv is NOT a true exec on Windows -- there is
                    # no such syscall, so CPython emulates it by spawning the
                    # child and BLOCKING the current process until the child
                    # exits. For an interactive chat loop that means the old
                    # process sits alive (still holding its ~GB of loaded
                    # models in memory) for the child's entire lifetime --
                    # every switch permanently doubles resident processes,
                    # which eventually causes new processes to hit
                    # ERROR_ACCESS_DENIED trying to mmap the same cross-
                    # encoder weights file. subprocess.Popen (detached, not
                    # waited on) + sys.exit() actually terminates this
                    # process immediately instead.
                    print(colors.status(f"  (switching to {parts[1]} -- restarting)"))
                    sys.stdout.flush()
                    import subprocess
                    subprocess.Popen([sys.executable, "-m", "chatbot", "--pipeline", parts[1]])
                    sys.exit(0)
                pipeline = parts[1]
                answer_fn = _load_answer_fn(pipeline)
                history.clear()
                print(colors.status(f"  (switched to {pipeline}; history cleared)"))
            elif len(parts) == 2 and parts[1] == pipeline:
                print(colors.status(f"  (already on {pipeline})"))
            else:
                print(colors.warn(f"  usage: /pipeline {'|'.join(PIPELINES)}"))
            continue
        if question.lower() in ("/prompts", "prompts"):
            prompts = PROMPT_SETS[pipeline]
            print(colors.status(f"\nSample prompts for '{pipeline}' (type a number to run one):"))
            for i, p in enumerate(prompts, 1):
                print(colors.status(f"  {i}. {p}"))
            print()
            continue
        prompts = PROMPT_SETS[pipeline]
        if question.isdigit() and 1 <= int(question) <= len(prompts):
            question = prompts[int(question) - 1]
            print(colors.status(f"  (running prompt {question!r})"))

        standalone = condense(client, history, question)
        if standalone != question:
            print(colors.status(f"  (standalone query: {standalone!r})"))

        result = answer_fn(standalone)
        history.append((question, _answer_text(result)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive chatbot over the RAG pipelines.")
    parser.add_argument("--pipeline", choices=list(PIPELINES), default="langchain")
    args = parser.parse_args()
    chat(args.pipeline)
