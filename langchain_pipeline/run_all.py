"""
End-to-end LangChain run.

    python -m langchain_pipeline.run_all --recreate "What was Q3 Cloud revenue?"
"""
from __future__ import annotations

import argparse

from langchain_pipeline import (step_a_ingest, step_b_index,
                                step_c_retrieve_rerank, step_d_cited_answer)

DEFAULT_QUERIES = [
    "What caused incident ERR-4417 and how was it fixed?",
    "How did Cloud Services revenue trend across FY2025 quarters?",
    "What is the company's policy on remote work?",     # must refuse
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*")
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--skip-ingest", action="store_true")
    args = ap.parse_args()

    if not args.skip_ingest:
        docs = step_a_ingest.build_documents()
        step_a_ingest.inspect(docs)
        step_b_index.main(recreate=args.recreate)

    queries = [" ".join(args.query)] if args.query else DEFAULT_QUERIES
    step_c_retrieve_rerank.compare(queries[0])
    for q in queries:
        step_d_cited_answer.answer(q)


if __name__ == "__main__":
    main()
