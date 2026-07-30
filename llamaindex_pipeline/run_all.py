"""
End-to-end LlamaIndex run.

    python -m llamaindex_pipeline.run_all --recreate "What was Q3 Cloud revenue?"
"""
from __future__ import annotations

import argparse

from llamaindex_pipeline import step_a_ingest, step_b_index, step_c_retrieve_rerank
from llamaindex_pipeline import step_d_cited_answer

DEFAULT_QUERIES = [
    "What caused incident ERR-4417 and how was it fixed?",   # exact ID -> needs SPARSE
    "How did Cloud Services revenue trend across FY2025 quarters?",  # -> needs the TABLE
    "What is the company's policy on remote work?",          # out of scope -> must refuse
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*", help="question(s); omit to run the demo set")
    ap.add_argument("--recreate", action="store_true", help="rebuild the collection")
    ap.add_argument("--skip-ingest", action="store_true")
    args = ap.parse_args()

    if not args.skip_ingest:
        nodes = step_a_ingest.build_nodes()
        step_a_ingest.inspect(nodes)
        step_b_index.main(recreate=args.recreate)

    queries = [" ".join(args.query)] if args.query else DEFAULT_QUERIES
    step_c_retrieve_rerank.compare(queries[0])
    for q in queries:
        step_d_cited_answer.answer(q)


if __name__ == "__main__":
    main()
