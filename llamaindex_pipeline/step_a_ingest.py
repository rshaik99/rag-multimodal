"""
STEP A -- Layout-aware ingestion & chunking  (LlamaIndex track)

    python -m llamaindex_pipeline.step_a_ingest

Output: storage/nodes.jsonl  (inspect it before moving on -- this is the gate)

Design decisions that matter:
  * Parse to MARKDOWN, not plain text -> headings survive, pipe tables survive.
  * Tables are ATOMIC nodes. Never length-split a table.
  * Metadata is attached at PARSE time (page numbers cannot be recovered later).
  * excluded_*_metadata_keys keeps file paths out of the embedding vector while
    still returning them for citation -- a small but real retrieval-quality win.
"""
from __future__ import annotations

import json
import random
import uuid

from llama_index.core.schema import TextNode

from common import config
from common.parsing import chunk_directory, summarize

# Namespace for deterministic chunk IDs. Keep it fixed across runs so that
# re-ingesting the same document UPSERTS rather than duplicating.
NS = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")


def chunk_uuid(key: str) -> str:
    """
    Qdrant point IDs MUST be a UUID or an unsigned integer. A readable id like
    'report.pdf::p1::3' is rejected at upsert time with
    "Point id ... is not a valid UUID" -- a classic Day-1 lab wall.

    Fix: hash the readable key into a deterministic UUID5, and keep the readable
    key in metadata as `chunk_key` for debugging and citations.
    """
    return str(uuid.uuid5(NS, key))


def build_nodes() -> list[TextNode]:
    config.banner("STEP A -- ingest & chunk")
    chunks = chunk_directory(config.DATA_DIR)
    summarize(chunks)

    nodes: list[TextNode] = []
    for i, c in enumerate(chunks):
        key = f"{c.metadata['file_name']}::p{c.metadata['page']}::{i}"
        node = TextNode(
            text=c.text,
            id_=chunk_uuid(key),
            metadata={**c.metadata, "chunk_key": key},
            # Keep noisy keys out of the embedded string, but keep them for citations.
            excluded_embed_metadata_keys=["file_path", "doc_type", "chunk_key"],
            excluded_llm_metadata_keys=["file_path", "chunk_key"],
        )
        nodes.append(node)

    config.NODES_CACHE.write_text(
        "\n".join(json.dumps({"id": n.id_, "text": n.text, "metadata": n.metadata})
                  for n in nodes),
        encoding="utf-8",
    )
    print(f"\n  wrote {len(nodes)} nodes -> {config.NODES_CACHE}")
    return nodes


def inspect(nodes: list[TextNode], k: int = 3) -> None:
    """THE GATE. If a table node looks like number soup, stop and switch parsers."""
    config.banner("INSPECT -- verify before indexing")
    tables = [n for n in nodes if n.metadata.get("content_type") == "table"]
    texts = [n for n in nodes if n.metadata.get("content_type") != "table"]

    for n in random.sample(texts, min(k, len(texts))):
        print(f"\n--- TEXT  {n.metadata['file_name']} p.{n.metadata['page']} "
              f"[{n.metadata.get('section', '')}] ---")
        print(n.text[:420].replace("\n", " ") + ("..." if len(n.text) > 420 else ""))

    if tables:
        n = tables[0]
        print(f"\n--- TABLE  {n.metadata['file_name']} p.{n.metadata['page']} ---")
        print(n.text[:900])
        print("\n  ^ Does this still look like a table (pipes / aligned columns)?")
        print("    If it is a flat run of numbers, STOP: switch parser "
              "(docling -> LlamaParse agentic) before indexing.")
    else:
        print("\n  ! No table nodes found. If your PDFs contain tables, your "
              "parser is flattening them -- switch parsers now.")


if __name__ == "__main__":
    inspect(build_nodes())
