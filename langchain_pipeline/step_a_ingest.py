"""
STEP A -- Layout-aware ingestion & chunking  (LangChain track)

    python -m langchain_pipeline.step_a_ingest

Same parsing/chunking core as the LlamaIndex track (common/parsing.py) so the
two pipelines are genuinely comparable. Only the framework object differs:
LangChain `Document` instead of LlamaIndex `TextNode`.

If you prefer pure LangChain loaders, the equivalent is:
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
...but neither keeps tables atomic, which is why we use the shared core.
"""
from __future__ import annotations

import json
import random

from langchain_core.documents import Document

from common import config
from common.parsing import chunk_directory, summarize
from llamaindex_pipeline.step_a_ingest import chunk_uuid


def build_documents() -> list[Document]:
    config.banner("STEP A (LangChain) -- ingest & chunk")
    chunks = chunk_directory(config.DATA_DIR)
    summarize(chunks)

    # NOTE: LangChain's QdrantVectorStore auto-generates UUID point IDs when you
    # do not pass `ids=`, so this track sidesteps the "Point id is not a valid
    # UUID" wall the LlamaIndex track hits. We still carry a readable chunk_key
    # for debugging, and a deterministic UUID so re-ingest upserts, not duplicates.
    docs = []
    for i, c in enumerate(chunks):
        key = f"{c.metadata['file_name']}::p{c.metadata['page']}::{i}"
        docs.append(Document(
            page_content=c.text,
            metadata={**c.metadata, "chunk_key": key, "chunk_id": chunk_uuid(key)},
        ))

    config.NODES_CACHE.write_text(
        "\n".join(json.dumps({"id": d.metadata["chunk_id"],
                              "text": d.page_content,
                              "metadata": d.metadata}) for d in docs),
        encoding="utf-8",
    )
    print(f"\n  wrote {len(docs)} documents -> {config.NODES_CACHE}")
    return docs


def inspect(docs: list[Document], k: int = 2) -> None:
    config.banner("INSPECT -- verify before indexing")
    tables = [d for d in docs if d.metadata.get("content_type") == "table"]
    texts = [d for d in docs if d.metadata.get("content_type") != "table"]
    for d in random.sample(texts, min(k, len(texts))):
        print(f"\n--- TEXT {d.metadata['file_name']} p.{d.metadata['page']} ---")
        print(d.page_content[:400].replace("\n", " "))
    if tables:
        print(f"\n--- TABLE {tables[0].metadata['file_name']} "
              f"p.{tables[0].metadata['page']} ---")
        print(tables[0].page_content[:800])
    else:
        print("\n  ! No table chunks -- your parser is flattening tables.")


if __name__ == "__main__":
    inspect(build_documents())
