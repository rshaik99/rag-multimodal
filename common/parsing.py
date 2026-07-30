"""
Shared, framework-agnostic layout-aware parsing + table-atomic chunking.

Both the LlamaIndex and the LangChain tracks call into this so the two
pipelines are genuinely comparable -- the only thing that differs downstream
is the framework, not the data.

Parser preference (best -> most available):
    1. LlamaParse  (API v2, tier="agentic")   -- needs LLAMA_CLOUD_API_KEY
    2. Docling                                -- free, local, TableFormer
    3. PyMuPDF4LLM                            -- fast, local, lab default
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from common import config


# ============================================================== data model
@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================== parsing
def parse_to_markdown(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, markdown_text), ...]. 1-based page numbers."""
    if config.LLAMA_CLOUD_API_KEY:
        try:
            return _parse_llamaparse(path)
        except Exception as exc:                      # noqa: BLE001
            print(f"  ! LlamaParse failed ({exc}); falling back to local parser")
    try:
        return _parse_pymupdf4llm(path)
    except ImportError:
        print("  ! pymupdf4llm not installed; trying docling")
        return _parse_docling(path)


def _parse_llamaparse(path: Path) -> list[tuple[int, str]]:
    """
    LlamaParse API v2.

    v2 replaced v1's flat parameter soup with a `tier` + nested option buckets.
    `parse_mode`, `premium_mode`, `gpt4o_mode`, `fast_mode` are all GONE.
    Tiers: fast | cost_effective | agentic | agentic_plus
    """
    from llama_cloud_services import LlamaParse

    parser = LlamaParse(
        api_key=config.LLAMA_CLOUD_API_KEY,
        tier="agentic",              # best table fidelity; use "cost_effective" to save credits
        result_type="markdown",
        num_workers=4,
        verbose=False,
    )
    docs = parser.load_data(str(path))
    return [(i + 1, d.text) for i, d in enumerate(docs)]


def _parse_pymupdf4llm(path: Path) -> list[tuple[int, str]]:
    """Local, fast, preserves headings and pipe tables. Lab default."""
    import pymupdf4llm

    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True)
    out = []
    for i, pg in enumerate(pages):
        text = pg["text"] if isinstance(pg, dict) else str(pg)
        page_no = (pg.get("metadata", {}).get("page") if isinstance(pg, dict) else None) or i + 1
        out.append((int(page_no), text))
    return out


def _parse_docling(path: Path) -> list[tuple[int, str]]:
    """Free local layout model with real table structure (TableFormer)."""
    from docling.document_converter import DocumentConverter

    result = DocumentConverter().convert(str(path))
    # Docling gives one coherent Markdown doc; page attribution needs the
    # provenance API, so we degrade gracefully to page 1.
    return [(1, result.document.export_to_markdown())]


# ============================================================== chunking
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _split_markdown_blocks(md: str) -> list[tuple[str, str]]:
    """
    Split markdown into ("table" | "text", block) preserving order.

    THE point of this function: a pipe-table block is emitted whole, so it can
    never be cut in half by a length-based splitter. A half table is worse than
    no table -- each half looks authoritative and is wrong.
    """
    blocks: list[tuple[str, str]] = []
    buf: list[str] = []
    mode = "text"

    def flush():
        if buf and "".join(buf).strip():
            blocks.append((mode, "\n".join(buf).strip()))
        buf.clear()

    for line in md.splitlines():
        is_table = bool(_TABLE_LINE.match(line))
        new_mode = "table" if is_table else "text"
        if new_mode != mode:
            flush()
            mode = new_mode
        # A heading starts a new block, so every block carries exactly one
        # section in its metadata instead of straddling two.
        elif _HEADING.match(line) and buf:
            flush()
        buf.append(line)
    flush()

    # A single stray "|" line is not a table; re-label short ones as text.
    return [(m if not (m == "table" and len(b.splitlines()) < 2) else "text", b)
            for m, b in blocks]


def _track_section(md: str) -> list[tuple[str, str]]:
    """Annotate each line's nearest enclosing heading. Returns [(section, line)]."""
    out, current = [], ""
    for line in md.splitlines():
        m = _HEADING.match(line)
        if m:
            # strip markdown emphasis so section names are clean metadata
            current = re.sub(r"[*_`#]", "", m.group(2)).strip()
        out.append((current, line))
    return out


def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)      # good enough; avoids a tiktoken dependency here


def _sentence_pack(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Token-aware packing on sentence boundaries -- never mid-sentence."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", text)
    chunks, cur, cur_tok = [], [], 0
    for s in sentences:
        st = _approx_tokens(s)
        if cur and cur_tok + st > max_tokens:
            chunks.append(" ".join(cur))
            # rebuild overlap from the tail
            back, tok = [], 0
            for prev in reversed(cur):
                tok += _approx_tokens(prev)
                back.insert(0, prev)
                if tok >= overlap_tokens:
                    break
            cur, cur_tok = back, tok
        cur.append(s)
        cur_tok += st
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if c.strip()]


def chunk_document(
    path: Path,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Chunk]:
    """Parse -> section-aware, table-atomic chunks with full metadata."""
    chunks: list[Chunk] = []
    for page_no, md in parse_to_markdown(path):
        sectioned = _track_section(md)
        # regroup lines back into markdown, remembering the section per block
        line_sections = {i: sec for i, (sec, _) in enumerate(sectioned)}
        rebuilt = "\n".join(l for _, l in sectioned)

        # --- resolve each block's section, then post-process ------------
        raw: list[tuple[str, str, str]] = []      # (kind, block, section)
        cursor = 0
        for kind, block in _split_markdown_blocks(rebuilt):
            idx = rebuilt.find(block, cursor)
            cursor = idx + len(block) if idx >= 0 else cursor
            raw.append((kind, block,
                        line_sections.get(rebuilt[:max(idx, 0)].count("\n"), "")))

        # (a) A short caption immediately AFTER a table belongs TO the table --
        #     "Table 1: Quarterly revenue by segment" is a strong retrieval
        #     signal and is useless as a standalone chunk.
        merged: list[tuple[str, str, str]] = []
        for kind, block, section in raw:
            if (merged and merged[-1][0] == "table" and kind == "text"
                    and len(block) < 200
                    and re.match(r"^\s*(table|figure|exhibit|chart)\b", block, re.I)):
                k, b, s = merged[-1]
                merged[-1] = (k, f"{b}\n\nCaption: {block.strip()}", s)
                continue
            merged.append((kind, block, section))

        # (b) Merge consecutive text blocks sharing a section, so a heading
        #     never becomes a lonely 40-character chunk.
        packed: list[tuple[str, str, str]] = []
        for kind, block, section in merged:
            if (packed and kind == "text" and packed[-1][0] == "text"
                    and packed[-1][2] == section):
                k, b, s = packed[-1]
                packed[-1] = (k, f"{b}\n\n{block}", s)
            else:
                packed.append((kind, block, section))

        for kind, block, section in packed:
            base_meta = {
                "file_name": path.name,
                "file_path": str(path),
                "page": page_no,
                "section": section,
                "doc_type": path.suffix.lstrip(".").lower(),
                "content_type": kind,
            }

            if kind == "table":
                # ATOMIC. Oversized is fine; truncated is not.
                header = f"[TABLE] {path.name} · page {page_no}"
                if section:
                    header += f" · section: {section}"
                chunks.append(Chunk(f"{header}\n\n{block}", dict(base_meta)))
            else:
                for piece in _sentence_pack(block, chunk_size, chunk_overlap):
                    chunks.append(Chunk(piece, dict(base_meta)))

    return [c for c in chunks if len(c.text.strip()) > 30]


def chunk_directory(data_dir: Path = config.DATA_DIR) -> list[Chunk]:
    exts = {".pdf", ".md", ".txt"}
    files = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in exts)
    if not files:
        raise SystemExit(
            f"No documents in {data_dir}.\n"
            "  Drop PDFs there, or run:  python make_sample_docs.py"
        )
    all_chunks: list[Chunk] = []
    for f in files:
        print(f"  parsing {f.name} ...")
        if f.suffix.lower() in {".md", ".txt"}:
            md = f.read_text(encoding="utf-8", errors="ignore")
            for kind, block in _split_markdown_blocks(md):
                meta = {"file_name": f.name, "file_path": str(f), "page": 1,
                        "section": "", "doc_type": f.suffix.lstrip("."),
                        "content_type": kind}
                if kind == "table":
                    all_chunks.append(Chunk(block, meta))
                else:
                    for piece in _sentence_pack(block, config.CHUNK_SIZE,
                                                config.CHUNK_OVERLAP):
                        all_chunks.append(Chunk(piece, dict(meta)))
        else:
            all_chunks.extend(chunk_document(f))
    return all_chunks


def summarize(chunks: Iterable[Chunk]) -> None:
    chunks = list(chunks)
    tables = sum(1 for c in chunks if c.metadata.get("content_type") == "table")
    lens = [len(c.text) for c in chunks] or [0]
    print(f"\n  chunks: {len(chunks)}   tables: {tables}   text: {len(chunks) - tables}")
    print(f"  chars  min={min(lens)}  mean={sum(lens) // len(lens)}  max={max(lens)}")
