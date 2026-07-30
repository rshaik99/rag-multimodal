"""
PHASE 2 -- Strategy A: Visual Summary Indexing.

    python -m phase2_multimodal.ingest_images

Extract every figure/chart/diagram from your PDFs, have a vision LLM produce a
DATA-EXTRACTING description (not a vibes description), and index a COMPOSITE
node with content_type="image".

Indexed into the LANGCHAIN collection (`<QDRANT_COLLECTION>_lc`), not the
LlamaIndex one -- because `langchain_pipeline/graph_router.py` is the only
router in this lab that actually does retrieval-driven visual routing.
Indexing into the LlamaIndex collection would leave the images unreachable
by the thing meant to retrieve them. Requires the LangChain collection to
already exist: run `python -m langchain_pipeline.run_all --recreate` first.

Why this is the right first move:
  * Reuses your whole Step B/C stack unchanged -- retrieval stays text-only.
  * ~80% of the value of full multi-modal RAG for ~10% of the work.
  * A good VLM summary captures axis labels, trends and annotations that
    CLIP-style joint embeddings routinely miss.

The composite node -- this is the part people get wrong:
    caption + surrounding page text + VLM description + TRANSCRIBED DATA POINTS
The transcribed numbers become searchable tokens for the SPARSE leg, which is
how "what was Q3 cloud revenue" finds a bar chart.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from common import config

VLM_PROMPT = """You are indexing this figure for a retrieval system. Produce a
dense, factual description optimised for SEARCH, not for prose quality.

Output exactly these sections:
TYPE: (bar chart | line chart | pie chart | table image | flow diagram |
       architecture diagram | screenshot | photo | other)
TITLE: any visible title or caption text, verbatim
AXES/LEGEND: axis labels, units, legend entries, verbatim
DESCRIPTION: 2-4 sentences on what the figure SHOWS -- trends, comparisons,
  inflection points, anomalies, and any annotations.
DATA POINTS: transcribe every readable value as "series: x=y" pairs. If exact
  values are not printed, give best estimates and mark them (approx).
ENTITIES: every proper noun, product name, code, or identifier visible.

Be exhaustive on numbers and labels. Do not editorialise."""


# ------------------------------------------------------------------ extraction
def extract_figures(pdf_path: Path, min_px: int = 180) -> list[dict]:
    """
    Pull raster images out of a PDF with page + bbox provenance, plus the text
    surrounding each image (which carries the caption).
    Requires: pip install pymupdf
    """
    import fitz  # PyMuPDF

    out: list[dict] = []
    doc = fitz.open(pdf_path)
    for pno in range(len(doc)):
        page = doc[pno]
        page_text = page.get_text("text")
        for i, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.width < min_px or pix.height < min_px:
                continue                       # skip logos, bullets, rules
            if pix.n - pix.alpha >= 4:         # CMYK -> RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
            name = f"{pdf_path.stem}_p{pno + 1}_fig{i + 1}.png"
            path = config.ASSETS_DIR / name
            pix.save(str(path))

            rects = page.get_image_rects(xref)
            bbox = list(rects[0]) if rects else None
            out.append({
                "image_path": str(path),
                "file_name": pdf_path.name,
                "page": pno + 1,
                "bbox": bbox,
                "nearby_text": page_text[:1200],
            })
    doc.close()
    return out


# ------------------------------------------------------------------ description
def describe(image_path: str, provider: str = "openai") -> str:
    """Vision-LLM description. `provider`: 'openai' | 'anthropic'."""
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode()

    if provider == "anthropic" and config.ANTHROPIC_API_KEY:
        import anthropic
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=1200,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/png", "data": b64}},
                {"type": "text", "text": VLM_PROMPT},
            ]}],
        )
        return msg.content[0].text

    from openai import OpenAI
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=config.VISION_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": VLM_PROMPT},
            # detail="high" costs more tokens but is required to read axis labels
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}},
        ]}],
    )
    return resp.choices[0].message.content


def build_composite(fig: dict, description: str) -> str:
    """The composite text that actually gets embedded. Order matters for recall."""
    return (
        f"[FIGURE] {fig['file_name']} · page {fig['page']}\n"
        f"{description}\n\n"
        f"SURROUNDING PAGE TEXT:\n{fig['nearby_text'][:800]}"
    )


# ------------------------------------------------------------------ indexing
def main(provider: str = "openai") -> None:
    config.banner("PHASE 2 -- visual summary indexing")
    config.require_openai()

    pdfs = sorted(config.DATA_DIR.rglob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs in {config.DATA_DIR}")

    figures: list[dict] = []
    for p in pdfs:
        figs = extract_figures(p)
        print(f"  {p.name}: {len(figs)} figures")
        figures.extend(figs)

    if not figures:
        print("  No figures found. (Vector/text-only PDFs have no raster images — "
              "for those, rasterise whole pages instead: page.get_pixmap(dpi=150).)")
        return

    from langchain_core.documents import Document
    from langchain_qdrant import QdrantVectorStore, RetrievalMode
    from qdrant_client import QdrantClient

    from langchain_pipeline.step_b_index import LC_COLLECTION, get_embeddings, get_sparse
    from llamaindex_pipeline.step_a_ingest import chunk_uuid

    if not QdrantClient(url=config.QDRANT_URL).collection_exists(LC_COLLECTION):
        raise SystemExit(
            f"Collection '{LC_COLLECTION}' does not exist yet.\n"
            "  Run:  python -m langchain_pipeline.run_all --recreate\n"
            "  first, so there is a hybrid collection for the image nodes to join."
        )

    docs, ids = [], []
    for i, fig in enumerate(figures):
        print(f"  describing {Path(fig['image_path']).name} ...")
        desc = describe(fig["image_path"], provider=provider)
        key = f"{fig['file_name']}::p{fig['page']}::img{i}"
        docs.append(Document(
            page_content=build_composite(fig, desc),
            metadata={
                "file_name": fig["file_name"],
                "page": fig["page"],
                "content_type": "image",       # <-- what the router keys off
                "image_path": fig["image_path"],
                "bbox": json.dumps(fig["bbox"]),
                "section": "",
                "doc_type": "pdf",
                "chunk_key": key,
            },
        ))
        # Qdrant point IDs must be a UUID or unsigned int -- same rule as Step A.
        # Hash the readable key deterministically so re-ingest upserts, not duplicates.
        ids.append(chunk_uuid(key))

    QdrantVectorStore.from_documents(
        docs,
        ids=ids,
        embedding=get_embeddings(),
        sparse_embedding=get_sparse(),
        retrieval_mode=RetrievalMode.HYBRID,
        url=config.QDRANT_URL,
        collection_name=LC_COLLECTION,
        vector_name="dense",
        sparse_vector_name="sparse",
        batch_size=64,
    )
    print(f"\n  indexed {len(docs)} image nodes into '{LC_COLLECTION}'")
    print("  now try:  python -m langchain_pipeline.graph_router "
          "\"what does the revenue chart show?\"")


if __name__ == "__main__":
    main()
