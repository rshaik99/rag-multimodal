"""
Generates a table-heavy sample PDF in ./data so you can run the whole lab with
zero of your own data.

    pip install reportlab
    python make_sample_docs.py

The document deliberately contains:
  * headings (tests section-metadata extraction)
  * a real grid table (tests table fidelity end to end)
  * an exact identifier "ERR-4417" (tests why you need the SPARSE leg --
    dense-only retrieval reliably fails on this)
"""
from pathlib import Path

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
OUT = DATA / "acme_fy2025_report.pdf"


def build() -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (PageBreak, Paragraph, SimpleDocTemplate,
                                    Spacer, Table, TableStyle)

    ss = getSampleStyleSheet()
    story = []

    def h(text, lvl=1):
        story.append(Paragraph(text, ss[f"Heading{lvl}"]))
        story.append(Spacer(1, 6))

    def p(text):
        story.append(Paragraph(text, ss["BodyText"]))
        story.append(Spacer(1, 6))

    h("Acme Corporation — FY2025 Annual Report", 1)

    h("1. Revenue by Segment", 2)
    p("Total revenue reached $1,048M in FY2025, up 21% year over year. The Cloud "
      "segment overtook Hardware for the first time in Q3 FY2025. Quarterly "
      "revenue by reporting segment is presented in Table 1 below.")

    rows = [
        ["Segment", "Q1 ($M)", "Q2 ($M)", "Q3 ($M)", "Q4 ($M)", "FY Total ($M)"],
        ["Cloud Services", "210", "240", "300", "340", "1,090"],
        ["Hardware", "180", "179", "181", "178", "718"],
        ["Professional Services", "150", "141", "130", "120", "541"],
        ["Licensing", "62", "64", "67", "71", "264"],
        ["Total", "602", "624", "678", "709", "2,613"],
    ]
    t = Table(rows, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde5ef")),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f2f2f2")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))
    p("Table 1: Quarterly revenue by segment, FY2025.")

    h("2. Operating Margin", 2)
    p("Consolidated operating margin expanded 340 basis points to 18.7%. Cloud "
      "Services carried a 31.2% segment margin, while Hardware margin compressed "
      "to 8.4% on component cost inflation.")

    story.append(PageBreak())

    h("3. Known Issues and Incident Log", 2)
    p("During Q3 FY2025 the platform experienced an elevated rate of failed "
      "provisioning requests. The root cause was tracked as incident "
      "<b>ERR-4417</b>: a race condition in the tenant-provisioning state "
      "machine that surfaced when two region-failover events overlapped within "
      "a 90-second window. The mitigation shipped in release 7.4.2 and "
      "introduced an idempotency key on the provisioning API. No customer data "
      "was lost. Mean time to recovery was 42 minutes.")

    p("A secondary issue, ERR-4418, affected billing reconciliation for 1,204 "
      "accounts and was resolved in release 7.4.3.")

    h("4. Headcount", 2)
    rows2 = [
        ["Function", "FY2024", "FY2025", "Change"],
        ["Engineering", "1,420", "1,690", "+270"],
        ["Sales & Marketing", "980", "1,010", "+30"],
        ["G&A", "310", "298", "-12"],
    ]
    t2 = Table(rows2, hAlign="LEFT")
    t2.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde5ef")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t2)
    story.append(Spacer(1, 10))
    p("Table 2: Headcount by function.")

    SimpleDocTemplate(str(OUT), pagesize=LETTER,
                      title="Acme FY2025 Annual Report").build(story)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    try:
        build()
    except ImportError:
        raise SystemExit("pip install reportlab  # then re-run")
