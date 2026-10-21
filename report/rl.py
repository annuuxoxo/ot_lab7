"""ReportLab helpers shared by the report sections."""
import os
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
                                NextPageTemplate, KeepTogether, Preformatted, FrameBreak)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from PIL import Image as PILImage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "figures")

base = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", parent=base["Title"], fontSize=22, leading=26, spaceAfter=10),
    "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=15, leading=18, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#1f3b66")),
    "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=12, leading=15, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1f3b66")),
    "h3": ParagraphStyle("h3", parent=base["Heading3"], fontSize=10, leading=12, spaceBefore=6, spaceAfter=2),
    "p": ParagraphStyle("p", parent=base["BodyText"], fontSize=9, leading=11.5, alignment=TA_JUSTIFY, spaceAfter=4),
    "pl": ParagraphStyle("pl", parent=base["BodyText"], fontSize=9, leading=11.5, spaceAfter=3),
    "small": ParagraphStyle("small", parent=base["BodyText"], fontSize=7.5, leading=9.2, spaceAfter=2),
    "tiny": ParagraphStyle("tiny", parent=base["BodyText"], fontSize=6.6, leading=7.9, spaceAfter=1),
    "cap": ParagraphStyle("cap", parent=base["BodyText"], fontSize=7.5, leading=9, textColor=colors.HexColor("#444444"), alignment=TA_CENTER, spaceAfter=6),
    "cell": ParagraphStyle("cell", parent=base["BodyText"], fontSize=7.4, leading=8.9, spaceAfter=0),
    "cellb": ParagraphStyle("cellb", parent=base["BodyText"], fontSize=7.4, leading=8.9, spaceAfter=0, fontName="Helvetica-Bold"),
    "code": ParagraphStyle("code", parent=base["Code"], fontSize=7, leading=8.4),
    "note": ParagraphStyle("note", parent=base["BodyText"], fontSize=8.4, leading=10.5, backColor=colors.HexColor("#fff6dd"), borderPadding=4, spaceAfter=6),
}


def P(t, s="p"):
    return Paragraph(t, S[s])


def H1(t): return Paragraph(t, S["h1"])
def H2(t): return Paragraph(t, S["h2"])
def H3(t): return Paragraph(t, S["h3"])


def img(name, width_mm=170, max_h_mm=None):
    path = name if os.path.isabs(name) else os.path.join(FIG, name)
    w, h = PILImage.open(path).size
    wd = width_mm * mm
    ht = wd * h / w
    if max_h_mm and ht > max_h_mm * mm:
        ht = max_h_mm * mm; wd = ht * w / h
    return Image(path, width=wd, height=ht)


def images_row(names, width_mm):
    """Several figures side by side (each width_mm wide)."""
    cells = [img(n, width_mm) for n in names]
    t = Table([cells], colWidths=[(width_mm + 1) * mm] * len(cells))
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    return t


def tbl(data, col_widths=None, header=True, font=7.4, zebra=True, align_right_from=None, repeat=1, extra=None):
    """Table from a list of rows; strings become Paragraphs so they wrap."""
    rows = []
    for i, r in enumerate(data):
        row = []
        for c in r:
            if isinstance(c, str):
                row.append(Paragraph(c, S["cellb"] if (header and i == 0) else S["cell"]))
            else:
                row.append(c)
        rows.append(row)
    t = Table(rows, colWidths=[w * mm for w in col_widths] if col_widths else None, repeatRows=repeat if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#999999")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 2.5), ("RIGHTPADDING", (0, 0), (-1, -1), 2.5), ("TOPPADDING", (0, 0), (-1, -1), 1.5),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe7f3"))]
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f5f7fb")))
    if extra:
        st += extra
    t.setStyle(TableStyle(st))
    return t


def fmt(v, d=4):
    try:
        import math
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "n/a"
        return f"{v:.{d}f}"
    except Exception:
        return str(v)


def pm(m, s, d=4):
    return f"{fmt(m, d)} ± {fmt(s, d)}"


def make_doc(path, title="Lab 7"):
    pw, ph = A4
    margin = 14 * mm
    fp = Frame(margin, margin + 4 * mm, pw - 2 * margin, ph - 2 * margin - 6 * mm, id="p")
    pwl, phl = landscape(A4)
    fl = Frame(10 * mm, 12 * mm, pwl - 20 * mm, phl - 20 * mm, id="l")

    def footer(c, d):
        c.saveState(); c.setFont("Helvetica", 7); c.setFillColor(colors.grey)
        c.drawString(margin, 8 * mm, "Optimization Techniques Lab 7 -- Multiobjective Optimization -- roll no. 77")
        c.drawRightString(c._pagesize[0] - margin, 8 * mm, f"page {d.page}")
        c.restoreState()

    doc = BaseDocTemplate(path, pagesize=A4, title=title, author="Roll no. 77",
                          pageTemplates=[PageTemplate(id="portrait", frames=[fp], onPage=footer, pagesize=A4),
                                         PageTemplate(id="landscape", frames=[fl], onPage=footer, pagesize=landscape(A4))])
    return doc
