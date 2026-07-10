"""Render the maintained Torch-OSQP pipeline Markdown as a polished PDF."""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "FULL_DEVELOPMENT_AND_VALIDATION_PIPELINE.md"
OUTPUT = ROOT / "output" / "pdf" / "Full Development and Validation Pipeline - Revised.pdf"


class PipelineDocument(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates(PageTemplate(id="main", frames=frame, onPage=draw_page))

    def afterFlowable(self, flowable):
        if not isinstance(flowable, Paragraph):
            return
        style = flowable.style.name
        match = re.fullmatch(r"Heading([1-3])Custom", style)
        if match:
            level = int(match.group(1)) - 1
            text = flowable.getPlainText()
            key = f"heading-{self.page}-{len(text)}-{level}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=False)
            self.notify("TOCEntry", (level, text, self.page, key))


def draw_page(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#D7DEE8"))
    canvas.line(document.leftMargin, 0.58 * inch, letter[0] - document.rightMargin, 0.58 * inch)
    canvas.setFillColor(colors.HexColor("#536273"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(document.leftMargin, 0.38 * inch, "Torch-OSQP Development and Validation Pipeline")
    canvas.drawRightString(letter[0] - document.rightMargin, 0.38 * inch, str(document.page))
    canvas.restoreState()


def styles():
    sheet = getSampleStyleSheet()
    navy = colors.HexColor("#173A5E")
    blue = colors.HexColor("#246B9E")
    sheet.add(ParagraphStyle(name="TitleCustom", parent=sheet["Title"], fontName="Helvetica-Bold", fontSize=25, leading=30, textColor=navy, alignment=TA_CENTER, spaceAfter=18))
    sheet.add(ParagraphStyle(name="Subtitle", parent=sheet["Normal"], fontSize=11, leading=16, textColor=colors.HexColor("#536273"), alignment=TA_CENTER, spaceAfter=8))
    sheet.add(ParagraphStyle(name="TOCTitle", parent=sheet["Heading2"], fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=blue, spaceBefore=18, spaceAfter=8))
    for level, size, before, after in ((1, 19, 18, 10), (2, 15, 15, 8), (3, 12, 12, 6)):
        sheet.add(ParagraphStyle(name=f"Heading{level}Custom", parent=sheet[f"Heading{level}"], fontName="Helvetica-Bold", fontSize=size, leading=size + 4, textColor=navy if level == 1 else blue, spaceBefore=before, spaceAfter=after, keepWithNext=True))
    sheet.add(ParagraphStyle(name="Subheading", parent=sheet["Heading4"], fontName="Helvetica-Bold", fontSize=10, leading=13, textColor=blue, spaceBefore=8, spaceAfter=4, keepWithNext=True))
    sheet.add(ParagraphStyle(name="BodyCustom", parent=sheet["BodyText"], fontSize=9.4, leading=13.2, textColor=colors.HexColor("#25313D"), spaceAfter=6))
    sheet.add(ParagraphStyle(name="TableHeaderCustom", parent=sheet["BodyText"], fontName="Helvetica-Bold", fontSize=9.0, leading=12.5, textColor=colors.white, spaceAfter=0))
    sheet.add(ParagraphStyle(name="BulletCustom", parent=sheet["BodyText"], fontSize=9.2, leading=12.8, leftIndent=4, textColor=colors.HexColor("#25313D")))
    sheet.add(ParagraphStyle(name="CodeCustom", fontName="Courier", fontSize=7.6, leading=10, leftIndent=8, rightIndent=8, borderColor=colors.HexColor("#C8D3DF"), borderWidth=0.6, borderPadding=8, backColor=colors.HexColor("#F4F7FA"), spaceBefore=4, spaceAfter=8))
    return sheet


def inline_markup(text):
    text = escape(text.strip())
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    return text


def parse_table(lines, sheet):
    rows = [[inline_markup(cell) for cell in line.strip().strip("|").split("|")] for line in lines]
    rows = [rows[0]] + rows[2:]
    data = [
        [
            Paragraph(
                cell,
                sheet["TableHeaderCustom"] if row_index == 0 else sheet["BodyCustom"],
            )
            for cell in row
        ]
        for row_index, row in enumerate(rows)
    ]
    table = Table(data, repeatRows=1, hAlign="LEFT", colWidths=[None] * len(data[0]))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#173A5E")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D3DF")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def markdown_story(text):
    sheet = styles()
    story = []
    lines = text.splitlines()
    index = 0
    first_heading = True
    while index < len(lines):
        line = lines[index]
        if line.strip() == "<!-- PAGEBREAK -->":
            story.append(PageBreak())
            index += 1
            continue
        if line.startswith("```"):
            code = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code.append(lines[index])
                index += 1
            story.append(KeepTogether(Preformatted("\n".join(code), sheet["CodeCustom"])))
            index += 1
            continue
        if line.startswith("|") and index + 1 < len(lines) and lines[index + 1].startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            story.append(parse_table(table_lines, sheet))
            story.append(Spacer(1, 8))
            continue
        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2)
            if first_heading:
                story.append(Spacer(1, 1.05 * inch))
                story.append(Paragraph(inline_markup(title), sheet["TitleCustom"]))
                first_heading = False
            else:
                if level == 4:
                    story.append(Paragraph(inline_markup(title), sheet["Subheading"]))
                    index += 1
                    continue
                if title == "Part I - Executive Summary":
                    toc = TableOfContents()
                    toc.levelStyles = [
                        ParagraphStyle(name="TOC1", fontSize=9, leading=13, leftIndent=0, textColor=colors.HexColor("#173A5E")),
                        ParagraphStyle(name="TOC2", fontSize=8.5, leading=12, leftIndent=14, textColor=colors.HexColor("#536273")),
                        ParagraphStyle(name="TOC3", fontSize=8, leading=11, leftIndent=28, textColor=colors.HexColor("#536273")),
                    ]
                    story.extend(
                        [
                            Paragraph("Table of Contents", sheet["TOCTitle"]),
                            toc,
                            PageBreak(),
                            Paragraph(inline_markup(title), sheet["Heading1Custom"]),
                        ]
                    )
                else:
                    document_level = max(1, level - 1)
                    story.append(
                        Paragraph(
                            inline_markup(title),
                            sheet[f"Heading{document_level}Custom"],
                        )
                    )
            index += 1
            continue
        if line.startswith("- "):
            items = []
            while index < len(lines) and lines[index].startswith("- "):
                items.append(ListItem(Paragraph(inline_markup(lines[index][2:]), sheet["BulletCustom"]), leftIndent=12))
                index += 1
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=18, bulletFontSize=6, spaceAfter=6))
            continue
        if re.match(r"^\d+\.\s", line):
            items = []
            while index < len(lines) and re.match(r"^\d+\.\s", lines[index]):
                content = re.sub(r"^\d+\.\s", "", lines[index])
                items.append(ListItem(Paragraph(inline_markup(content), sheet["BulletCustom"]), leftIndent=14))
                index += 1
            story.append(ListFlowable(items, bulletType="1", leftIndent=20, spaceAfter=6))
            continue
        if not line.strip():
            index += 1
            continue
        paragraph = [line.strip()]
        index += 1
        while index < len(lines) and lines[index].strip() and not re.match(r"^(#{1,4})\s|^-\s|^\d+\.\s|^```|^\||^<!--", lines[index]):
            paragraph.append(lines[index].strip())
            index += 1
        text_value = " ".join(paragraph)
        style = sheet["Subtitle"] if text_value.startswith(("Version:", "Date:", "Status:")) else sheet["BodyCustom"]
        story.append(KeepTogether([Paragraph(inline_markup(text_value), style)]))
    return story


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = PipelineDocument(
        str(OUTPUT),
        pagesize=letter,
        leftMargin=0.68 * inch,
        rightMargin=0.68 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.72 * inch,
        title="Full Development and Validation Pipeline",
        author="PyGRANSO Research Team",
    )
    story = markdown_story(SOURCE.read_text(encoding="utf-8"))
    document.multiBuild(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
