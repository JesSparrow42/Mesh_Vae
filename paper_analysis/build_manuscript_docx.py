#!/usr/bin/env python3
"""Build the editable REACT-Q working manuscript from paper_draft.md."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


INK = "000000"
TEAL = "0F6F73"
PALE_TEAL = "EAF4F4"
PALE_GRAY = "F5F7F8"
BORDER = "D9E1E5"
PLACEHOLDER = "9C3D30"


def set_cell_shading(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_margins(cell, top: int = 100, start: int = 120, bottom: int = 100, end: int = 120) -> None:
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for edge, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = margins.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color: str = BORDER, size: str = "6") -> None:
    properties = table._tbl.tblPr
    borders = properties.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        properties.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = borders.find(qn(f"w:{edge}"))
        if node is None:
            node = OxmlElement(f"w:{edge}")
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), size)
        node.set(qn("w:color"), color)


def keep_with_next(paragraph) -> None:
    properties = paragraph._p.get_or_add_pPr()
    if properties.find(qn("w:keepNext")) is None:
        properties.append(OxmlElement("w:keepNext"))


def remove_paragraph_borders(element) -> None:
    properties = element.get_or_add_pPr()
    borders = properties.find(qn("w:pBdr"))
    if borders is not None:
        properties.remove(borders)


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instruction, separate, end])


def add_inline_markup(paragraph, text: str) -> None:
    token = re.compile(r"(\*\*.+?\*\*|`.+?`|\[\[.+?\]\])")
    position = 0
    for match in token.finditer(text):
        if match.start() > position:
            paragraph.add_run(text[position : match.start()])
        value = match.group(0)
        if value.startswith("**"):
            run = paragraph.add_run(value[2:-2])
            run.bold = True
            if "[[" in value:
                run.font.color.rgb = RGBColor.from_string(PLACEHOLDER)
                run.font.highlight_color = 7
        elif value.startswith("`"):
            run = paragraph.add_run(value[1:-1])
            run.font.name = "Consolas"
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor.from_string("334E5C")
        else:
            run = paragraph.add_run(value)
            run.bold = True
            run.font.color.rgb = RGBColor.from_string(PLACEHOLDER)
            run.font.highlight_color = 7
        position = match.end()
    if position < len(text):
        paragraph.add_run(text[position:])


def configure_styles(document: Document) -> None:
    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    normal.paragraph_format.line_spacing = 1.08

    title = styles["Title"]
    title.font.name = "Arial"
    title.font.size = Pt(18)
    title.font.bold = True
    title.font.color.rgb = RGBColor.from_string(INK)
    title.paragraph_format.space_after = Pt(8)
    remove_paragraph_borders(title._element)

    for name, size, before, after in (
        ("Heading 1", 14, 15, 5),
        ("Heading 2", 11.5, 10, 4),
        ("Heading 3", 10.5, 8, 3),
    ):
        style = styles[name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(INK)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    for name in ("List Bullet", "List Number"):
        styles[name].font.name = "Arial"
        styles[name].font.size = Pt(10.5)
        styles[name].paragraph_format.space_after = Pt(3)


def add_markdown_table(document: Document, rows: list[list[str]]) -> None:
    if len(rows) < 2:
        return
    rows = [row for index, row in enumerate(rows) if index != 1]
    width = len(rows[0])
    table = document.add_table(rows=len(rows), cols=width)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    set_table_borders(table)
    for row_index, values in enumerate(rows):
        for column_index, value in enumerate(values):
            cell = table.cell(row_index, column_index)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            set_cell_shading(cell, TEAL if row_index == 0 else (PALE_GRAY if row_index % 2 == 0 else "FFFFFF"))
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            add_inline_markup(paragraph, value.strip())
            for run in paragraph.runs:
                run.font.name = "Arial"
                run.font.size = Pt(9.2)
                if row_index == 0:
                    run.bold = True
                    run.font.color.rgb = RGBColor(255, 255, 255)
    document.add_paragraph().paragraph_format.space_after = Pt(0)


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    index = start
    while index < len(lines) and lines[index].strip().startswith("|"):
        rows.append([cell.strip() for cell in lines[index].strip().strip("|").split("|")])
        index += 1
    return rows, index


def insert_figure(document: Document, figure: Path, caption: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(figure), width=Inches(6.35))
    caption_paragraph = document.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    caption_paragraph.paragraph_format.space_before = Pt(2)
    caption_paragraph.paragraph_format.space_after = Pt(8)
    first = caption_paragraph.add_run("Figure 4. ")
    first.bold = True
    caption_paragraph.add_run(caption)


def build_document(markdown_path: Path, figure_path: Path, output_path: Path) -> None:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    document = Document()
    section = document.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(1.8)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)
    configure_styles(document)
    add_page_number(section.footer.paragraphs[0])

    title_text = "Three dimensional body shape representations for detecting subclinical atherosclerosis in the REACT study"
    title = document.add_paragraph(style="Title")
    remove_paragraph_borders(title._p)
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title.add_run(title_text)
    subtitle = document.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(12)
    run = subtitle.add_run("Working manuscript")
    run.bold = True
    run.font.size = Pt(10.5)
    run.font.color.rgb = RGBColor.from_string(TEAL)

    index = 1
    inserted_figure = False
    while index < len(lines):
        raw = lines[index]
        line = raw.strip()
        if not line:
            index += 1
            continue
        if line.startswith("**Working manuscript, analysis plan, and results scaffold"):
            index += 1
            continue
        if line == "## Proposed final figures and tables":
            break
        if line.startswith("|"):
            rows, index = parse_table(lines, index)
            add_markdown_table(document, rows)
            continue
        if line.startswith("### "):
            paragraph = document.add_paragraph(style="Heading 2")
            add_inline_markup(paragraph, line[4:])
        elif line.startswith("## "):
            paragraph = document.add_paragraph(style="Heading 1")
            add_inline_markup(paragraph, line[3:])
        elif line.startswith("# "):
            paragraph = document.add_paragraph(style="Heading 1")
            add_inline_markup(paragraph, line[2:])
        elif re.match(r"^\d+\.\s", line):
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Cm(0.65)
            paragraph.paragraph_format.first_line_indent = Cm(-0.45)
            paragraph.paragraph_format.space_after = Pt(3)
            add_inline_markup(paragraph, line)
        elif line.startswith("- "):
            paragraph = document.add_paragraph(style="List Bullet")
            add_inline_markup(paragraph, line[2:])
        elif line.startswith("**[[Insert Figure 4") and figure_path.exists():
            caption_path = figure_path.with_name("reconstruction_examples_caption.txt")
            caption = (
                caption_path.read_text(encoding="utf-8").strip()
                if caption_path.exists()
                else "Representative held-out point-cloud inputs and deterministic MeshVAE reconstructions."
            )
            insert_figure(document, figure_path, caption)
            inserted_figure = True
        elif inserted_figure and line.startswith(
            "Representative held-out point-cloud inputs and deterministic MeshVAE reconstructions illustrate"
        ):
            pass
        elif line.startswith("**Status:"):
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.space_after = Pt(10)
            add_inline_markup(paragraph, line)
        else:
            paragraph = document.add_paragraph()
            add_inline_markup(paragraph, line)
        index += 1

    if not inserted_figure and figure_path.exists():
        document.add_section(WD_SECTION.NEW_PAGE)
        heading = document.add_paragraph("Model generated body examples", style="Heading 1")
        keep_with_next(heading)
        insert_figure(document, figure_path, "Representative held-out point-cloud inputs and deterministic MeshVAE reconstructions.")

    core = document.core_properties
    core.title = title_text
    core.subject = "REACT-Q working manuscript"
    core.keywords = "REACT-Q, MeshVAE, 3D body scan, atherosclerosis"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path)
    print(f"[OUT] {output_path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", type=Path, default=Path("paper_analysis/paper_draft.md"))
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("paper_analysis/figures/reconstruction_examples.png"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_analysis/REACT_Q_manuscript_working_draft.docx"),
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    build_document(args.markdown, args.figure, args.output)
