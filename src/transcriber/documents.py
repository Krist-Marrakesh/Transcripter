"""Export to PDF and DOCX.

Apart from `export`: there the functions are pure and hand back a string, while
these are binary formats that can only write a file. Putting both in one module
would mean making the signatures incompatible for the sake of a shared name.

Both dependencies are optional — without them the text formats remain.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .export import clock, formula_lines, group_into_lines
from .models import Transcript


def font_candidates() -> tuple[tuple[str, Path], ...]:
    """Serif faces with Cyrillic, best first.

    Vera, which ships with reportlab, has no Cyrillic at all — DejaVu was created
    to close exactly that gap. The order puts what matches the window's typography
    first and dependable fallbacks after.

    Windows is asked where it was installed rather than told: the drive is C on
    almost every machine and on the rest this is the difference between a PDF and
    an error naming a folder that was never there.
    """
    windows = Path(os.environ.get("WINDIR") or "C:/Windows") / "Fonts"
    return (
        ("Georgia", Path("/System/Library/Fonts/Supplemental/Georgia.ttf")),
        ("TimesNewRoman", Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")),
        ("Georgia", windows / "georgia.ttf"),
        ("TimesNewRoman", windows / "times.ttf"),
        ("DejaVuSerif", Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")),
        ("ArialUnicode", Path("/Library/Fonts/Arial Unicode.ttf")),
    )


def find_font() -> tuple[str, Path]:
    """The first available face that has Cyrillic in it."""
    candidates = font_candidates()
    for name, path in candidates:
        if path.exists():
            return name, path
    raise RuntimeError(
        "no font with Cyrillic was found for the PDF. Paths tried:\n  "
        + "\n  ".join(str(path) for _, path in candidates)
    )


def write_pdf(transcript: Transcript, path: Path, *, title: str | None = None) -> Path:
    """Lays the transcript into a PDF the way the window shows it."""
    # A document is read by a person, so the people are what it names. Labels
    # nobody identified come through unchanged.
    transcript = transcript.as_named()
    try:
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError as exc:  # pragma: no cover — зависит от установки
        raise ImportError('PDF недоступен: uv pip install -e ".[documents]"') from exc

    font, font_path = find_font()
    if font not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(font, str(font_path)))

    body = ParagraphStyle(
        "body",
        fontName=font,
        fontSize=11,
        leading=16.5,
        alignment=TA_LEFT,
        spaceAfter=7,
    )
    head = ParagraphStyle("head", parent=body, fontSize=17, leading=22, spaceAfter=4)
    meta = ParagraphStyle("meta", parent=body, fontSize=9, textColor="#7d6f58", spaceAfter=14)
    stamp = ParagraphStyle("stamp", parent=body, fontSize=8.5, textColor="#9a5b1c", spaceAfter=1)
    # Формула — исходник LaTeX, а не набранная запись, поэтому моноширинным и с
    # отступом: видно, что это приписка к сказанному, а не часть речи.
    written = ParagraphStyle(
        "formula", parent=body, fontName="Courier", fontSize=9.5, leftIndent=14, spaceAfter=3
    )

    document = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        topMargin=22 * mm,
        bottomMargin=22 * mm,
        leftMargin=24 * mm,
        rightMargin=24 * mm,
        title=title or transcript.source,
    )

    flow = [
        Paragraph(_escape(title or Path(transcript.source).name), head),
        Paragraph(_meta_line(transcript), meta),
    ]
    for speaker, batch in group_into_lines(transcript.segments):
        text = " ".join(s.text.strip() for s in batch if s.text.strip())
        if not text:
            continue
        label = clock(batch[0].start)
        if speaker:
            label = f"{label} · {speaker}"
        flow.append(Paragraph(_escape(label), stamp))
        flow.append(Paragraph(_escape(text), body))
        for line in formula_lines(batch):
            flow.append(_formula_flowable(line, written))

    flow.append(Spacer(1, 6))
    document.build(flow)
    return path


def _formula_flowable(latex: str, fallback_style):
    """Формула для PDF: набранной картинкой, а если не собралась — исходником.

    Не собирается закономерно: mathtext знает подмножество LaTeX, и матрицы в
    него не входят. Расшифровку это сохранить не мешает — читателю достаётся
    запись как есть, ровно то, что было здесь до набора.
    """
    from io import BytesIO

    from reportlab.platypus import Image, KeepTogether, Paragraph, Spacer

    from .typeset import draw

    drawn = draw(latex)
    if drawn is None:
        return Paragraph(_escape(latex), fallback_style)
    image = Image(BytesIO(drawn.png), width=drawn.width, height=drawn.height)
    # Влево, как и абзац рядом: середина отрывала бы формулу от слов, под
    # которыми она стоит.
    image.hAlign = "LEFT"
    # Картинка сама по себе стоит вплотную к следующей строке — у неё нет
    # межабзацного отступа, который есть у текста, и метка времени следующей
    # реплики прилипала к формуле.
    return KeepTogether([image, Spacer(1, 7)])


def write_docx(transcript: Transcript, path: Path, *, title: str | None = None) -> Path:
    """The same in DOCX — the format for anyone who will edit the text further."""
    transcript = transcript.as_named()
    try:
        from io import BytesIO

        from docx import Document
        from docx.shared import Pt, RGBColor

        from .typeset import draw
    except ImportError as exc:  # pragma: no cover — зависит от установки
        raise ImportError('DOCX недоступен: uv pip install -e ".[documents]"') from exc

    document = Document()
    normal = document.styles["Normal"]
    normal.font.name = "Georgia"
    normal.font.size = Pt(11)

    document.add_heading(title or Path(transcript.source).name, level=1)
    note = document.add_paragraph()
    run = note.add_run(_plain_meta(transcript))
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x7D, 0x6F, 0x58)

    for speaker, batch in group_into_lines(transcript.segments):
        text = " ".join(s.text.strip() for s in batch if s.text.strip())
        if not text:
            continue

        label = clock(batch[0].start)
        if speaker:
            label = f"{label} · {speaker}"
        stamp = document.add_paragraph()
        stamp_run = stamp.add_run(label)
        stamp_run.font.size = Pt(8.5)
        stamp_run.font.color.rgb = RGBColor(0x9A, 0x5B, 0x1C)
        stamp.paragraph_format.space_after = Pt(1)

        document.add_paragraph(text)
        for line in formula_lines(batch):
            written = document.add_paragraph()
            written.paragraph_format.left_indent = Pt(14)
            drawn = draw(line)
            if drawn is None:
                # Та же причина, что и в PDF: mathtext знает не весь LaTeX, и
                # запись как есть лучше пропущенной формулы.
                run = written.add_run(line)
                run.font.name = "Courier New"
                run.font.size = Pt(9.5)
            else:
                written.add_run().add_picture(BytesIO(drawn.png), width=Pt(drawn.width))

    document.save(str(path))
    return path


def _meta_line(transcript: Transcript) -> str:
    parts = [f"Language: {transcript.language}", clock(transcript.duration), transcript.asr_model]
    if transcript.speakers:
        parts.insert(1, ", ".join(transcript.speakers))
    return " · ".join(parts)


def _plain_meta(transcript: Transcript) -> str:
    return _meta_line(transcript)


def _escape(text: str) -> str:
    """reportlab reads a paragraph as small markup, so angle brackets are escaped."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def available() -> dict[str, bool]:
    """Which binary formats can actually be written in this environment."""
    return {
        "pdf": _importable("reportlab") and _has_font(),
        "docx": _importable("docx"),
    }


def _importable(module: str) -> bool:
    from importlib.util import find_spec

    try:
        return find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _has_font() -> bool:
    try:
        find_font()
    except RuntimeError:
        return False
    return True


if __name__ == "__main__":  # pragma: no cover — ручная проверка
    print("font:", find_font(), file=sys.stderr)
    print("available:", available(), file=sys.stderr)
