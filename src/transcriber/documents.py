"""Экспорт в PDF и DOCX.

Отдельно от `export`: там чистые функции, отдающие строку, а здесь бинарные
форматы, которые умеют только писать файл. Смешивать их в одном модуле значило
бы сделать сигнатуры несовместимыми ради общего имени.

Обе зависимости необязательные — без них остаются текстовые форматы.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .export import clock, group_by_speaker
from .models import Transcript

# Антиква с кириллицей. Vera из поставки reportlab её не содержит — DejaVu в своё
# время и создавали, чтобы закрыть этот пробел. Порядок: сначала то, что совпадает
# с оформлением окна, потом надёжные запасные варианты.
FONT_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("Georgia", "/System/Library/Fonts/Supplemental/Georgia.ttf"),
    ("TimesNewRoman", "/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
    ("Georgia", "C:/Windows/Fonts/georgia.ttf"),
    ("TimesNewRoman", "C:/Windows/Fonts/times.ttf"),
    ("DejaVuSerif", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    ("ArialUnicode", "/Library/Fonts/Arial Unicode.ttf"),
)


def find_font() -> tuple[str, Path]:
    """Первый доступный шрифт с кириллицей."""
    for name, path in FONT_CANDIDATES:
        candidate = Path(path)
        if candidate.exists():
            return name, candidate
    raise RuntimeError(
        "не найден шрифт с кириллицей для PDF. Проверенные пути:\n  "
        + "\n  ".join(path for _, path in FONT_CANDIDATES)
    )


def write_pdf(transcript: Transcript, path: Path, *, title: str | None = None) -> Path:
    """Складывает транскрипт в PDF, повторяя вид окна: антиква, поля, таймкоды."""
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
    for speaker, batch in group_by_speaker(transcript.segments):
        text = " ".join(s.text.strip() for s in batch if s.text.strip())
        if not text:
            continue
        label = clock(batch[0].start)
        if speaker:
            label = f"{label} · {speaker}"
        flow.append(Paragraph(_escape(label), stamp))
        flow.append(Paragraph(_escape(text), body))

    flow.append(Spacer(1, 6))
    document.build(flow)
    return path


def write_docx(transcript: Transcript, path: Path, *, title: str | None = None) -> Path:
    """То же самое в DOCX — формат для тех, кто правит текст дальше."""
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor
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

    for speaker, batch in group_by_speaker(transcript.segments):
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

    document.save(str(path))
    return path


def _meta_line(transcript: Transcript) -> str:
    parts = [f"Язык: {transcript.language}", clock(transcript.duration), transcript.asr_model]
    if transcript.speakers:
        parts.insert(1, ", ".join(transcript.speakers))
    return " · ".join(parts)


def _plain_meta(transcript: Transcript) -> str:
    return _meta_line(transcript)


def _escape(text: str) -> str:
    """reportlab читает абзац как мини-разметку, поэтому угловые скобки экранируем."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def available() -> dict[str, bool]:
    """Какие бинарные форматы реально можно записать в этом окружении."""
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
    print("шрифт:", find_font(), file=sys.stderr)
    print("доступно:", available(), file=sys.stderr)
