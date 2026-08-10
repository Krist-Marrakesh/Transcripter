"""Командный интерфейс."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import device
from .app import state
from .config import LLM_MODELS, WHISPER_MODELS, load_settings
from .export import FORMATS, render, summary_to_markdown
from .models import Transcript
from .pipeline import Pipeline
from .report import Report
from .text import plural

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Локальный транскрибатор аудио, видео и ссылок на видео.",
)
console = Console()
errors = Console(stderr=True)


def _notify(message: str) -> None:
    console.print(f"[dim]·[/dim] {message}", highlight=False)


def _safe_stem(name: str) -> str:
    """Имя файла без символов, ломающих путь."""
    cleaned = "".join(ch if ch.isalnum() or ch in " -_.," else "_" for ch in name).strip()
    return (cleaned[:120] or "transcript").rstrip(". ")


def _write(transcript: Transcript, formats: list[str], directory: Path, stem: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = directory / f"{stem}.{fmt}"
        path.write_text(render(transcript, fmt), encoding="utf-8")
        written.append(path)
    return written


@app.command()
def transcribe(
    target: Annotated[
        str, typer.Argument(help="путь к файлу или ссылка на видео: YouTube, VK Video и другие")
    ],
    language: Annotated[
        str | None, typer.Option("--lang", "-l", help="ru, en… по умолчанию определяется само")
    ] = None,
    diarize: Annotated[
        bool, typer.Option("--diarize/--no-diarize", help="размечать, кто когда говорит")
    ] = False,
    speakers: Annotated[
        int | None,
        typer.Option(
            "--speakers",
            help="точное число говорящих: подсчёт — самая слабая часть разметки, "
            "и ошибается он в дробление (на записи троих находилось девять)",
        ),
    ] = None,
    min_speakers: Annotated[
        int | None, typer.Option("--min-speakers", help="нижняя граница числа говорящих")
    ] = None,
    max_speakers: Annotated[
        int | None, typer.Option("--max-speakers", help="верхняя граница числа говорящих")
    ] = None,
    translate_to: Annotated[
        str | None, typer.Option("--to", help="перевести транскрипт на язык (ru, en…)")
    ] = None,
    summary: Annotated[bool, typer.Option("--summary", help="сделать саммари через LLM")] = False,
    summary_lang: Annotated[
        str | None,
        typer.Option("--summary-lang", help="язык саммари, по умолчанию ru независимо от записи"),
    ] = None,
    formats: Annotated[
        list[str] | None,
        typer.Option("--format", "-f", help=f"можно несколько раз: {', '.join(FORMATS)}"),
    ] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="куда складывать")] = None,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help=f"ASR-модель: {', '.join(WHISPER_MODELS)}")
    ] = None,
    words: Annotated[bool, typer.Option("--words", help="таймкоды для каждого слова")] = False,
    no_vad: Annotated[bool, typer.Option("--no-vad", help="не вырезать тишину")] = False,
    prompt: Annotated[
        str | None, typer.Option("--prompt", help="подсказка декодеру: имена, термины")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="пересчитать, игнорируя кэш")] = False,
) -> None:
    """Распознать запись и, по желанию, перевести и сжать её."""
    settings = load_settings(
        asr_model=model,
        word_timestamps=words or None,
        initial_prompt=prompt,
        vad_enabled=False if no_vad else None,
        diarization_enabled=diarize or None,
        num_speakers=speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        output_dir=output,
        summary_language=summary_lang,
    )
    pipeline = Pipeline(settings, Report(say=_notify))
    chosen = formats or ["txt"]

    # Папка, выбранная мышью в окне, действует и в терминале: это одно
    # приложение, а не два. Явный `--output` перекрывает её — иначе флаг ничего
    # не значил бы. Различить «флаг» и «умолчание» можно только здесь: в
    # `Settings` они уже слиты в одно поле.
    target_dir = output if output else state.output_dir(settings.output_dir)

    try:
        source, transcript = pipeline.run(target, language=language, diarize=diarize, force=force)
    except Exception as exc:
        errors.print(f"[red]Ошибка:[/red] {exc}")
        raise typer.Exit(1) from exc

    stem = _safe_stem(source.title)
    written = _write(transcript, chosen, target_dir, stem)

    if translate_to:
        translation = pipeline.translate(transcript, target_language=translate_to, force=force)
        translated = transcript.model_copy(
            update={"segments": translation.segments, "language": translation.target_language}
        )
        written += _write(translated, chosen, target_dir, f"{stem}.{translate_to}")

    if summary:
        result = pipeline.summarize(transcript, language=settings.summary_language, force=force)
        path = target_dir / f"{stem}.summary.md"
        path.write_text(summary_to_markdown(result, title=source.title), encoding="utf-8")
        written.append(path)

        console.print()
        console.print(f"[bold]{result.overview}[/bold]")
        for point in result.key_points:
            console.print(f"  • {point}")

    count = len(transcript.segments)
    speakers = len(transcript.speakers)
    console.print()
    console.print(
        f"[green]Готово[/green] · {count} {plural(count, 'сегмент', 'сегмента', 'сегментов')} · "
        f"язык [bold]{transcript.language}[/bold]"
        + (f" · {speakers} {plural(speakers, 'спикер', 'спикера', 'спикеров')}" if speakers else "")
    )
    for path in written:
        console.print(f"  [cyan]{path}[/cyan]")


@app.command()
def info() -> None:
    """Проверить окружение: ffmpeg, Metal, доступность моделей."""
    settings = load_settings()
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("ffmpeg", "[green]есть[/green]" if shutil.which("ffmpeg") else "[red]нет[/red]")

    # Главная строка вывода: на чём в действительности пойдёт счёт. Молчаливый
    # откат на процессор — самая частая причина вопроса «почему так медленно».
    resolved = device.detect()
    colour = "green" if resolved != "cpu" else "yellow"
    table.add_row("Устройство", f"[{colour}]{device.describe()}[/{colour}]")

    try:
        import mlx.core as mx

        table.add_row("mlx", f"[green]{mx.default_device()}[/green]")
    except ImportError:
        # Вне Apple Silicon его и не должно быть — это норма, а не поломка.
        table.add_row("mlx", "[dim]недоступен на этой платформе[/dim]")

    table.add_row("", "")
    table.add_row("ASR-бэкенд", f"{settings.asr_backend} · {settings.asr_repo}")
    table.add_row("LLM-бэкенд", f"{settings.llm_backend} · {settings.llm_repo}")
    if settings.llm_backend == "openai":
        # Чужая машина — единственная часть, которой может не оказаться на
        # месте, поэтому её состояние проверяется прямо здесь.
        from .nlp.llm import check_remote

        failure = check_remote(settings.llm_base_url, api_key=settings.llm_api_key)
        state = "[green]отвечает[/green]" if failure is None else f"[red]{failure}[/red]"
        table.add_row("Сервер LLM", f"{settings.llm_base_url} · {state}")
        if settings.llm_api_key:
            # Значение не печатаем: вывод `info` попадает в чужие issue.
            table.add_row("Ключ сервера", "[green]задан[/green]")
    table.add_row("Диаризация", settings.diarization_model)
    table.add_row(
        "HF-токен",
        "[green]задан[/green]"
        if settings.hf_token
        else "[yellow]нет (нужен для спикеров)[/yellow]",
    )
    table.add_row("Кэш", str(settings.cache_dir))

    console.print(table)


@app.command(name="app")
def desktop(
    debug: Annotated[bool, typer.Option("--debug", help="открыть инспектор webview")] = False,
) -> None:
    """Открыть десктопное окно."""
    try:
        from .app import run
    except ImportError as exc:
        errors.print('[red]окно не установлено:[/red] uv pip install -e ".[app]"')
        raise typer.Exit(1) from exc
    run(debug=debug)


@app.command()
def shortcut(
    to: Annotated[Path, typer.Option("--to", help="куда положить ярлык")] = Path("~/Desktop"),
) -> None:
    """Собрать ярлык приложения на рабочем столе."""
    from .app.shortcut import create

    try:
        bundle = create(to)
    except RuntimeError as exc:
        errors.print(f"[red]Ошибка:[/red] {exc}")
        raise typer.Exit(1) from exc

    # Сборка ярлыка — это и есть установка с точки зрения пользователя, поэтому
    # папка для транскриптов появляется здесь, а не после первого распознавания.
    folder = state.output_dir(load_settings().output_dir)
    created = state.ensure(folder)

    console.print(f"[green]Ярлык собран[/green] · [cyan]{bundle}[/cyan]")
    if created:
        console.print(f"[green]Папка для транскриптов[/green] · [cyan]{folder}[/cyan]")
    else:
        errors.print(f"[yellow]Не удалось создать папку[/yellow] {folder} — выбери другую в окне")
    console.print("[dim]Привязан к этому окружению: переедет проект — пересобери.[/dim]")


@app.command()
def models() -> None:
    """Показать короткие имена моделей."""
    settings = load_settings()

    asr = Table("ASR", "репозиторий", box=None, padding=(0, 2))
    for alias, repo in WHISPER_MODELS.items():
        asr.add_row(alias, repo)

    console.print(asr)
    console.print()

    # На удалённом сервере имена задаёт он сам, и наш реестр к ним отношения не
    # имеет — спрашиваем список у сервера, а не показываем таблицу прочерков.
    if settings.llm_backend == "openai":
        from .nlp.llm import RemoteLanguageModel

        remote = RemoteLanguageModel(
            settings.llm_repo, settings.llm_base_url, api_key=settings.llm_api_key
        )
        served = remote.available()
        llm = Table("LLM", f"на сервере · {settings.llm_base_url}", box=None, padding=(0, 2))
        for name in served or ["[dim]сервер не ответил[/dim]"]:
            llm.add_row("→" if name == settings.llm_repo else "", name)
        console.print(llm)
        return

    # У Ollama и mlx разные системы имён, поэтому показываем то, что реально
    # уйдёт в текущий бэкенд, а не оба варианта сразу.
    llm = Table("LLM", f"идентификатор · {settings.llm_backend}", "вес", box=None, padding=(0, 2))
    for alias, choice in LLM_MODELS.items():
        # Прочерк — модель под этот бэкенд не заведена: mlx-сборки существуют не
        # для всех моделей Ollama и наоборот.
        #
        # Вес показан рядом, потому что именно он решает, какая модель достанется
        # этой машине по умолчанию: без него выбор выглядит необъяснимым.
        llm.add_row(alias, choice.ids.get(settings.llm_backend, "—"), f"{choice.gigabytes:.1f} ГБ")

    console.print(llm)


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        errors.print("[yellow]прервано[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
