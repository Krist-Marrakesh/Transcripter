# transcript

A local transcriber for audio, video and YouTube links. Russian and English,
translation both ways, summaries. Everything runs on your own hardware — the
network is needed once, to fetch model weights.

Recordings and transcripts are never sent anywhere. The one request the
application makes on its own is a question to GitHub about newer versions; it
sends nothing and can be switched off with `TRANSCRIPT_UPDATE_CHECK=false`.

## Pipeline

```
file / video / YouTube
        ↓  yt-dlp
        ↓  ffmpeg → 16 kHz mono
        ↓  VAD (Silero) — silence cut out
        ↓  ASR (Whisper large-v3 on Metal)     ┐ expensive, minutes
        ↓  diarization (sherpa-onnx / pyannote)┘
   segments.json  ← cached by audio sha256
        ↓
   translation · summary · srt/vtt/txt/md/json   cheap, seconds
```

The key decision is that `segments.json` is persisted and keyed by audio content
plus the parameters of the step. Recognising an hour-long recording takes
minutes; translation and summarising take seconds. Change the summary prompt or
the target language and transcription is not repeated. Turn on diarization and
ASR is not recomputed either — it has a key of its own.

## Download (macOS, Apple Silicon)

A ready bundle is attached to every [release](https://github.com/Krist-Marrakesh/Transcripter/releases):
`Transcripter-<version>-macos.zip`, about 21 MB. Unpack it, move
`Транскрибатор.app` to Applications, and before the first launch remove the
quarantine mark:

```bash
xattr -dr com.apple.quarantine /Applications/Транскрибатор.app
```

That step is not optional. The bundle carries no developer signature — `spctl`
rejects it with `no usable signature` — so macOS refuses to open a copy that came
from a browser and will not say why in any useful way.

What happens on the first launch: the application asks permission, then builds
its own environment beside itself in `~/Library/Application Support/transcript`
— about 1.7 GB, five to ten minutes. It carries a wheel of the package and `uv`,
and installs a native Python for the machine it landed on — nothing is frozen for
the architecture the build happened to run on. The speech models are a separate
download, about 3 GB, and the window asks again before fetching them.

Nothing else has to be installed by hand, and nothing has to be agreed to. ffmpeg
and ffprobe arrive as a package of static builds; a system ffmpeg is still
preferred when there is one, since it is usually newer.

### Two ways to label speakers

Speaker labels work on a bare machine. The default backend is sherpa-onnx, whose
models are published openly — the window downloads them once, about 104 MB, and
nothing has to be arranged by hand.

pyannote is offered rather than required, because its models are gated: the API
reports `gated: auto`, so the terms have to be accepted on the model page and a
token created. No installer can sign an agreement on someone's behalf. The window
says so on the first run, links to both pages, and takes the token; declining is
a real answer, and entering a token later switches over at any time.

What the difference is, measured rather than guessed — five recordings, four and
a half hours, the two backends against each other:

| | pyannote | sherpa |
|---|---|---|
| three monologues | 1 speaker | 1 speaker, no divergence at all |
| 75-minute seminar | 3 speakers | 2 speakers, 1.8% divergence |
| 108-minute lecture | 4 speakers | 2 speakers, 2.8% divergence |
| speed | 1.9 min on Metal | 4.1 min on four CPU cores |

So sherpa invents no speakers who are not there — the failure everyone fears from
clustering — and misses rare brief ones who are. A question from the room gets
attributed to the lecturer. The window says that above a transcript it labelled,
rather than leaving it here.

The clustering threshold is pinned at 1.3 in `diarize/sherpa_backend.py`. The
library's own default of 0.5 turns one lecturer into sixty-four speakers, so the
number has the measurement written beside it.

A token given in the window is kept in `~/.config/transcript/app.json`, next to
the other choices made with the mouse. For the CLI it can also be set the usual
way, in `~/.config/transcript/.env`:

```
TRANSCRIPT_HF_TOKEN=hf_...
```

That path rather than a `.env` beside the application: a window started from
Finder does not choose its own working directory, so a relative one is read from
wherever the system happened to put it. A checkout still overrides it with its
own `.env`.

### Updates

The application asks GitHub about newer versions when the window opens, and says
so in the header when there is one. Installing replaces the package rather than
the bundle — a quarter of a megabyte instead of twenty-one, because everything of
ours lives in one wheel while the launcher and `uv` around it barely change. The
window then asks for a restart; nothing is replaced underneath a running process.

That is why a release carries two assets. The zip is for arriving; the bare
`.whl` beside it is what an installed copy fetches for itself.

The launcher knows not to undo this. It records the version and the hash of what
it installed, and when the environment holds something newer than the wheel in
the bundle — which is exactly what a self-update leaves behind — it keeps its
hands off. Otherwise every launch would quietly roll the update back.

To build both assets from a checkout:

```bash
.venv/bin/python tools/release.py
```

## Install from source

macOS on Apple Silicon is the platform this is built and tested on: MLX and
Metal, and every measurement in this file was taken there.

> **Windows is not finished.** The code for it is written and some of it is
> covered by tests, but no part of it has ever run on Windows — see the section
> at the end for what exists and what remains unverified. Treat it as work in
> progress rather than as a supported platform.

**macOS (Apple Silicon)**

```bash
brew install ffmpeg uv
uv venv --python 3.13
uv pip install -e ".[dev]"
```

**Windows with NVIDIA**

```powershell
winget install Gyan.FFmpeg astral-sh.uv
uv venv --python 3.13
uv pip install -e ".[dev]"
# CUDA torch lives on a separate index: PyPI installs the CPU build
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Translation and summaries need [Ollama](https://ollama.com/download) there — it
comes with its own installer and claims the GPU by itself:

```powershell
ollama pull qwen3:8b
```

Check the environment; the first line names the device the work will actually
run on:

```bash
transcript info
```

It shows up during a run as well: `cuda · float16` or `cpu · int8` is printed
before recognition starts.

> **Untested on hardware.** Everything in this section was written on macOS and
> never run on Windows with NVIDIA. Three places worth re-checking:
>
> 1. **The CUDA channel version.** `cu126` matches the RTX 40 series, but the
>    current channel keeps moving. Check with `pip index versions torch`.
> 2. **Torch is downloaded twice.** The main install pulls the CPU build from
>    PyPI (~200 MB) and the next command replaces it. A `[tool.uv.sources]` entry
>    pointing at the separate index would fix this — not done, because it cannot
>    be verified from here.
> 3. **cuBLAS and cuDNN.** CTranslate2 looks for them in `PATH`. If
>    `transcript info` says `cpu` while the card is alive, the cause is either
>    here or in the CPU torch build from point 2.

### What Windows would need — written, not verified

Five things behave differently enough there to be written separately, and the
reasons are worth naming because none of them fail loudly. All of it is code
that has never executed on Windows; three of the five are covered by tests that
fake the platform, and the rest cannot be checked from a Mac at all.

**The single-instance lock does not ask with a signal.** On POSIX `os.kill(pid, 0)`
sends nothing and only asks whether there is someone to send to. On Windows
CPython implements `os.kill` through `TerminateProcess` for every signal but the
two console ones — the check would have killed the copy it was looking for, so a
second launch would close the first window instead of raising it. `app/process.py`
keeps the two ways apart: process rights and an exit code on one side, a signal
and `ps` on the other.

**Console children flash a window.** A shortcut points at `pythonw.exe`, a program
without a console; every console child it starts — ffmpeg, ffprobe, the weight
downloader — is given a console of its own and it is shown. Redirected streams do
not help: the window comes from the subsystem the child was built for.
`subproc.quiet_flags()` is the one place that knows this.

**Folders.** Windows has no split into config, data and cache — everything an
application owns lives under `%LOCALAPPDATA%\transcript`, and the separation is
made by subfolders. The Documents folder is read from the registry rather than
assumed to be `~/Documents`: OneDrive relocates it, and a transcript written to
the old path lands where nobody looks.

**Virtual environments are laid out differently.** The interpreter is
`Scripts/python.exe`, not `bin/python`. The weight downloader runs as a separate
process and has to find an interpreter that actually has our dependencies.

**Diarization on the GPU no longer fails quietly.** A model that cannot be moved
to the card falls back to the processor, which is correct but costs tens of
minutes on a long recording. It now says so.

Of these, the folder layout, the venv layout and the console flag are covered by
tests that run anywhere — they read `sys.platform` when called, so faking the
system exercises the real branch. The rest (`ctypes` calls into the Win32 API,
building a `.lnk` through PowerShell) can only be checked on Windows itself.

## The window

```bash
uv pip install -e ".[app]"
transcript app
```

A desktop utility: drop a file in or paste a YouTube link. The transcript is
clickable — clicking a line seeks the recording, and the current line is
highlighted.

The design is a reading mode: warm paper `#fbf4e0` instead of white, not a
single blue pixel. Speech is set in a serif, service labels in a sans — lecture
text is read in sequence, labels are caught at a glance. The system theme is
deliberately ignored; night mode is a button in the header and is remembered.

On the first launch a panel at the top lists the weights that are missing and
downloads them on a button press, with a bar counting real bytes on disk. The
application downloads nothing on its own: the recognition model is 2.9 GB, the
translation model is 19 GB, and spending that traffic is a person's decision,
not a program's. While the weights are missing, "Transcribe" refuses to start
and explains what is absent — otherwise the first press would disappear into a
silent multi-minute download.

A download can be stopped, but it cannot be continued from where it broke, which
is why the button says "Download again". The library itself supports resuming —
it sends `Range` from the size of the started file — but with Xet enabled (the
`hf_xet` package, installed by default) every attempt opens a temporary file of
its own. One interrupted download accumulated four pieces weighing more than the
model itself, so abandoned ones are removed before a start and only the freshest
counts towards progress. If resuming matters more than speed, Xet is switched
off with `HF_HUB_DISABLE_XET=1`.

A frozen connection is noticed without help: if less than a megabyte arrives in
three quarters of a minute, the downloader is restarted — up to five times, then
a refusal with an explanation. Waiting is pointless there, since the process
stays alive and looks perfectly healthy.

A desktop shortcut:

```bash
transcript shortcut
```

The bundle is thin — it holds a launch of the current interpreter rather than a
copy of the environment. Packing everything with PyInstaller makes no sense:
torch and the weights would move inside, gigabytes for the sake of a double
click. The price is that the shortcut is tied to the project directory — move
the project and rebuild it.

If the window does not open, look here — Finder gives no terminal, so the output
goes to a file:

```bash
cat ~/Library/Logs/Транскрибатор.log
```

The architecture is set explicitly in the launcher (`arch -arm64`). Python from
python.org is a universal binary, and for a script bundle LaunchServices picks
the x86_64 slice even though the compiled packages are built for arm64. The
import failed before the window appeared: the icon bounced once and went out.

The window is a system webview (WKWebView on macOS, WebView2 on Windows) rather
than a browser of its own: about a megabyte against Electron's hundred and fifty.
Audio reaches the window over HTTP from the loopback address — WKWebView does not
let `<audio>` read files over `file://` outside the page directory. The server
supports Range requests; without them seeking an hour-long lecture would wait for
all ~115 MB to load.

## From the terminal

```bash
# a file
transcript transcribe recording.m4a --lang ru

# video — the track is extracted automatically
transcript transcribe lecture.mp4 -f srt -f md

# YouTube
transcript transcribe "https://youtube.com/watch?v=..." --lang en

# who speaks when
transcript transcribe interview.wav --diarize --speakers 2

# translation and summary
transcript transcribe meeting.mp3 --to ru --summary
```

Useful flags:

| Flag | What it does |
|---|---|
| `--lang ru` | language of the recording; detected automatically without it |
| `-f srt -f md` | output formats, several allowed |
| `--diarize` | label speakers (needs an HF token) |
| `--speakers 2` | a known speaker count — noticeably improves quality |
| `--min-speakers 2 --max-speakers 5` | bounds when the exact count is unknown |
| `--to en` | translate the transcript |
| `--summary` | summary through the local LLM |
| `--summary-lang ru` | summary language; `ru` by default regardless of the recording |
| `--prompt "..."` | decoder hint for the opening seconds — see the note below |
| `--words` | timestamps for every word |
| `--no-vad` | keep silence |
| `--force` | recompute, ignoring the cache |

## Configuration

Copy `.env.example` to `.env`. Diarization requires a HuggingFace token and
acceptance of the model's terms — it is gated:

```
TRANSCRIPT_HF_TOKEN=hf_...
```

Models are switched by short names (`transcript models` prints the list):

```bash
transcript transcribe file.mp3 -m turbo        # faster, worse on Russian
TRANSCRIPT_LLM_MODEL=qwen3-8b transcript ...   # if 35B is too heavy
```

The default LLM is chosen per platform. On Apple Silicon it is `qwen3.6-35b` — a
MoE with 35B parameters of which ~3B are active: it occupies 19 GB of shared
memory, loads in 6 seconds and translates a paragraph in about one. On NVIDIA the
default differs — `qwen3-8b`: the model lives in VRAM, and partial offloading to
RAM costs more than the extra size gives.

## An LLM on another machine

Recognition and diarization always run on your own hardware, but translation and
summarising can move to a server — when local memory cannot hold a larger model,
say, or when several people share one. The application stays local either way:
only the transcript text leaves the machine, the audio never does.

The `openai` backend is not OpenAI's cloud but its protocol. vLLM, llama.cpp
server, LM Studio, TGI and Ollama itself all speak it, so one setting fits any of
them:

```bash
# on the server — any of these
vllm serve Qwen/Qwen3.6-35B-A3B --port 8000
llama-server -m qwen3.6-35b.gguf --port 8000
ollama serve                      # its OpenAI-compatible address is :11434/v1
```

```bash
# on your own machine
TRANSCRIPT_LLM_BACKEND=openai
TRANSCRIPT_LLM_BASE_URL=http://192.168.1.50:8000/v1
TRANSCRIPT_LLM_MODEL=Qwen/Qwen3.6-35B-A3B
```

The model name is the server's to define, not our registry's: with this backend
`transcript models` asks the server for its list, and `transcript info` shows
whether it answers. If the model is named wrong, the refusal arrives together
with what the server actually serves — no need to look for the cause on your own
machine.

One subtlety that otherwise returns translations full of noise. A reasoning model
keeps its train of thought to itself only when asked to, and the chat template is
applied on the server, without us — a local `enable_thinking=False` never gets
there. So the request carries `chat_template_kwargs`; a server that does not know
the field refuses, and then we repeat the request without it and strip the
reasoning on our side.

## Layout

| Module | Responsible for |
|---|---|
| `ingest/` | ffmpeg and yt-dlp, normalising to 16 kHz mono |
| `vad.py` | speech detection, time-axis compression and the reverse mapping |
| `asr/` | interchangeable Whisper backends behind one protocol |
| `diarize/` | two labelling backends and stitching speakers onto segments |
| `nlp/` | batched translation, map-reduce summaries |
| `export.py` | pure rendering functions for srt/vtt/txt/md |
| `pipeline.py` | the only place where steps are joined and the cache kicks in |
| `cache.py` | file cache of artifacts by content hash |
| `device.py` | the only place where CUDA / Metal / CPU is chosen |
| `weights.py` | accounting of weights in the HuggingFace cache and their download |
| `app/` | the desktop window: bridge into the pipeline, audio serving, markup |

## Code style

`ruff` decides the mechanical part and there is nothing to argue about with it:

```bash
.venv/bin/python -m ruff check src tests tools
.venv/bin/python -m ruff format src tests tools
```

Four indentation spaces, double quotes, a hundred columns. The quotes and the
width are not preferences — the formatter rewrites anything else on the next run,
so a file that disagrees with it disagrees for exactly as long as nobody runs it.

The rest is not mechanical.

**Names say what the thing is, in as few words as carry it.** `sweep`, `advance`,
`quiet_flags`, `truncation_warning` — a reader who has never opened the file
should be able to guess what comes back. Length is not the goal: `_relabel` is
short because renaming is all it does, while `truncation_warning` is longer
because "warning" and "truncation" are both load-bearing. What a name must never
do is describe the implementation — `run_clustering_loop` ages the moment the
loop goes.

**Imports at the top of the file.** One exception, and it is measured rather than
assumed: a heavy dependency that is not needed on every run is imported where it
is used, with a line saying what it costs.

```
package        cost
pyannote.audio  2.9 s
mlx_lm          2.1 s
mlx_whisper     0.7 s
torch           0.5 s
```

Together they are 2.7 seconds — the window currently opens in 0.11. And most of
that is paid for nothing: `pyannote.audio` is only reached by someone who turned
speaker labels on *and* chose that backend, `mlx_lm` only by someone who asked
for a translation or a summary. Someone who recognised one file and closed the
window would have waited for both.

So the rule for an import inside a function is that the comment above it says why:

```python
# A lazy import: pyannote pulls in torch and lightning, seconds at startup.
from pyannote.audio import Pipeline
```

Without that line it is indistinguishable from an import somebody forgot to move.

**Comments explain the reason, not the mechanics.** What the code does is legible
from the code. Why it does that rather than the obvious thing is legible from
nowhere else — and this project has a lot of those, most of them found by
measuring something that turned out untrue.

## Things that are easy to trip over

**Whisper only translates into English.** `task=translate` means X→EN, and that
is a limit of the training data rather than the architecture. EN→RU is therefore
done by the local LLM, not by Whisper.

**VAD is a necessity, not an option.** Whisper is autoregressive: on silence it
has nothing to lean on, yet it must emit a token. Hence phantom sign-offs and
loops. `--no-vad` is worth enabling only on a recording known to be clean.

**mlx-whisper has no beam search** — only a greedy decoder is implemented. On
this backend `beam_size` is translated into `best_of` (sampling several
trajectories during the temperature fallback). Real beam search exists in
faster-whisper, but on macOS it runs on the CPU:
`uv pip install -e ".[faster]"` and `TRANSCRIPT_ASR_BACKEND=faster`.

**Do not set `--lang` unless you are sure of the language.** The language token
is a condition for generation, not a check. The pair `<|ru|><|transcribe|>` only
ever occurred with Russian audio during training, so on an English recording the
model lands in a mode it was never taught and improvises. In a measured run
`--lang ru` on an English dialogue produced smooth Russian in which "before
Friday afternoon" turned into "by tomorrow". Auto-detection on the same file got
it right. The failure is silent: the output is coherent text with substituted
facts.

**`--prompt` is a hint for the opening seconds, not a glossary — and on a long
recording it hurts.** The prompt is not applied to every window; it is the
beginning of a sliding context. Each 30-second window receives the prompt plus
everything already recognised (`transcribe.py:296`), and the decoder truncates
that to the last 223 tokens (`decoding.py:502`, `n_text_ctx // 2 - 1`). A minute
of speech exceeds that, so the hint is displaced within a window or two, and the
first temperature fallback above 0.5 drops the context entirely.

Measured on a 110-minute lecture, three runs, divergence from a reference
transcript:

| section | no prompt | list of terms | natural phrase |
|---|---|---|---|
| first 5 min | **0.9%** | 1.8% | 2.4% |
| 5–15 min | 3.8% | **3.1%** | 3.6% |
| whole lecture | **3.8%** | 4.0% | 4.1% |

No prompt wins overall, and the wording does not rescue it: a natural phrase did
slightly worse than a comma-separated list. Worse still for the intended purpose,
the word "Euler" was recognised twice without a prompt and **zero times with
either one** — the hint damaged the very term it was meant to fix. Whisper has no
glossary mechanism; if terminology matters, a replacement dictionary in
post-processing is the honest tool. Keep `--prompt` for short recordings, where
the opening seconds are the whole recording.

**Translation is done by the LLM, not by Whisper.** So "an English lecture → a
Russian summary" is `--to ru`, and the summary language is set separately
(`--summary-lang`, `ru` by default): it need not match the recording.

**`turbo` is worse on Russian.** The distilled model is biased towards English
and loses noticeably on names and numbers. Keep it as a quick-draft mode.

**A reasoning model answers with its train of thought by default.** The Qwen3.6
chat template appends an opening `<think>` to the end of the prompt, so the model
carries on from the reasoning and only its tail plus the closing tag remain in the
answer — stripping a paired `<think>…</think>` does not match at all. Translation
and summarising do not need the reasoning, and the adapter turns it off with the
regular `enable_thinking=False`: on a single paragraph that is 1.2 seconds instead
of 5.1 and a clean answer instead of two and a half kilobytes of deliberation.
Templates of non-reasoning models do not know the flag and silently ignore it.

**Qwen3-ASR does not replace Whisper: it has no timestamps.** The `qwen3-asr-mlx`
package returns a `TranscriptionResult` with `text`, `language` and `duration` —
solid text and not a word about time. Without per-segment timestamps there is no
clickable transcript, no srt/vtt and no speaker stitching, which is everything
segments exist for. On a test fragment it did not beat `whisper-large-v3` on speed
either, so the branch is closed.

## Tests

```bash
pytest -q
ruff check src/
```

Covered is the pure logic that needs no models: reverse timestamp mapping after
VAD, speaker stitching by overlap, format rendering, parsing of LLM answers.
