"""A content-addressed artifact cache.

The point: ASR costs minutes while translation and summarising cost seconds.
Without a cache, changing the summary prompt would mean transcribing the whole
recording again.

Key = fingerprint of the source file plus the parameters of the step. Renaming
the file does not change the key; changing the model or the language does.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

_HASH_CHUNK = 1 << 20  # 1 MiB: sha256 is hardware-backed on Apple Silicon, the disk is the limit

T = TypeVar("T", bound=BaseModel)


@lru_cache(maxsize=256)
def _hash_file(path: str, _size: int, _mtime_ns: int) -> str:
    """Streaming sha256. Size and mtime take part only in the lru_cache key."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: Path) -> str:
    """Fingerprint of the file contents.

    Memoised by (path, size, mtime), so repeated lookups of the same file within
    one run are free, while editing the file invalidates the fingerprint.
    """
    stat = path.stat()
    return _hash_file(str(path), stat.st_size, stat.st_mtime_ns)


def stable_key(*parts: object, **params: object) -> str:
    """A deterministic key built from arbitrary parameters.

    Key order is normalised, so the same set of parameters always yields the same
    string no matter how it was assembled.
    """
    payload = json.dumps([parts, params], sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@contextmanager
def building(target: Path) -> Iterator[Path]:
    """Yields a path to fill; it becomes `target` only once the block finishes.

    A cache entry must not exist until it is whole. Written straight to the final
    name, a file interrupted halfway stays there and every later run accepts it as
    ready — a recording silently short of itself, with nothing on screen to say so.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    # The mark goes before the extension, not after it: ffmpeg picks the container
    # by the last suffix, and given `.part` it writes nothing at all — while still
    # exiting with code zero, so the failure would arrive as a missing file much
    # further on.
    partial = target.with_name(f"{target.stem}.part{target.suffix}")
    try:
        yield partial
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


class ArtifactCache:
    """A file cache of pydantic artifacts, laid out by step namespaces."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, namespace: str, key: str, suffix: str = ".json") -> Path:
        return self.root / namespace / f"{key}{suffix}"

    def load(self, namespace: str, key: str, model: type[T]) -> T | None:
        """Reads an artifact or returns None. A corrupt file counts as a miss."""
        target = self.path(namespace, key)
        if not target.exists():
            return None
        try:
            return model.model_validate_json(target.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            # An interrupted write or a schema that no longer matches after a model
            # change — recomputing is cheaper than investigating.
            return None

    def store(self, namespace: str, key: str, artifact: BaseModel) -> Path:
        """Atomic write: into a temporary file first, then a swap."""
        target = self.path(namespace, key)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
        os.replace(tmp, target)
        return target

    def reserve(self, namespace: str, key: str, suffix: str) -> Path:
        """A path for an unstructured artifact, such as the normalised WAV."""
        target = self.path(namespace, key, suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target
