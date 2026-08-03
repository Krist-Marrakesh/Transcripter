"""Склонение числительных в сообщениях пользователю."""

from __future__ import annotations

import pytest

from transcriber.text import plural

FORMS = ("сегмент", "сегмента", "сегментов")


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (1, "сегмент"),
        (2, "сегмента"),
        (4, "сегмента"),
        (5, "сегментов"),
        (0, "сегментов"),
        # 11–14 — исключение: «одиннадцать сегментов», а не «сегмент».
        (11, "сегментов"),
        (12, "сегментов"),
        (14, "сегментов"),
        (21, "сегмент"),
        (22, "сегмента"),
        (25, "сегментов"),
        (101, "сегмент"),
        (111, "сегментов"),
    ],
)
def test_plural_forms(count, expected):
    assert plural(count, *FORMS) == expected
