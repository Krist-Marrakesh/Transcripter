"""Small text helpers for user-facing messages."""

from __future__ import annotations


def plural(count: int, one: str, few: str, many: str) -> str:
    """Russian noun form after a numeral.

    Russian picks between three forms, so a plain "s" will not do — this is used
    where counts are printed in the CLI.

    >>> plural(1, "сегмент", "сегмента", "сегментов")
    'сегмент'
    >>> plural(22, "сегмент", "сегмента", "сегментов")
    'сегмента'
    >>> plural(15, "сегмент", "сегмента", "сегментов")
    'сегментов'
    """
    tens = count % 100
    if 11 <= tens <= 14:
        return many

    units = count % 10
    if units == 1:
        return one
    if 2 <= units <= 4:
        return few
    return many
