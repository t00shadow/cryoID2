from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Mapping, Sequence, TypeVar

from .models import SequenceMode


_INTEGER_TEXT = re.compile(r"^\+?\d+$")
_RankedRow = TypeVar("_RankedRow", bound=Mapping[str, object])


@dataclass(frozen=True)
class StructureSelection:
    """A text-preserving count or ranked-list fraction.

    Integer spelling is deliberately significant: ``1`` selects one row,
    while decimal spelling (``1.0``) selects 100 percent of the rows.
    """

    raw: str
    kind: str
    numeric: Decimal

    def selected_count(self, ranking_count: int) -> int:
        total = max(0, int(ranking_count))
        if total == 0:
            return 0
        if self.kind == "count":
            return min(total, int(self.numeric))
        return min(total, max(1, math.ceil(total * float(self.numeric))))

    def details(self, ranking_count: int) -> dict[str, object]:
        return {
            "input": self.raw,
            "kind": self.kind,
            "value": int(self.numeric) if self.kind == "count" else float(self.numeric),
            "ranking_count": int(ranking_count),
            "selected_count": self.selected_count(ranking_count),
        }


def parse_structure_selection(value: str | int | float) -> StructureSelection:
    if isinstance(value, bool):
        raise ValueError("Structure selection must be a positive integer count or a decimal fraction.")

    raw = str(value).strip()
    if not raw:
        raise ValueError("Structure selection cannot be blank when explicitly set.")

    if isinstance(value, int) or _INTEGER_TEXT.fullmatch(raw):
        count = int(raw)
        if count < 1:
            raise ValueError("Structure selection count must be at least 1.")
        return StructureSelection(raw=raw, kind="count", numeric=Decimal(count))

    try:
        fraction = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(
            "Structure selection must be a positive integer count or a decimal fraction from 0 to 1."
        ) from exc
    if not fraction.is_finite() or fraction <= 0:
        raise ValueError("Structure selection fraction must be greater than 0 and at most 1.0.")
    if fraction > 1:
        raise ValueError(
            "Non-integer structure selection values above 1 are invalid; use an integer count."
        )
    return StructureSelection(raw=raw, kind="fraction", numeric=fraction)


def default_structure_selection(mode: SequenceMode) -> StructureSelection:
    return parse_structure_selection("1" if mode == "protein" else "0.25")


def resolve_effective_structure_selection(
    configured: str | int | float | None,
    mode: SequenceMode,
) -> StructureSelection:
    if configured is None or str(configured).strip() == "":
        return default_structure_selection(mode)
    return parse_structure_selection(configured)


def select_ranked_rows(
    ranking: Sequence[_RankedRow],
    selection: StructureSelection,
) -> list[_RankedRow]:
    return list(ranking[: selection.selected_count(len(ranking))])
