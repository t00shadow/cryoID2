"""BLAST significance calculations that remain usable below float range."""
from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation


def alignment_log10_evalue(evalue: float, *, bit_score: float | None = None,
                           search_space: int | None = None) -> float:
    if not math.isfinite(evalue) or evalue < 0:
        raise ValueError(f"Invalid BLAST E-value: {evalue}")
    if evalue > 0:
        return math.log10(evalue)
    if (bit_score is None or not math.isfinite(bit_score) or bit_score <= 0
            or search_space is None or search_space <= 0):
        raise ValueError("A zero BLAST E-value requires its bit score and positive search space.")
    # E = search_space * 2**(-bit_score). SeqBlast explicitly supplies
    # -searchsp, so this uses the same search space as the alignment.
    # https://www.ncbi.nlm.nih.gov/BLAST/tutorial/Altschul-1.html
    return math.log10(search_space) - bit_score * math.log10(2)


def linear_evalue(log10_value: float) -> float:
    """Compatibility value only; ranking and formatting use the log value."""
    try:
        return 10.0 ** log10_value
    except OverflowError:
        return math.inf


def format_log10_evalue(log10_value: float) -> str:
    if not math.isfinite(log10_value):
        if math.isnan(log10_value):
            raise ValueError("Cannot format a NaN composite E-value")
        return "inf" if log10_value > 0 else "0.0e+00"
    exponent = math.floor(log10_value)
    mantissa = f"{10.0 ** (log10_value - exponent):.1f}"
    if mantissa == "10.0":
        mantissa, exponent = "1.0", exponent + 1
    return f"{mantissa}e{exponent:+03d}"


def negative_log10_composite(row) -> float | str:
    """Use the unrounded log column, with support for legacy summary rows."""
    log_value = row.get("composite_log10_evalue")
    try:
        if log_value not in (None, ""):
            value = float(log_value)
            return -value if math.isfinite(value) else ""
        value = Decimal(str(row.get("composite_evalue") or ""))
        if value.is_finite() and value > 0:
            return -float(value.log10())
        # An old summary that rounded away its value cannot recover its
        # significance; retain the historical display fallback in that case.
        return 300.0 if value.is_zero() else ""
    except (InvalidOperation, ValueError, TypeError, OverflowError):
        return ""
