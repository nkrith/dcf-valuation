from __future__ import annotations
from typing import Iterable, Optional
import math


def first_valid(values: Iterable[float]) -> Optional[float]:
    for v in values:
        if v is not None and not math.isnan(v):
            return float(v)
    return None


def avg_last_valid(values: list[float], n: int = 3) -> Optional[float]:
    valid = [float(v) for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return None
    return sum(valid[:n]) / min(n, len(valid))
