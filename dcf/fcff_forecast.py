from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FcffFadeAssumptions:
    """Growth fades linearly from start_growth to stable_growth over the explicit period."""
    years: int
    start_growth: float
    stable_growth: float


def build_fcff_fade_forecast(base_fcff: float, a: FcffFadeAssumptions) -> List[float]:
    """Build FCFF forecast where growth rate linearly fades from start to stable."""
    if a.years <= 0:
        raise ValueError("years must be positive.")

    fcffs: List[float] = []
    x = base_fcff

    for t in range(1, a.years + 1):
        if a.years == 1:
            g = a.stable_growth
        else:
            w = (t - 1) / (a.years - 1)
            g = a.start_growth + w * (a.stable_growth - a.start_growth)
        x *= (1.0 + g)
        fcffs.append(x)

    return fcffs
