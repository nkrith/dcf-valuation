from __future__ import annotations

from dataclasses import dataclass
from typing import List

from dcf.dcf_engine import run_dcf
from dcf.types import CapitalStructure, DcfAssumptions, ForecastFcff, ShareCount
from dcf.fcff_forecast import FcffFadeAssumptions, build_fcff_fade_forecast


@dataclass(frozen=True)
class SensitivityConfig:
    years: int
    start_growth: float
    wacc_values: List[float]
    terminal_g_values: List[float]


def print_wacc_g_sensitivity(
    base_fcff: float,
    capital_structure: CapitalStructure,
    shares: ShareCount,
    cfg: SensitivityConfig,
) -> None:
    """
    Prints an intrinsic price/share table:
      rows = WACC, cols = terminal growth g.
    Uses fade forecast where stable growth = terminal g for each column.
    """
    print("\nSensitivity (Intrinsic Price/Share): rows=WACC, cols=terminal g")
    header = "WACC\\g  " + "  ".join([f"{g*100:.2f}%" for g in cfg.terminal_g_values])
    print(header)

    for w in cfg.wacc_values:
        row = [f"{w*100:.2f}%"]
        for g in cfg.terminal_g_values:
            fade = FcffFadeAssumptions(years=cfg.years, start_growth=cfg.start_growth, stable_growth=g)
            fcffs = build_fcff_fade_forecast(base_fcff, fade)

            res = run_dcf(
                forecast=ForecastFcff(fcff=fcffs),
                assumptions=DcfAssumptions(forecast_years=cfg.years, wacc=w, terminal_growth=g),
                capital_structure=capital_structure,
                shares=shares,
            )
            row.append(f"{res.intrinsic_price_per_share:,.2f}")
        print("  ".join(row))
