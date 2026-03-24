from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class ForecastAssumptions:
    years: int
    start_revenue_growth: float
    stable_growth: float
    start_operating_margin: float
    terminal_operating_margin: float
    sales_to_capital: Optional[float] = None
    fallback_reinvestment_rate: float = 0.20
    terminal_roic: float = 0.12
    tax_rate: float = 0.21


@dataclass(frozen=True)
class ForecastYear:
    year: int
    revenue: float
    op_margin: float
    ebit: float
    nopat: float
    reinvestment: float
    fcff: float


def _linear_converge(start: float, end: float, t: int, n: int) -> float:
    if n <= 1:
        return end
    w = (t - 1) / (n - 1)
    return start + w * (end - start)


def build_operating_forecast(
    base_revenue: float,
    assumptions: ForecastAssumptions,
) -> List[ForecastYear]:
    """
    Build a multi-year operating forecast:
      - Revenue grows along a linear-convergence path (start_g → stable_g)
      - Margins converge from current to terminal
      - Reinvestment via sales-to-capital ratio (preferred) or NOPAT fallback
      - FCFF = NOPAT − Reinvestment
    """
    out: List[ForecastYear] = []
    revenue = base_revenue

    for t in range(1, assumptions.years + 1):
        g_t = _linear_converge(assumptions.start_revenue_growth, assumptions.stable_growth,
                               t, assumptions.years)
        margin_t = _linear_converge(assumptions.start_operating_margin, assumptions.terminal_operating_margin,
                                    t, assumptions.years)

        revenue_next = revenue * (1.0 + g_t)
        delta_rev = revenue_next - revenue

        ebit = revenue_next * margin_t
        nopat = ebit * (1.0 - assumptions.tax_rate)

        if assumptions.sales_to_capital and assumptions.sales_to_capital > 0:
            reinvestment = max(0.0, delta_rev / assumptions.sales_to_capital)
        else:
            reinvestment = max(0.0, nopat * assumptions.fallback_reinvestment_rate)

        fcff = nopat - reinvestment

        out.append(ForecastYear(
            year=t, revenue=revenue_next, op_margin=margin_t,
            ebit=ebit, nopat=nopat, reinvestment=reinvestment, fcff=fcff,
        ))

        revenue = revenue_next

    return out


def terminal_value_stable_growth(
    last_year_nopat: float,
    wacc: float,
    stable_growth: float,
    terminal_roic: float,
) -> float:
    """
    ROIC-consistent terminal value:
      Reinvestment Rate = g / ROIC
      FCFF₍ₙ₊₁₎ = NOPAT₍ₙ₊₁₎ × (1 − g/ROIC)
      TV = FCFF₍ₙ₊₁₎ / (WACC − g)
    """
    if wacc <= stable_growth:
        raise ValueError(f"WACC ({wacc:.4f}) must exceed terminal g ({stable_growth:.4f}).")
    if terminal_roic <= stable_growth:
        raise ValueError(f"Terminal ROIC ({terminal_roic:.4f}) must exceed terminal g ({stable_growth:.4f}).")

    nopat_next = last_year_nopat * (1.0 + stable_growth)
    reinvestment_rate = stable_growth / terminal_roic
    fcff_next = nopat_next * (1.0 - reinvestment_rate)

    return fcff_next / (wacc - stable_growth)
