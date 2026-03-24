from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AutoAssumptions:
    forecast_years: int
    start_revenue_growth: float
    terminal_growth: float
    terminal_roic: float
    terminal_operating_margin: float
    fallback_reinvestment_rate: float
    rd_pre_tax: float
    current_roic: Optional[float]
    invested_capital_est: Optional[float]


def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        return None if v != v else v  # NaN check
    except Exception:
        return None


def _normalize_index(s: str) -> str:
    return str(s).strip().lower().replace("&", "and").replace("-", " ").replace("_", " ")


def _get_latest_row_value(df: pd.DataFrame, labels: list[str]) -> Optional[float]:
    for lab in labels:
        if lab in df.index:
            row = df.loc[lab].dropna()
            if row.empty:
                continue
            row = row.sort_index()
            return _safe_float(row.iloc[-1])
    return None


def _get_latest_by_keywords(df: pd.DataFrame, keyword_groups: list[list[str]]) -> Optional[float]:
    if df is None or df.empty:
        return None
    norm_index = [(idx, _normalize_index(idx)) for idx in df.index]
    for group in keyword_groups:
        group_norm = [_normalize_index(k) for k in group]
        for idx, nidx in norm_index:
            if all(k in nidx for k in group_norm):
                row = df.loc[idx].dropna()
                if row.empty:
                    continue
                row = row.sort_index()
                return _safe_float(row.iloc[-1])
    return None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _estimate_revenue_cagr(income_stmt: pd.DataFrame, years: int = 3) -> Optional[float]:
    for label in ["Total Revenue", "Revenue", "TotalRevenue"]:
        if label in income_stmt.index:
            row = income_stmt.loc[label].dropna()
            if len(row) < years + 1:
                continue
            row = row.sort_index()  # oldest → newest
            latest = float(row.iloc[-1])
            older = float(row.iloc[-(years + 1)])
            if older <= 0:
                return None
            return (latest / older) ** (1.0 / years) - 1.0
    return None


def _estimate_start_growth(income_stmt: pd.DataFrame, market_cap: float) -> float:
    """
    Robust start-growth estimate.
    Takes the max of 1y/3y/5y revenue CAGRs, with size-based floors and caps.
    """
    c1 = _estimate_revenue_cagr(income_stmt, years=1)
    c3 = _estimate_revenue_cagr(income_stmt, years=3)
    c5 = _estimate_revenue_cagr(income_stmt, years=5)

    vals = [v for v in (c1, c3, c5) if v is not None]
    g = max(vals) if vals else 0.06

    # Size-based floor (avoids bad-window understatement for mega-caps)
    if market_cap >= 500e9:
        g = max(g, 0.05)
    elif market_cap >= 50e9:
        g = max(g, 0.04)

    cap = 0.30 if g > 0.12 else 0.18
    result = _clamp(g, 0.0, cap)

    log.info("Start growth: 1y=%s  3y=%s  5y=%s -> selected=%.2f%%",
             f"{c1:.2%}" if c1 is not None else "N/A",
             f"{c3:.2%}" if c3 is not None else "N/A",
             f"{c5:.2%}" if c5 is not None else "N/A",
             result * 100)

    return result


def build_auto_assumptions(
    income_stmt: pd.DataFrame,
    balance: pd.DataFrame,
    hist_nopat: float,
    hist_reinvestment: float,
    base_margin: float,
    market_cap: float,
    rf: float,
) -> AutoAssumptions:
    """
    Fully automated, ticker-agnostic assumptions.

    Philosophy:
      - Prefer stable, robust heuristics over fragile invested-capital parsing.
      - Growth runway scales with observed growth (market prices in runway).
      - Terminal g conservative for USD DCF (2–3%).
    """

    # 1) Start growth
    start_g = _estimate_start_growth(income_stmt, market_cap=market_cap)

    # 2) Forecast horizon — quality compounders need longer runways.
    #    The terminal value assumes 2.5% growth forever, so a short horizon
    #    kills value for any company still growing above that.
    if market_cap >= 500e9:
        forecast_years = 15
    elif market_cap >= 50e9:
        forecast_years = 12
    else:
        forecast_years = 10

    if start_g >= 0.10:
        forecast_years = max(forecast_years, 15)
    if start_g >= 0.20:
        forecast_years = max(forecast_years, 18)

    # 3) Terminal growth
    terminal_g = _clamp(0.025, 0.02, 0.03)

    # 4) ROIC proxy (margin-band heuristic)
    if base_margin >= 0.35:
        current_roic = 0.35
    elif base_margin >= 0.25:
        current_roic = 0.25
    elif base_margin >= 0.15:
        current_roic = 0.18
    elif base_margin >= 0.08:
        current_roic = 0.12
    else:
        current_roic = 0.08

    mature_anchor = 0.16 if base_margin >= 0.30 else 0.12
    terminal_roic = 0.5 * current_roic + 0.5 * mature_anchor
    terminal_roic = _clamp(terminal_roic, 0.08, 0.22)

    if terminal_roic <= terminal_g + 0.01:
        terminal_roic = terminal_g + 0.01

    # 5) Terminal margin (mild mean-reversion)
    terminal_margin = _clamp(base_margin * 0.95, 0.03, 0.45)

    # 6) Reinvestment fallback (explicit period only)
    #
    #    IMPORTANT: g/ROIC determines reinvestment in the TERMINAL VALUE,
    #    which is already handled by terminal_value_stable_growth().
    #    In the explicit period, reinvestment should reflect the company's
    #    actual capital intensity.
    #
    #    When historical reinvestment is negative (D&A > CapEx), that's a
    #    real signal: the company is asset-light and grows without heavy
    #    reinvestment (e.g. AAPL, MSFT, Google).  We use a small positive
    #    rate (10%) to be conservative, but NOT g/ROIC — that would
    #    overstate reinvestment and crush the explicit-period FCFFs.
    fallback_reinv = 0.20
    if hist_nopat and hist_nopat > 0 and hist_reinvestment > 0:
        r = hist_reinvestment / hist_nopat
        if r == r:  # not NaN
            fallback_reinv = _clamp(r, 0.05, 0.70)
    else:
        # Asset-light company: negative/zero reinvestment historically.
        # Use a conservative small positive rate.
        fallback_reinv = 0.10

    # 7) Pre-tax cost of debt (Rf + size spread)
    if market_cap >= 500e9:
        rd_pre_tax = _clamp(rf + 0.010, 0.035, 0.070)
    elif market_cap >= 50e9:
        rd_pre_tax = _clamp(rf + 0.015, 0.040, 0.080)
    else:
        rd_pre_tax = _clamp(rf + 0.025, 0.050, 0.100)

    auto = AutoAssumptions(
        forecast_years=forecast_years,
        start_revenue_growth=start_g,
        terminal_growth=terminal_g,
        terminal_roic=terminal_roic,
        terminal_operating_margin=terminal_margin,
        fallback_reinvestment_rate=fallback_reinv,
        rd_pre_tax=rd_pre_tax,
        current_roic=current_roic,
        invested_capital_est=None,
    )

    log.info("Auto assumptions: years=%d  g0=%.2f%%  gT=%.2f%%  ROIC_T=%.2f%%  "
             "margin_T=%.2f%%  reinv=%.2f%%  Rd=%.2f%%",
             auto.forecast_years, auto.start_revenue_growth * 100,
             auto.terminal_growth * 100, auto.terminal_roic * 100,
             auto.terminal_operating_margin * 100, auto.fallback_reinvestment_rate * 100,
             auto.rd_pre_tax * 100)

    return auto
