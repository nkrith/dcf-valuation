from __future__ import annotations

from dataclasses import dataclass
from typing import List

from dcf.forecasting import ForecastYear, terminal_value_stable_growth
from dcf.types import (
    CapitalStructure,
    DcfAssumptions,
    DcfResult,
    ForecastFcff,
    ShareCount,
)


def _present_value(amount: float, rate: float, t: int) -> float:
    return amount / ((1.0 + rate) ** t)


def _terminal_value_gordon(last_fcff: float, wacc: float, g: float) -> float:
    """Gordon Growth terminal value: TV = FCFF₍ₙ₊₁₎ / (WACC − g)."""
    if wacc <= g:
        raise ValueError("WACC must be greater than terminal growth.")
    return last_fcff * (1.0 + g) / (wacc - g)


def run_dcf(
    forecast: ForecastFcff,
    assumptions: DcfAssumptions,
    capital_structure: CapitalStructure,
    shares: ShareCount,
) -> DcfResult:
    """Simple FCFF DCF with Gordon Growth terminal value."""
    if assumptions.forecast_years <= 0:
        raise ValueError("forecast_years must be positive.")
    if len(forecast.fcff) != assumptions.forecast_years:
        raise ValueError("Forecast length must equal forecast_years.")
    if shares.diluted_shares <= 0:
        raise ValueError("diluted_shares must be positive.")

    pv_explicit = sum(
        _present_value(fcff, assumptions.wacc, t)
        for t, fcff in enumerate(forecast.fcff, start=1)
    )

    tv = _terminal_value_gordon(forecast.fcff[-1], assumptions.wacc, assumptions.terminal_growth)
    pv_terminal = _present_value(tv, assumptions.wacc, assumptions.forecast_years)
    enterprise_value = pv_explicit + pv_terminal

    equity_value = (
        enterprise_value
        + capital_structure.cash_and_equivalents
        - capital_structure.total_debt
        - capital_structure.minority_interest
        - capital_structure.preferred_equity
    )

    return DcfResult(
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        intrinsic_price_per_share=equity_value / shares.diluted_shares,
        pv_of_explicit_period=pv_explicit,
        pv_of_terminal_value=pv_terminal,
    )


# ---------------------------------------------------------------------------
# Operating-model DCF (ROIC-consistent terminal value)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OperatingForecast:
    """Rich forecast: revenue, margins, NOPAT, reinvestment, FCFF per year."""
    years: List[ForecastYear]


def run_dcf_from_operating_forecast(
    op_forecast: OperatingForecast,
    assumptions: DcfAssumptions,
    terminal_roic: float,
    capital_structure: CapitalStructure,
    shares: ShareCount,
) -> DcfResult:
    """
    Full operating-model DCF with ROIC-consistent terminal value:
      TV = NOPAT₍ₙ₊₁₎ × (1 − g/ROIC) / (WACC − g)
    """
    if assumptions.forecast_years <= 0:
        raise ValueError("forecast_years must be positive.")
    if shares.diluted_shares <= 0:
        raise ValueError("diluted_shares must be positive.")
    if len(op_forecast.years) != assumptions.forecast_years:
        raise ValueError("Operating forecast length must match forecast_years.")

    # 1) PV of explicit period
    pv_explicit = sum(
        _present_value(y.fcff, assumptions.wacc, t)
        for t, y in enumerate(op_forecast.years, start=1)
    )

    # 2) Terminal value
    tv_at_n = terminal_value_stable_growth(
        last_year_nopat=op_forecast.years[-1].nopat,
        wacc=assumptions.wacc,
        stable_growth=assumptions.terminal_growth,
        terminal_roic=terminal_roic,
    )
    pv_terminal = _present_value(tv_at_n, assumptions.wacc, assumptions.forecast_years)

    # 3) Enterprise → Equity → Price
    enterprise_value = pv_explicit + pv_terminal
    equity_value = (
        enterprise_value
        + capital_structure.cash_and_equivalents
        - capital_structure.total_debt
        - capital_structure.minority_interest
        - capital_structure.preferred_equity
    )

    return DcfResult(
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        intrinsic_price_per_share=equity_value / shares.diluted_shares,
        pv_of_explicit_period=pv_explicit,
        pv_of_terminal_value=pv_terminal,
    )
