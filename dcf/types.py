from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DcfAssumptions:
    forecast_years: int
    wacc: float
    terminal_growth: float


@dataclass(frozen=True)
class CapitalStructure:
    cash_and_equivalents: float
    total_debt: float
    minority_interest: float = 0.0
    preferred_equity: float = 0.0


@dataclass(frozen=True)
class ShareCount:
    diluted_shares: float


@dataclass(frozen=True)
class ForecastFcff:
    fcff: List[float]


@dataclass(frozen=True)
class DcfResult:
    enterprise_value: float
    equity_value: float
    intrinsic_price_per_share: float
    pv_of_explicit_period: float
    pv_of_terminal_value: float
