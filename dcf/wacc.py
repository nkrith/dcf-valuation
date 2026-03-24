from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapmInputs:
    """
    CAPM inputs to compute cost of equity.
    All rates are decimals: 0.04 = 4%.
    """
    risk_free_rate: float
    equity_risk_premium: float
    beta: float


@dataclass(frozen=True)
class DebtInputs:
    """Pre-tax cost of debt and effective tax rate."""
    pre_tax_cost_of_debt: float
    tax_rate: float


@dataclass(frozen=True)
class MarketValueInputs:
    """Market-value weights for WACC (not book values)."""
    market_value_of_equity: float
    market_value_of_debt: float


def cost_of_equity_capm(capm: CapmInputs) -> float:
    """Re = Rf + β × ERP"""
    return capm.risk_free_rate + capm.beta * capm.equity_risk_premium


def after_tax_cost_of_debt(debt: DebtInputs) -> float:
    """Rd_after_tax = Rd × (1 − T)"""
    if not (0.0 <= debt.tax_rate <= 1.0):
        raise ValueError(f"tax_rate must be between 0 and 1, got {debt.tax_rate}")
    return debt.pre_tax_cost_of_debt * (1.0 - debt.tax_rate)


def wacc(capm: CapmInputs, debt: DebtInputs, mv: MarketValueInputs) -> float:
    """WACC = (E/(D+E)) × Re  +  (D/(D+E)) × Rd × (1−T)"""
    if mv.market_value_of_equity <= 0:
        raise ValueError("market_value_of_equity must be positive.")
    if mv.market_value_of_debt < 0:
        raise ValueError("market_value_of_debt cannot be negative.")

    total = mv.market_value_of_equity + mv.market_value_of_debt
    if total <= 0:
        raise ValueError("Total capital (D+E) must be positive.")

    re = cost_of_equity_capm(capm)
    rd_at = after_tax_cost_of_debt(debt)

    weight_e = mv.market_value_of_equity / total
    weight_d = mv.market_value_of_debt / total

    return weight_e * re + weight_d * rd_at
