from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass(frozen=True)
class FcHistory:
    years: List[str]
    cfo: List[float]
    capex: List[float]
    fcf: List[float]
    interest_paid: Optional[float]
    tax_rate: Optional[float]
    fcff_unlevered: Optional[float]


def _pick_row(df: pd.DataFrame, candidates: list[str]) -> Optional[pd.Series]:
    for name in candidates:
        if name in df.index:
            return df.loc[name]
    return None


def _clean_float(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(v) else v


def _as_list(series: pd.Series) -> List[float]:
    return [_clean_float(x) if _clean_float(x) is not None else float("nan") for x in series.values]


def extract_fcf_history(
    income_stmt: pd.DataFrame,
    cashflow: pd.DataFrame,
    n_years: int = 5,
) -> FcHistory:
    """
    FCF = FreeCashFlow (preferred) or OperatingCashFlow − CapEx.
    FCFF ≈ FCF + InterestPaid × (1 − T).
    """
    fcf_row = _pick_row(cashflow, ["FreeCashFlow"])
    cfo_row = _pick_row(cashflow, ["OperatingCashFlow", "CashFlowFromContinuingOperatingActivities"])
    capex_row = _pick_row(cashflow, ["CapitalExpenditure", "PurchaseOfPPE"])
    interest_paid_row = _pick_row(cashflow, ["InterestPaidSupplementalData"])

    if cfo_row is None or capex_row is None:
        raise ValueError("Could not find OperatingCashFlow or CapitalExpenditure.")

    cfo_series = cfo_row.iloc[:n_years]
    capex_series = capex_row.iloc[:n_years]
    fcf_series = fcf_row.iloc[:n_years] if fcf_row is not None else cfo_series - capex_series

    years = [str(c) for c in fcf_series.index]

    interest_paid = None
    if interest_paid_row is not None:
        interest_paid = _clean_float(abs(float(interest_paid_row.iloc[0])))

    tax_rate = None
    pretax_row = _pick_row(income_stmt, ["PretaxIncome", "EarningsBeforeTax", "Pretax Income", "Earnings Before Tax"])
    tax_row = _pick_row(income_stmt, ["TaxProvision", "IncomeTaxExpense", "Tax Provision", "Income Tax Expense"])
    if pretax_row is not None and tax_row is not None:
        pretax = _clean_float(pretax_row.iloc[0])
        tax = _clean_float(tax_row.iloc[0])
        if pretax is not None and pretax > 0 and tax is not None:
            tax_rate = max(0.0, min(1.0, tax / pretax))

    fcf_base = None
    for v in fcf_series.values:
        vv = _clean_float(v)
        if vv is not None:
            fcf_base = vv
            break

    if fcf_base is None:
        raise ValueError("FCF series contains no valid numeric values.")

    fcff_unlevered = None
    if interest_paid is not None and tax_rate is not None:
        fcff_unlevered = fcf_base + interest_paid * (1.0 - tax_rate)
        if pd.isna(fcff_unlevered):
            fcff_unlevered = None

    return FcHistory(
        years=years, cfo=_as_list(cfo_series), capex=_as_list(capex_series),
        fcf=_as_list(fcf_series), interest_paid=interest_paid,
        tax_rate=tax_rate, fcff_unlevered=fcff_unlevered,
    )
