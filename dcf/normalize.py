from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HistoricalMetrics:
    revenue: float
    ebit: float
    tax_rate: float
    nopat: float
    reinvestment: float
    fcff: float
    sales_to_capital: Optional[float]


def _pick_row(df: pd.DataFrame, candidates: list[str]) -> Optional[pd.Series]:
    for name in candidates:
        if name in df.index:
            return df.loc[name]
    return None


def _latest_value(row: pd.Series) -> float:
    row = row.dropna()
    if row.empty:
        raise ValueError("Row is empty after dropping NaNs.")
    return float(row.loc[max(row.index)])


def _safe_div(a: float, b: float) -> Optional[float]:
    if b == 0:
        return None
    return a / b


def derive_historical_metrics(
    income_stmt: pd.DataFrame,
    cashflow: pd.DataFrame,
    years_back_for_sales_to_capital: int = 3,
) -> HistoricalMetrics:
    """
    Derive base-year operating metrics from financial statements:
      Revenue, EBIT, tax rate, NOPAT, reinvestment, FCFF, sales-to-capital.
    """

    rev_row = _pick_row(income_stmt, ["Total Revenue", "Revenue", "TotalRevenue"])
    ebit_row = _pick_row(
        income_stmt,
        ["EBIT", "Operating Income", "Operating Income or Loss",
         "OperatingIncome", "OperatingIncomeOrLoss"],
    )
    pretax_row = _pick_row(income_stmt, [
        "Pretax Income", "Earnings Before Tax", "PretaxIncome", "EarningsBeforeTax",
    ])
    tax_row = _pick_row(income_stmt, [
        "Tax Provision", "Income Tax Expense", "TaxProvision", "IncomeTaxExpense",
    ])

    if rev_row is None or ebit_row is None:
        raise ValueError("Could not find Revenue and EBIT/Operating Income in income statement.")

    revenue = _latest_value(rev_row)
    ebit = _latest_value(ebit_row)

    # Tax rate
    if pretax_row is None or tax_row is None:
        tax_rate = 0.21
    else:
        pretax = _latest_value(pretax_row)
        tax = _latest_value(tax_row)
        tax_rate = max(0.0, min(1.0, tax / pretax)) if pretax > 0 else 0.21

    nopat = ebit * (1.0 - tax_rate)

    # Cash flow items
    capex_row = _pick_row(cashflow, [
        "Capital Expenditures", "Capital Expenditure", "CapitalExpenditure", "PurchaseOfPPE",
    ])
    da_row = _pick_row(cashflow, [
        "Depreciation", "Depreciation & Amortization", "Depreciation And Amortization",
        "DepreciationAndAmortization", "DepreciationAmortizationDepletion",
    ])
    dnwc_row = _pick_row(cashflow, [
        "Change In Working Capital", "Change in Working Capital",
        "Changes In Working Capital", "ChangeInWorkingCapital",
    ])

    if capex_row is None:
        raise ValueError("Could not find CapEx in cash flow statement.")

    capex = abs(_latest_value(capex_row))
    da = abs(_latest_value(da_row)) if da_row is not None else 0.0
    dnwc = _latest_value(dnwc_row) if dnwc_row is not None else 0.0

    reinvestment = capex - da + dnwc
    fcff = nopat - reinvestment

    # Sales-to-capital ratio (multi-year average)
    sales_to_capital = _compute_sales_to_capital(
        rev_row, capex_row, da_row, dnwc_row, years_back_for_sales_to_capital,
    )

    log.info("Historical metrics: rev=%.0f  ebit=%.0f  tax=%.2f%%  nopat=%.0f  "
             "reinv=%.0f  fcff=%.0f  s2c=%s",
             revenue, ebit, tax_rate * 100, nopat, reinvestment, fcff,
             f"{sales_to_capital:.2f}" if sales_to_capital else "N/A")

    return HistoricalMetrics(
        revenue=revenue,
        ebit=ebit,
        tax_rate=tax_rate,
        nopat=nopat,
        reinvestment=reinvestment,
        fcff=fcff,
        sales_to_capital=sales_to_capital,
    )


def _compute_sales_to_capital(
    rev_row: pd.Series,
    capex_row: pd.Series,
    da_row: Optional[pd.Series],
    dnwc_row: Optional[pd.Series],
    years_back: int,
) -> Optional[float]:
    """Compute average ΔRevenue / Reinvestment over recent years."""
    try:
        n_cols = max(2, years_back + 1)
        rev_s = rev_row.iloc[:n_cols]
        capex_s = capex_row.iloc[:n_cols]

        n = min(len(rev_s), len(capex_s))
        rev_s = rev_s.iloc[:n]
        capex_s = capex_s.iloc[:n]
        da_s = da_row.iloc[:n] if da_row is not None else None
        dnwc_s = dnwc_row.iloc[:n] if dnwc_row is not None else None

        reinv_s = capex_s.abs()
        if da_s is not None:
            reinv_s = reinv_s - da_s.abs()
        if dnwc_s is not None:
            reinv_s = reinv_s + dnwc_s

        deltas, reinvs = [], []
        for i in range(n - 1):
            delta_rev = float(rev_s.iloc[i] - rev_s.iloc[i + 1])
            reinv = float(reinv_s.iloc[i])
            if reinv > 0:
                deltas.append(delta_rev)
                reinvs.append(reinv)

        if not reinvs:
            return None

        avg_delta = sum(deltas) / len(deltas)
        avg_reinv = sum(reinvs) / len(reinvs)
        ratio = _safe_div(avg_delta, avg_reinv)

        # Guard against nonsensical values
        if ratio is not None and (ratio < 0.5 or ratio > 4.0):
            log.warning("Sales-to-capital ratio %.2f outside sane range [0.5, 4.0]; discarding", ratio)
            return None

        return ratio
    except Exception:
        return None
