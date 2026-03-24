from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class BuybackStats:
    net_buybacks: float
    buyback_shrink_rate: Optional[float]
    repurchases: Optional[float]
    issuance: Optional[float]


def _pick_row(df: pd.DataFrame, candidates: list[str]) -> Optional[pd.Series]:
    for name in candidates:
        if name in df.index:
            return df.loc[name]
    return None


def _latest_value(row: pd.Series) -> Optional[float]:
    if row is None:
        return None
    row = row.dropna()
    if row.empty:
        return None
    return float(row.loc[max(row.index)])


def extract_buybacks(
    cashflow: pd.DataFrame,
    current_price: float,
    current_shares: float,
) -> BuybackStats:
    """
    Net buybacks = |RepurchaseOfCapitalStock| − |IssuanceOfCapitalStock|.
    Implied share shrink rate ≈ (net_buybacks / price) / shares_outstanding.
    """
    rep_row = _pick_row(cashflow, ["RepurchaseOfCapitalStock", "Repurchase Of Capital Stock"])
    iss_row = _pick_row(cashflow, ["IssuanceOfCapitalStock", "Issuance Of Capital Stock"])

    rep = _latest_value(rep_row)
    iss = _latest_value(iss_row)

    rep_spend = abs(rep) if rep is not None else None
    iss_inflow = abs(iss) if iss is not None else 0.0

    if rep_spend is None:
        return BuybackStats(net_buybacks=0.0, buyback_shrink_rate=None,
                            repurchases=None, issuance=None)

    net_buybacks = max(0.0, rep_spend - iss_inflow)

    shrink_rate = None
    if current_price > 0 and current_shares > 0 and net_buybacks > 0:
        shares_retired = net_buybacks / current_price
        shrink_rate = shares_retired / current_shares

    return BuybackStats(
        net_buybacks=net_buybacks,
        buyback_shrink_rate=shrink_rate,
        repurchases=rep_spend,
        issuance=iss_inflow,
    )
