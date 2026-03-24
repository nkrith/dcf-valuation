from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import yfinance as yf
import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketSnapshot:
    price: float
    market_cap: float
    shares_outstanding: float
    beta: Optional[float]


@dataclass(frozen=True)
class Statements:
    income: pd.DataFrame
    cashflow: pd.DataFrame
    balance: pd.DataFrame


def _to_float(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fetch_market_snapshot(ticker: str) -> MarketSnapshot:
    t = yf.Ticker(ticker)
    info = t.info or {}

    price = _to_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    market_cap = _to_float(info.get("marketCap"))
    shares = _to_float(info.get("sharesOutstanding"))
    beta = _to_float(info.get("beta"))

    if price is None or market_cap is None or shares is None:
        available = {k: info.get(k) for k in ("currentPrice", "regularMarketPrice",
                                                "marketCap", "sharesOutstanding")}
        raise ValueError(
            f"Missing market data for {ticker}. "
            f"Got: {available}. Check the ticker symbol is valid."
        )

    log.info("Snapshot %s: price=%.2f  mcap=%.0f  shares=%.0f  beta=%s",
             ticker, price, market_cap, shares, beta)

    return MarketSnapshot(price=price, market_cap=market_cap,
                          shares_outstanding=shares, beta=beta)


def fetch_statements_annual(ticker: str) -> Statements:
    """
    Compatible across yfinance versions.
    Tries newer getter methods first, then falls back to classic attributes.
    """
    t = yf.Ticker(ticker)

    # --- Try getter methods (newer yfinance ≥ 0.2.31) ---
    if hasattr(t, "get_income_stmt") and hasattr(t, "get_cashflow") and hasattr(t, "get_balance_sheet"):
        try:
            income = t.get_income_stmt()
            cashflow = t.get_cashflow()
            balance = t.get_balance_sheet()
            if (income is not None and cashflow is not None and balance is not None
                    and not income.empty and not cashflow.empty and not balance.empty):
                log.info("Fetched statements via get_* methods (%d inc rows, %d cf rows, %d bs rows)",
                         len(income), len(cashflow), len(balance))
                return Statements(income=income, cashflow=cashflow, balance=balance)
        except Exception as exc:
            log.debug("get_* methods failed (%s), falling back to attributes", exc)

    # --- Fallback: classic attributes ---
    income = getattr(t, "financials", None)
    cashflow = getattr(t, "cashflow", None)
    balance = getattr(t, "balance_sheet", None)

    if income is None or cashflow is None or balance is None:
        raise RuntimeError(f"Could not fetch statements for {ticker}. "
                           "Ensure yfinance is up to date: pip install -U yfinance")

    if income.empty or cashflow.empty or balance.empty:
        raise RuntimeError(f"Statements returned empty for {ticker}.")

    log.info("Fetched statements via classic attributes (%d inc rows, %d cf rows, %d bs rows)",
             len(income), len(cashflow), len(balance))

    return Statements(income=income, cashflow=cashflow, balance=balance)


def fetch_risk_free_rate_from_tnx() -> float:
    """
    Pulls the 10-Year Treasury yield from ^TNX.
    Handles Yahoo's inconsistent scaling (42 = 4.2% vs 4.2 = 4.2%).
    """
    tnx = yf.Ticker("^TNX")
    hist = tnx.history(period="5d")
    if hist.empty:
        raise ValueError("Could not fetch ^TNX data — check your internet connection.")
    last = float(hist["Close"].iloc[-1])

    # Yahoo sometimes reports yield × 10 (e.g. 42 for 4.2%)
    pct = last / 10.0 if last >= 20 else last
    rf = pct / 100.0

    log.info("Risk-free rate (^TNX): raw=%.2f -> Rf=%.4f", last, rf)
    return rf
