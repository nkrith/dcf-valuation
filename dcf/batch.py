"""
Batch runner — value multiple tickers and produce a summary table.

Usage:
    python -m dcf.batch                          # default: 10 large-caps
    python -m dcf.batch AAPL MSFT GOOG NVDA      # specific tickers
    python -m dcf.batch --sp500 --limit 50       # first 50 S&P 500 constituents
    python -m dcf.batch --sp500                  # all ~500 (takes ~30 min)
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from dcf.app import (
    _ensure_dir,
    _intrinsic_buyback_adjusted,
    _pick_balance_value,
    _run_operating_dcf,
)
from dcf.auto_assumptions import build_auto_assumptions
from dcf.beta_policy import pick_beta
from dcf.buybacks import extract_buybacks
from dcf.normalize import derive_historical_metrics
from dcf.wacc import CapmInputs, DebtInputs, MarketValueInputs, wacc
from dcf.types import CapitalStructure
from dcf.yf_data import fetch_market_snapshot, fetch_risk_free_rate_from_tnx, fetch_statements_annual

import yfinance as yf

log = logging.getLogger("dcf.batch")

# A representative sample for quick testing
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "JPM", "JNJ", "PG", "KO"]

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _fetch_sp500_tickers() -> list[str]:
    """Scrape current S&P 500 constituents from Wikipedia."""
    try:
        import pandas as pd
        tables = pd.read_html(SP500_URL)
        df = tables[0]
        return sorted(df["Symbol"].str.replace(".", "-", regex=False).tolist())
    except Exception as exc:
        log.error("Could not fetch S&P 500 list: %s", exc)
        log.info("Falling back to default tickers")
        return DEFAULT_TICKERS


def _value_single_ticker(
    ticker: str, rf: float, erp: float, beta_method: str,
) -> dict | None:
    """Run the full DCF pipeline for one ticker. Returns a summary dict or None on failure."""
    try:
        snap = fetch_market_snapshot(ticker)
        stmts = fetch_statements_annual(ticker)
        hist = derive_historical_metrics(stmts.income, stmts.cashflow)
        base_margin = (hist.ebit / hist.revenue) if hist.revenue != 0 else 0.0

        cash = _pick_balance_value(stmts.balance, [
            "Cash And Cash Equivalents Including Short Term Investments",
            "Cash And Cash Equivalents",
            "CashAndCashEquivalentsIncludingShortTermInvestments",
            "CashAndCashEquivalents",
        ], [["cash"], ["short", "term", "invest"]])

        total_debt = _pick_balance_value(stmts.balance, [
            "Total Debt", "Long Term Debt",
            "Long Term Debt And Capital Lease Obligation",
            "TotalDebt", "LongTermDebt",
        ], [["total", "debt"], ["long", "term", "debt"]])

        cap_struct = CapitalStructure(cash_and_equivalents=cash, total_debt=total_debt)

        auto = build_auto_assumptions(
            income_stmt=stmts.income, balance=stmts.balance,
            hist_nopat=hist.nopat, hist_reinvestment=hist.reinvestment,
            base_margin=base_margin, market_cap=snap.market_cap, rf=rf,
        )

        bb = extract_buybacks(stmts.cashflow, current_price=snap.price,
                              current_shares=snap.shares_outstanding)

        info = yf.Ticker(ticker).info or {}
        industry_guess = info.get("industry") or info.get("sector") or ""
        beta_pick = pick_beta(
            beta_raw=snap.beta, method=beta_method,
            industry_guess=industry_guess,
            debt=total_debt, equity_mkt=snap.market_cap, tax_rate=hist.tax_rate,
        )

        computed_wacc = wacc(
            capm=CapmInputs(risk_free_rate=rf, equity_risk_premium=erp, beta=beta_pick.beta_used),
            debt=DebtInputs(pre_tax_cost_of_debt=auto.rd_pre_tax, tax_rate=hist.tax_rate),
            mv=MarketValueInputs(market_value_of_equity=snap.market_cap, market_value_of_debt=total_debt),
        )

        res, _ = _run_operating_dcf(
            hist=hist, snap=snap, cap_struct=cap_struct,
            computed_wacc=computed_wacc,
            forecast_years=auto.forecast_years,
            start_growth=auto.start_revenue_growth,
            terminal_g=auto.terminal_growth,
            terminal_roic=auto.terminal_roic,
            terminal_margin=auto.terminal_operating_margin,
            fallback_reinvestment_rate=auto.fallback_reinvestment_rate,
        )

        intrinsic = _intrinsic_buyback_adjusted(
            res.equity_value, snap.shares_outstanding,
            bb.buyback_shrink_rate, auto.forecast_years,
        )

        market = snap.price
        upside = (intrinsic / market - 1.0) * 100

        if upside > 15:
            signal = "UNDERVALUED"
        elif upside < -15:
            signal = "OVERVALUED"
        else:
            signal = "FAIR"

        return {
            "ticker": ticker,
            "market_price": round(market, 2),
            "intrinsic_price": round(intrinsic, 2),
            "upside_pct": round(upside, 1),
            "signal": signal,
            "wacc_pct": round(computed_wacc * 100, 2),
            "start_growth_pct": round(auto.start_revenue_growth * 100, 2),
            "reinv_rate_pct": round(auto.fallback_reinvestment_rate * 100, 2),
            "forecast_years": auto.forecast_years,
            "beta": round(beta_pick.beta_used, 2),
        }

    except Exception as exc:
        log.warning("%s — FAILED: %s", ticker, exc)
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Batch DCF valuation across multiple tickers.")
    p.add_argument("tickers", nargs="*", help="Ticker symbols (default: 10 large-caps)")
    p.add_argument("--sp500", action="store_true", help="Run against S&P 500 constituents")
    p.add_argument("--limit", type=int, default=0, help="Limit number of tickers (0 = no limit)")
    p.add_argument("--erp", type=float, default=0.05, help="Equity risk premium")
    p.add_argument("--beta-method", default="mean_revert", choices=["raw", "mean_revert", "damodaran"])
    p.add_argument("--outdir", default="outputs", help="Output directory")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    # Pick tickers
    if args.sp500:
        tickers = _fetch_sp500_tickers()
    elif args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = DEFAULT_TICKERS

    if args.limit > 0:
        tickers = tickers[:args.limit]

    print(f"\nBatch DCF — {len(tickers)} tickers\n{'─'*50}")

    # Fetch risk-free rate once
    rf = fetch_risk_free_rate_from_tnx()

    results = []
    failed = []

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i:>3}/{len(tickers)}] {ticker:<6}", end=" … ", flush=True)
        row = _value_single_ticker(ticker, rf=rf, erp=args.erp, beta_method=args.beta_method)
        if row:
            results.append(row)
            print(f"${row['market_price']:>8,.2f} → ${row['intrinsic_price']:>8,.2f}  "
                  f"({row['upside_pct']:>+6.1f}%)  {row['signal']}")
        else:
            failed.append(ticker)
            print("FAILED")

        time.sleep(0.3)  # rate-limit yfinance

    # Summary
    print(f"\n{'='*60}")
    print(f"  Completed: {len(results)}/{len(tickers)}  |  Failed: {len(failed)}")

    if results:
        undervalued = [r for r in results if r["signal"] == "UNDERVALUED"]
        overvalued = [r for r in results if r["signal"] == "OVERVALUED"]
        fair = [r for r in results if r["signal"] == "FAIR"]
        print(f"  Undervalued: {len(undervalued)}  |  Fair: {len(fair)}  |  Overvalued: {len(overvalued)}")

        # Top 5 most undervalued
        by_upside = sorted(results, key=lambda x: x["upside_pct"], reverse=True)
        print(f"\n  Top 5 by upside:")
        for r in by_upside[:5]:
            print(f"    {r['ticker']:<6}  {r['upside_pct']:>+6.1f}%  "
                  f"(${r['market_price']:.2f} → ${r['intrinsic_price']:.2f})")

    # Save CSV
    if results:
        outdir = _ensure_dir(args.outdir)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        csv_path = outdir / f"batch_{ts}.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print(f"\n  Saved: {csv_path}")

    if failed:
        print(f"\n  Failed tickers: {', '.join(failed)}")

    print()


if __name__ == "__main__":
    main()
