"""
DCF Valuation Engine — CLI entry point.

Pulls live market data, derives operating metrics, builds a multi-year
FCFF forecast with margin convergence, and computes intrinsic value per
share using an ROIC-consistent terminal value.

Usage:
    python -m dcf AAPL
    python -m dcf MSFT --erp 0.055 --beta-method damodaran
"""

from __future__ import annotations

import argparse
import jsongit push -u origin main --force
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from dcf.beta_policy import pick_beta
from dcf.yf_data import fetch_market_snapshot, fetch_risk_free_rate_from_tnx, fetch_statements_annual
from dcf.normalize import derive_historical_metrics
from dcf.buybacks import extract_buybacks
from dcf.forecasting import ForecastAssumptions, build_operating_forecast
from dcf.wacc import CapmInputs, DebtInputs, MarketValueInputs, wacc
from dcf.types import CapitalStructure, DcfAssumptions, ShareCount
from dcf.dcf_engine import OperatingForecast, run_dcf_from_operating_forecast
from dcf.visuals import HeatmapSpec, plot_heatmap
from dcf.auto_assumptions import build_auto_assumptions

log = logging.getLogger("dcf")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Automated intrinsic-value engine: FCFF DCF with ROIC-consistent "
            "terminal value, buyback adjustment, and WACC × g sensitivity analysis."
        ),
    )
    p.add_argument("ticker", nargs="?", default="AAPL",
                   help="Ticker symbol (default: AAPL)")
    p.add_argument("--erp", type=float, default=0.05,
                   help="Equity risk premium (decimal, default 0.05)")
    p.add_argument("--beta-method", choices=["raw", "mean_revert", "damodaran"],
                   default="mean_revert",
                   help="Beta selection policy (default: mean_revert)")
    p.add_argument("--outdir", default="outputs",
                   help="Output directory (default: outputs)")
    p.add_argument("--no-plot", action="store_true",
                   help="Suppress plot window (still saves PNG)")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable debug logging")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> Path:
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pick_balance_value(
    balance: pd.DataFrame,
    candidates: list[str],
    keywords: Optional[list[list[str]]] = None,
) -> float:
    """Extract a scalar from the balance sheet by label or keyword match."""
    for n in candidates:
        if n in balance.index:
            v = float(balance.loc[n].iloc[0])
            return v if v == v else 0.0

    if keywords:
        for kw_group in keywords:
            for idx in balance.index:
                s = str(idx).lower()
                if all(k.lower() in s for k in kw_group):
                    v = float(balance.loc[idx].iloc[0])
                    return v if v == v else 0.0

    return 0.0


def _intrinsic_buyback_adjusted(
    equity_value: float,
    shares_today: float,
    buyback_shrink_rate: Optional[float],
    years: int,
) -> float:
    """Adjust intrinsic price for expected share-count reduction from buybacks."""
    effective_shares = shares_today
    if buyback_shrink_rate is not None and buyback_shrink_rate > 0:
        mid_year = years / 2.0
        effective_shares = shares_today * ((1.0 - buyback_shrink_rate) ** mid_year)

    if effective_shares <= 0:
        effective_shares = shares_today

    return equity_value / effective_shares


# ---------------------------------------------------------------------------
# Core model runner
# ---------------------------------------------------------------------------

def _run_operating_dcf(hist, snap, cap_struct, computed_wacc,
                       forecast_years, start_growth, terminal_g,
                       terminal_roic, terminal_margin, fallback_reinvestment_rate):
    base_margin = (hist.ebit / hist.revenue) if hist.revenue != 0 else 0.0

    f_assump = ForecastAssumptions(
        years=forecast_years,
        start_revenue_growth=start_growth,
        stable_growth=terminal_g,
        start_operating_margin=base_margin,
        terminal_operating_margin=terminal_margin,
        sales_to_capital=hist.sales_to_capital,
        fallback_reinvestment_rate=fallback_reinvestment_rate,
        terminal_roic=terminal_roic,
        tax_rate=hist.tax_rate,
    )

    years_data = build_operating_forecast(hist.revenue, f_assump)

    dcf_assumptions = DcfAssumptions(
        forecast_years=forecast_years,
        wacc=computed_wacc,
        terminal_growth=terminal_g,
    )

    res = run_dcf_from_operating_forecast(
        op_forecast=OperatingForecast(years=years_data),
        assumptions=dcf_assumptions,
        terminal_roic=terminal_roic,
        capital_structure=cap_struct,
        shares=ShareCount(diluted_shares=snap.shares_outstanding),
    )

    return res, years_data


def build_wacc_g_matrix(
    hist, snap, cap_struct, forecast_years, start_growth,
    terminal_roic, terminal_margin, fallback_reinvestment_rate,
    wacc_values, g_values, buyback_shrink_rate,
) -> list[list[float]]:
    """Build a 2-D matrix of intrinsic prices across WACC × terminal-g."""
    matrix: list[list[float]] = []
    for w in wacc_values:
        row: list[float] = []
        for g in g_values:
            res, _ = _run_operating_dcf(
                hist=hist, snap=snap, cap_struct=cap_struct,
                computed_wacc=w, forecast_years=forecast_years,
                start_growth=start_growth, terminal_g=g,
                terminal_roic=terminal_roic, terminal_margin=terminal_margin,
                fallback_reinvestment_rate=fallback_reinvestment_rate,
            )
            price = _intrinsic_buyback_adjusted(
                res.equity_value, snap.shares_outstanding,
                buyback_shrink_rate, forecast_years,
            )
            row.append(price)
        matrix.append(row)
    return matrix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    ticker = args.ticker.upper()
    outdir = _ensure_dir(args.outdir)

    # --- Market data ---
    log.info("Fetching data for %s …", ticker)
    snap = fetch_market_snapshot(ticker)
    stmts = fetch_statements_annual(ticker)

    # --- Historical operating metrics ---
    hist = derive_historical_metrics(stmts.income, stmts.cashflow)
    base_margin = (hist.ebit / hist.revenue) if hist.revenue != 0 else 0.0

    # --- Cash & debt (EV → equity bridge) ---
    cash = _pick_balance_value(
        stmts.balance,
        candidates=[
            "Cash And Cash Equivalents Including Short Term Investments",
            "Cash And Cash Equivalents",
            "CashAndCashEquivalentsIncludingShortTermInvestments",
            "CashAndCashEquivalents",
        ],
        keywords=[["cash"], ["short", "term", "invest"]],
    )
    total_debt = _pick_balance_value(
        stmts.balance,
        candidates=[
            "Total Debt", "Long Term Debt",
            "Long Term Debt And Capital Lease Obligation",
            "TotalDebt", "LongTermDebt",
        ],
        keywords=[["total", "debt"], ["long", "term", "debt"]],
    )
    log.info("Balance sheet: cash=%.0f  debt=%.0f", cash, total_debt)

    cap_struct = CapitalStructure(cash_and_equivalents=cash, total_debt=total_debt)

    # --- Risk-free rate + ERP ---
    rf = fetch_risk_free_rate_from_tnx()
    erp = float(args.erp)

    # --- Auto assumptions ---
    auto = build_auto_assumptions(
        income_stmt=stmts.income, balance=stmts.balance,
        hist_nopat=hist.nopat, hist_reinvestment=hist.reinvestment,
        base_margin=base_margin, market_cap=snap.market_cap, rf=rf,
    )

    # --- Buybacks ---
    bb = extract_buybacks(stmts.cashflow, current_price=snap.price,
                          current_shares=snap.shares_outstanding)

    # --- Beta ---
    info = yf.Ticker(ticker).info or {}
    industry_guess = info.get("industry") or info.get("sector") or ""

    beta_pick = pick_beta(
        beta_raw=snap.beta, method=args.beta_method,
        industry_guess=industry_guess,
        debt=total_debt, equity_mkt=snap.market_cap, tax_rate=hist.tax_rate,
    )

    # --- WACC ---
    computed_wacc = wacc(
        capm=CapmInputs(risk_free_rate=rf, equity_risk_premium=erp, beta=beta_pick.beta_used),
        debt=DebtInputs(pre_tax_cost_of_debt=auto.rd_pre_tax, tax_rate=hist.tax_rate),
        mv=MarketValueInputs(market_value_of_equity=snap.market_cap, market_value_of_debt=total_debt),
    )

    # --- Run DCF ---
    res, forecast_years_data = _run_operating_dcf(
        hist=hist, snap=snap, cap_struct=cap_struct,
        computed_wacc=computed_wacc,
        forecast_years=auto.forecast_years,
        start_growth=auto.start_revenue_growth,
        terminal_g=auto.terminal_growth,
        terminal_roic=auto.terminal_roic,
        terminal_margin=auto.terminal_operating_margin,
        fallback_reinvestment_rate=auto.fallback_reinvestment_rate,
    )

    intrinsic_base = res.intrinsic_price_per_share
    intrinsic_bb = _intrinsic_buyback_adjusted(
        res.equity_value, snap.shares_outstanding,
        bb.buyback_shrink_rate, auto.forecast_years,
    )

    market = snap.price
    diff = intrinsic_base / market - 1.0
    threshold = 0.15
    if diff > threshold:
        signal = "UNDERVALUED"
    elif diff < -threshold:
        signal = "OVERVALUED"
    else:
        signal = "FAIRLY VALUED"

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"  {ticker} — DCF Valuation Summary")
    print(f"{'='*60}")
    print(f"  Market price:                ${market:>12,.2f}")
    print(f"  Intrinsic price (base):      ${intrinsic_base:>12,.2f}")
    print(f"  Intrinsic price (BB-adj):    ${intrinsic_bb:>12,.2f}")
    print(f"  Upside / (Downside):          {diff*100:>+11.1f}%")
    print(f"  Signal (±{threshold*100:.0f}% band):          {signal}")
    print(f"{'='*60}")

    print(f"\n  Assumptions (auto-derived)")
    print(f"  {'─'*40}")
    print(f"  Forecast horizon:     {auto.forecast_years} years")
    print(f"  Start revenue growth: {auto.start_revenue_growth*100:.2f}%")
    print(f"  Terminal growth:      {auto.terminal_growth*100:.2f}%")
    print(f"  Terminal ROIC:        {auto.terminal_roic*100:.2f}%")
    print(f"  Terminal op margin:   {auto.terminal_operating_margin*100:.2f}%")
    print(f"  Fallback reinv rate:  {auto.fallback_reinvestment_rate*100:.2f}%")
    print(f"  Rd (pre-tax):         {auto.rd_pre_tax*100:.2f}%")

    print(f"\n  Cost of Capital")
    print(f"  {'─'*40}")
    print(f"  Risk-free (^TNX):     {rf*100:.2f}%")
    print(f"  ERP:                  {erp*100:.2f}%")
    print(f"  Beta raw (vendor):    {beta_pick.beta_raw if beta_pick.beta_raw is not None else 'N/A'}")
    print(f"  Beta used ({beta_pick.method}): {beta_pick.beta_used:.2f}")
    if beta_pick.damodaran_industry:
        print(f"  Damodaran industry:   {beta_pick.damodaran_industry}")
    print(f"  WACC:                 {computed_wacc*100:.2f}%")

    # --- Exports ---
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{ticker}_{ts}"
    json_path = outdir / f"{run_id}_result.json"
    csv_path = outdir / f"{run_id}_forecast.csv"
    png_path = outdir / f"{run_id}_heatmap.png"

    # Forecast CSV
    df_forecast = pd.DataFrame([{
        "year": y.year,
        "revenue": y.revenue,
        "op_margin": y.op_margin,
        "ebit": y.ebit,
        "nopat": y.nopat,
        "reinvestment": y.reinvestment,
        "fcff": y.fcff,
    } for y in forecast_years_data])
    df_forecast.to_csv(csv_path, index=False)

    # Result JSON
    payload = {
        "ticker": ticker,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "market_price": market,
        "intrinsic_price_base": intrinsic_base,
        "intrinsic_price_buyback_adj": intrinsic_bb,
        "signal": signal,
        "inputs": {
            "rf": rf, "erp": erp,
            "beta_raw": beta_pick.beta_raw, "beta_used": beta_pick.beta_used,
            "beta_method": beta_pick.method, "wacc": computed_wacc,
            "forecast_years": auto.forecast_years,
            "start_growth": auto.start_revenue_growth,
            "terminal_g": auto.terminal_growth,
            "terminal_roic": auto.terminal_roic,
            "terminal_margin": auto.terminal_operating_margin,
            "rd_pre_tax": auto.rd_pre_tax,
        },
        "dcf_result": asdict(res),
        "balance_bridge": {"cash": cash, "debt": total_debt, "shares": snap.shares_outstanding},
        "buybacks": {"net_buybacks": bb.net_buybacks, "shrink_rate": bb.buyback_shrink_rate},
        "files": {"forecast_csv": str(csv_path), "result_json": str(json_path), "heatmap_png": str(png_path)},
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # --- Heatmap (wider grid, centred on market price) ---
    wacc_centre = computed_wacc
    wacc_vals = sorted({
        round(wacc_centre - 0.02, 4),
        round(wacc_centre - 0.01, 4),
        round(wacc_centre, 4),
        round(wacc_centre + 0.01, 4),
        round(wacc_centre + 0.02, 4),
    })
    g_vals = [0.015, 0.020, 0.025, 0.030, 0.035]

    matrix = build_wacc_g_matrix(
        hist=hist, snap=snap, cap_struct=cap_struct,
        forecast_years=auto.forecast_years,
        start_growth=auto.start_revenue_growth,
        terminal_roic=auto.terminal_roic,
        terminal_margin=auto.terminal_operating_margin,
        fallback_reinvestment_rate=auto.fallback_reinvestment_rate,
        wacc_values=wacc_vals, g_values=g_vals,
        buyback_shrink_rate=bb.buyback_shrink_rate,
    )

    plot_heatmap(
        matrix,
        HeatmapSpec(
            title=f"{ticker} — Intrinsic Price Sensitivity (buyback-adjusted)",
            x_labels=[f"{g*100:.1f}%" for g in g_vals],
            y_labels=[f"{w*100:.1f}%" for w in wacc_vals],
            market_price=market,
        ),
        save_path=str(png_path),
        show=not args.no_plot,
    )

    print(f"\n  Saved: {json_path}")
    print(f"  Saved: {csv_path}")
    print(f"  Saved: {png_path}")


if __name__ == "__main__":
    main()
