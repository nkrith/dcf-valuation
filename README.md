# DCF Valuation Engine

Automated intrinsic-value calculator using a Discounted Cash Flow (FCFF) model with ROIC-consistent terminal value, buyback adjustment, and sensitivity analysis.

Runs on **any US-listed ticker** — pulls live data from Yahoo Finance, auto-generates assumptions, and produces a valuation summary with sensitivity heatmap in under 10 seconds.

## Quick Start

```bash
git clone https://github.com/nkrith/dcf-valuation.git
cd dcf-valuation
pip install -r requirements.txt
python -m dcf AAPL
```

## Sample Output

```
============================================================
  AAPL — DCF Valuation Summary
============================================================
  Market price:                $      251.49
  Intrinsic price (base):      $      186.32
  Intrinsic price (BB-adj):    $      210.66
  Upside / (Downside):                -25.9%
  Signal (±15% band):          OVERVALUED
============================================================

  Assumptions (auto-derived)
  ────────────────────────────────────────
  Forecast horizon:     10 years
  Start revenue growth: 6.43%
  Terminal growth:      2.50%
  Terminal ROIC:        20.50%
  Terminal op margin:   30.37%

  Cost of Capital
  ────────────────────────────────────────
  Risk-free (^TNX):     4.33%
  ERP:                  5.00%
  Beta used (mean_revert): 1.08
  WACC:                 9.59%
```

Each run also produces:
- **JSON** — full valuation summary with all inputs and intermediate results
- **CSV** — year-by-year operating forecast (revenue, margins, NOPAT, reinvestment, FCFF)
- **PNG** — WACC x terminal-growth sensitivity heatmap, colour-coded green/red against market price

## Batch Mode — Test Across Multiple Tickers

```bash
# Quick test: 10 large-caps
python -m dcf.batch

# Specific tickers
python -m dcf.batch AAPL MSFT GOOG NVDA AMZN

# First 50 S&P 500 constituents
python -m dcf.batch --sp500 --limit 50

# Full S&P 500 (~30 min)
python -m dcf.batch --sp500
```

Batch mode produces a summary CSV in `outputs/` with ticker, market price, intrinsic price, upside, signal, and all key assumptions.

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `ticker` | `AAPL` | Ticker symbol |
| `--erp` | `0.05` | Equity risk premium (decimal) |
| `--beta-method` | `mean_revert` | `raw`, `mean_revert`, or `damodaran` |
| `--outdir` | `outputs` | Directory for result files |
| `--no-plot` | off | Suppress plot window (PNG still saved) |
| `-v` | off | Verbose / debug logging |

## How It Works

1. **Pulls live data** — market price, financial statements, and the 10-Year Treasury yield via Yahoo Finance
2. **Derives operating metrics** — revenue, EBIT, NOPAT, reinvestment, FCFF, tax rate, and sales-to-capital ratio
3. **Auto-generates assumptions** — forecast horizon, revenue growth path, terminal margin, terminal ROIC, and cost of debt, calibrated by company size and profitability
4. **Selects beta** — raw, Blume mean-reverted, or bottom-up from Damodaran's industry dataset (unlevered, cash-corrected, relevered to the company's capital structure)
5. **Computes WACC** — CAPM cost of equity + after-tax cost of debt, market-value weighted
6. **Builds an operating forecast** — revenue growth and margins converge linearly to terminal assumptions; reinvestment derived from sales-to-capital ratio or g/ROIC economics
7. **Values the firm** — PV of explicit FCFF + ROIC-consistent terminal value (reinvestment rate = g / ROIC), bridged from enterprise value to equity value to price per share
8. **Adjusts for buybacks** — estimates implied share-count reduction from historical repurchase activity
9. **Runs sensitivity analysis** — generates a WACC x terminal-growth heatmap centred on computed WACC, colour-coded against market price
10. **Exports results** — JSON, CSV, and heatmap PNG

## Architecture

```
dcf/
├── __init__.py          # Package metadata
├── __main__.py          # python -m dcf entry point
├── app.py               # CLI + orchestration (single ticker)
├── batch.py             # Batch runner (multi-ticker / S&P 500)
├── yf_data.py           # Yahoo Finance data layer
├── normalize.py         # Historical operating metrics
├── auto_assumptions.py  # Ticker-agnostic assumption generator
├── beta_policy.py       # Beta selection (raw / mean-revert / Damodaran)
├── buybacks.py          # Share repurchase analysis
├── wacc.py              # CAPM + WACC calculation
├── forecasting.py       # Operating forecast builder + terminal value
├── dcf_engine.py        # DCF valuation (simple + operating-model)
├── fcff_forecast.py     # FCFF fade-growth forecast
├── fcff_history.py      # Historical FCF extraction
├── sensitivity.py       # WACC x g sensitivity table
├── visuals.py           # Heatmap plotting
├── series_tools.py      # Numeric helpers
├── io.py                # File I/O utilities
└── types.py             # Core dataclasses
```

## Key Design Decisions

- **ROIC-consistent terminal value** — reinvestment rate = g / ROIC, preventing the common mistake of implying infinite returns on capital in perpetuity
- **Economics-based reinvestment fallback** — when historical reinvestment is negative (D&A > CapEx, common in mature tech), the model derives a reinvestment rate from g / ROIC rather than using a meaningless historical ratio
- **Margin-band ROIC proxy** — avoids fragile invested-capital parsing from yfinance; maps operating margin bands to reasonable ROIC estimates
- **Blume mean-reversion on beta** — vendor regression betas are noisy; mean-reverting toward 1.0 with a clamp produces more stable discount rates
- **Buyback adjustment** — mid-period share-count reduction estimated from latest repurchase data, improving per-share valuation for heavy repurchasers
- **Sales-to-capital guardrails** — the ratio is discarded if outside [0.5, 4.0], falling back to NOPAT-based reinvestment

## Requirements

- Python 3.10+
- `yfinance`, `pandas`, `numpy`, `matplotlib`

## License

MIT
