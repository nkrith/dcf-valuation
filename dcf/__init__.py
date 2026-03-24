"""
dcf — Automated intrinsic-value engine (FCFF DCF with ROIC-consistent terminal value).

Pulls live market data via yfinance, derives operating metrics from financial
statements, builds a multi-year operating forecast with margin convergence,
and computes intrinsic value per share with buyback adjustment and
WACC × terminal-growth sensitivity analysis.
"""

__version__ = "1.0.0"
