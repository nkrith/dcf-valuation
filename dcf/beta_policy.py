from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

DAMODARAN_BETAS_XLS = "https://www.stern.nyu.edu/~adamodar/pc/datasets/betas.xls"


@dataclass(frozen=True)
class BetaResult:
    beta_raw: Optional[float]
    beta_used: float
    method: str
    damodaran_industry: Optional[str] = None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _mean_revert_beta(beta_raw: float) -> float:
    """Blume-style mean reversion: β_adj = 0.67 × β_raw + 0.33 × 1.0, clamped [0.6, 1.8]."""
    return _clamp(0.67 * beta_raw + 0.33, 0.6, 1.8)


@lru_cache(maxsize=1)
def _load_damodaran_betas() -> pd.DataFrame:
    """Load Damodaran's industry betas spreadsheet (updated ~annually)."""
    xls = pd.ExcelFile(DAMODARAN_BETAS_XLS)
    best_df = None

    for name in xls.sheet_names:
        df = xls.parse(name)
        cols = [str(c).strip().lower() for c in df.columns]
        if any("industry" in c and "name" in c for c in cols):
            best_df = df
            break

    if best_df is None:
        best_df = xls.parse(xls.sheet_names[0])

    best_df.columns = [str(c).strip() for c in best_df.columns]
    return best_df


def _find_industry_row(df: pd.DataFrame, industry_guess: str) -> Optional[pd.Series]:
    """Fuzzy-match yfinance industry string to Damodaran 'Industry Name'."""
    if df is None or df.empty or not industry_guess:
        return None

    ind_col = None
    for c in df.columns:
        if "industry" in str(c).lower() and "name" in str(c).lower():
            ind_col = c
            break
    if ind_col is None:
        return None

    target = industry_guess.lower()

    def score(name: str) -> int:
        n = str(name).lower()
        tokens = [t for t in target.replace("&", " ").replace("/", " ").split() if len(t) > 2]
        return sum(1 for t in tokens if t in n)

    df2 = df[[ind_col]].dropna().copy()
    df2["__score__"] = df2[ind_col].apply(score)
    df2 = df2.sort_values("__score__", ascending=False)

    if df2.empty or int(df2.iloc[0]["__score__"]) <= 0:
        return None

    best_name = df2.iloc[0][ind_col]
    return df[df[ind_col] == best_name].iloc[0]


def _extract_unlevered_beta_corrected_for_cash(row: pd.Series) -> Optional[float]:
    """Extract the unlevered beta corrected for cash from a Damodaran row."""
    for c in row.index:
        lc = str(c).lower()
        if ("unlevered" in lc and "correct" in lc and "cash" in lc) or "unlevered beta corrected" in lc:
            try:
                v = float(row[c])
                if v == v:
                    return v
            except Exception:
                pass
    return None


def _relever_beta(beta_u: float, debt: float, equity_mkt: float, tax_rate: float) -> float:
    """β_L = β_U × (1 + (1 − T) × D/E)"""
    if equity_mkt <= 0:
        return beta_u
    d_over_e = max(0.0, debt) / equity_mkt
    return beta_u * (1.0 + (1.0 - tax_rate) * d_over_e)


def pick_beta(
    *,
    beta_raw: Optional[float],
    method: str,
    industry_guess: Optional[str],
    debt: float,
    equity_mkt: float,
    tax_rate: float,
) -> BetaResult:
    """
    Select beta using the chosen policy:
      'raw'         — use vendor beta (fallback 1.0)
      'mean_revert' — Blume adjustment + clamp
      'damodaran'   — bottom-up from Damodaran, relevered; falls back to mean_revert
    """
    beta_raw_val = beta_raw if beta_raw is not None and beta_raw == beta_raw else None

    if method == "raw":
        return BetaResult(beta_raw=beta_raw_val, beta_used=float(beta_raw_val or 1.0), method="raw")

    if method == "mean_revert":
        if beta_raw_val is None:
            return BetaResult(beta_raw=None, beta_used=1.0, method="mean_revert(fallback_1.0)")
        return BetaResult(beta_raw=beta_raw_val, beta_used=_mean_revert_beta(beta_raw_val), method="mean_revert")

    if method == "damodaran":
        try:
            df = _load_damodaran_betas()
            row = _find_industry_row(df, industry_guess or "")
            if row is not None:
                beta_u = _extract_unlevered_beta_corrected_for_cash(row)
                if beta_u is not None:
                    beta_l = _clamp(_relever_beta(beta_u, debt, equity_mkt, tax_rate), 0.6, 2.0)
                    industry_name = None
                    for c in row.index:
                        if "industry" in str(c).lower() and "name" in str(c).lower():
                            industry_name = str(row[c])
                            break
                    log.info("Damodaran beta: industry=%s  β_U=%.3f  β_L=%.3f",
                             industry_name, beta_u, beta_l)
                    return BetaResult(
                        beta_raw=beta_raw_val, beta_used=beta_l,
                        method="damodaran(unlevered_cash_corrected->relevered)",
                        damodaran_industry=industry_name,
                    )
        except Exception as exc:
            log.warning("Damodaran beta lookup failed: %s", exc)

        # Fallback
        if beta_raw_val is None:
            return BetaResult(beta_raw=None, beta_used=1.0, method="damodaran_failed->fallback_1.0")
        return BetaResult(beta_raw=beta_raw_val, beta_used=_mean_revert_beta(beta_raw_val),
                          method="damodaran_failed->mean_revert")

    # Unknown method
    if beta_raw_val is None:
        return BetaResult(beta_raw=None, beta_used=1.0, method="unknown_method->fallback_1.0")
    return BetaResult(beta_raw=beta_raw_val, beta_used=_mean_revert_beta(beta_raw_val),
                      method="unknown_method->mean_revert")
