"""Pure helpers for the short-signal backtest. No streamlit / IO deps so this
module is importable from tests."""

from __future__ import annotations

import re
import unicodedata
from typing import Callable

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Manufacturer name normalization (mirrors ticker_mapping/join_tickers.py)
# -----------------------------------------------------------------------------

_LEGAL_FORMS = [
    "INCORPORATED", "INC", "CORPORATION", "CORP", "COMPANY", "CO",
    "LIMITED", "LTD", "LLC", "PLC", "GMBH", "AG", "SAS", "SA",
    "BV", "NV", "KG", "OY", "AB", "PTY", "SE", "SPA", "SRL",
]
_LEGAL_RE = re.compile(r"\b(?:" + "|".join(_LEGAL_FORMS) + r")\b", re.IGNORECASE)
_DOTTED_INITIALS_RE = re.compile(r"\b([A-Z])\.\s*([A-Z])\.\s*([A-Z])?\.?")
_PUNCT_RE = re.compile(r"[^A-Z0-9 ]+")
_WS_RE = re.compile(r"\s+")


def normalize_manu(s):
    if not isinstance(s, str):
        return None
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().upper()
    # Collapse dotted initialisms first (S.A. → SA, L.L.C. → LLC, S.P.A. → SPA),
    # otherwise the trailing-dot \b boundary prevents the legal-form regex from matching.
    s = _DOTTED_INITIALS_RE.sub(lambda m: "".join(g for g in m.groups() if g), s)
    s = _PUNCT_RE.sub(" ", s)
    s = _LEGAL_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s or None


# -----------------------------------------------------------------------------
# Strategies (boolean masks) and continuous factors per condition
# -----------------------------------------------------------------------------

STRATEGIES: dict[str, Callable] = {
    "p0 > t":              lambda p0, p1, p2, t: p0 > t,
    "p1 > t":              lambda p0, p1, p2, t: p1 > t,
    "p2 > t":              lambda p0, p1, p2, t: p2 > t,
    "p0 > t OR p1 > t":    lambda p0, p1, p2, t: (p0 > t) | (p1 > t),
    "p1 > t OR p2 > t":    lambda p0, p1, p2, t: (p1 > t) | (p2 > t),
    "any prob > t":        lambda p0, p1, p2, t: (p0 > t) | (p1 > t) | (p2 > t),
}


def factor_for_condition(sig: pd.DataFrame, condition: str) -> pd.Series:
    """Continuous factor underlying each condition (short triggers when factor > threshold).
    OR conditions reduce to max(...) because (a>t OR b>t) <=> max(a,b)>t."""
    p0, p1, p2 = sig["prob_class_0"], sig["prob_class_1"], sig["prob_class_2"]
    if condition == "p0 > t": return p0
    if condition == "p1 > t": return p1
    if condition == "p2 > t": return p2
    if condition == "p0 > t OR p1 > t": return pd.concat([p0, p1], axis=1).max(axis=1)
    if condition == "p1 > t OR p2 > t": return pd.concat([p1, p2], axis=1).max(axis=1)
    if condition == "any prob > t":     return pd.concat([p0, p1, p2], axis=1).max(axis=1)
    raise ValueError(f"unknown condition: {condition}")


# -----------------------------------------------------------------------------
# Price-data validation and cleaning
# -----------------------------------------------------------------------------

def validate_prices(prices: pd.DataFrame, *, spike_thresh: float = 0.25,
                    revert_thresh: float = 0.05, stale_run: int = 5) -> dict:
    """Audit a wide price DataFrame (rows=date, cols=ticker). Returns a dict of issues."""
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("prices must have a DatetimeIndex")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices index must be sorted ascending")

    issues = {}
    nan_pct = prices.isna().mean()
    issues["all_nan_tickers"] = nan_pct[nan_pct == 1.0].index.tolist()
    issues["partial_nan_tickers"] = nan_pct[(nan_pct > 0) & (nan_pct < 1)].to_dict()
    issues["non_positive_obs"] = int(((prices <= 0) & prices.notna()).sum().sum())

    # Spike-revert: a single-day move > spike_thresh whose 2-day round-trip is < revert_thresh.
    # Uses the geometric round-trip ratio shift(-1)/shift(1) so the test catches genuine bad ticks
    # (price spikes back to its prior level a day later) regardless of arithmetic-return asymmetry.
    r = prices.pct_change(fill_method=None)
    big_move = r.abs() > spike_thresh
    round_trip = (prices.shift(-1) / prices.shift(1) - 1).abs()
    revert = round_trip < revert_thresh
    suspect = big_move & revert
    issues["spike_revert_count"] = int(suspect.sum().sum())
    if issues["spike_revert_count"]:
        issues["spike_revert_locations"] = [
            (d.date().isoformat(), t) for d, t in suspect.stack()[lambda s: s].index
        ]

    diff_zero = prices.diff() == 0
    stale = diff_zero.rolling(stale_run).sum() == stale_run
    issues["stale_run_count"] = int(stale.sum().sum())

    return issues


def clean_prices(prices: pd.DataFrame, *, drop_all_nan: bool = True,
                 max_ffill: int = 2) -> pd.DataFrame:
    """Drop tickers with no data; forward-fill short NaN runs (default 2 days max).
    Long gaps stay NaN so they're caught downstream."""
    out = prices.copy()
    if drop_all_nan:
        all_nan = out.columns[out.isna().all()]
        out = out.drop(columns=all_nan)
    if max_ffill > 0:
        out = out.ffill(limit=max_ffill)
    return out


# -----------------------------------------------------------------------------
# Backtest
# -----------------------------------------------------------------------------

def simulate(sig: pd.DataFrame, prices: pd.DataFrame, condition: str,
             threshold: float, entry_delay: int, hold_days: int,
             balanced_weight: float = 0.5, exit_threshold: float = 0.0):
    """Run the short-basket backtest. Returns (per-trade df, short_book daily Series,
    balanced daily Series). Both daily Series are indexed by every trading day in `prices`,
    with 0 on no-position days.

    Trade rule:
      - When `condition(p0,p1,p2) > threshold` for ticker T at signal_date d (the AE date),
        go SHORT T at close of d + entry_delay trading days.
      - Position is held for at most `hold_days` trading days (the natural exit).
      - **Early exit**: if a subsequent ticker-day row for T has `factor < exit_threshold`,
        the position closes at close of (that signal_date + entry_delay), provided that's
        before the natural exit. `exit_threshold` must be < `threshold`. Setting it to 0
        effectively disables early exits (probabilities are non-negative).

    Daily portfolio math (true daily marks, no overlapping-position double-counting):
      held(d) = {tickers with at least one active short on day d}
      r[d, t] = prices[d, t] / prices[d-1, t] - 1                          (1-day adj-close return)
      short_book[d] = -(1/|held(d)|) * sum_{t in held(d)} r[d, t]          (equal-weight, sign-flipped)
      balanced[d]   = balanced_weight * short_book[d] + (1 - balanced_weight) * SPY_return[d]

    Sharpe = mean/std × √252 of the daily series.

    The per-trade DataFrame carries each signal's compound stock-return between its actual entry
    and actual exit (early-exit-aware) for transparency.
    """
    if entry_delay < 1:
        raise ValueError("entry_delay must be >= 1")
    if hold_days < 1:
        raise ValueError("hold_days must be >= 1")
    if not (0.0 <= balanced_weight <= 1.0):
        raise ValueError("balanced_weight must be in [0, 1]")
    if exit_threshold >= threshold:
        raise ValueError(f"exit_threshold ({exit_threshold}) must be strictly < entry threshold ({threshold})")

    def _empty(s_template):
        cols = list(s_template.columns) + ["trade_date", "exit_date", "stock_ret", "short_ret"]
        return pd.DataFrame(columns=cols), pd.Series(dtype=float), pd.Series(dtype=float)

    factor_vals = factor_for_condition(sig, condition).to_numpy()
    sig_with_factor = sig.assign(_factor=factor_vals)
    sig_with_factor = sig_with_factor[sig_with_factor["ticker"].isin(prices.columns)]

    # Pre-build per-ticker sorted exit-trigger dates (rows with factor < exit_threshold).
    # exit_threshold = 0 ⇒ never triggers (probs are >= 0), so this map is empty by default.
    exit_rows = sig_with_factor[sig_with_factor["_factor"] < exit_threshold]
    exit_dates_per_ticker = {
        tk: np.array(sorted(d.values), dtype="datetime64[ns]")
        for tk, d in exit_rows.groupby("ticker")["signal_date"]
    }

    s = sig_with_factor[sig_with_factor["_factor"] > threshold].copy()
    if s.empty:
        return _empty(sig)

    trading_days = prices.index
    trading_days_np = trading_days.values.astype("datetime64[ns]")
    n_days = len(trading_days)
    sig_dates_np = s["signal_date"].values.astype("datetime64[ns]")
    entry_idx_arr = np.searchsorted(trading_days_np, sig_dates_np, side="right") + (entry_delay - 1)
    valid = (entry_idx_arr + 1) < n_days  # need at least one day of P&L runway
    s = s.iloc[valid].copy()
    entry_idx_arr = entry_idx_arr[valid]
    sig_dates_np = sig_dates_np[valid]
    if s.empty:
        return _empty(sig)

    col_idx = {c: i for i, c in enumerate(prices.columns)}
    tk_col = np.array([col_idx[t] for t in s["ticker"]])

    # Per-signal exit index: min(natural_exit, first_early_exit_after_entry_signal)
    natural_exit_arr = np.minimum(entry_idx_arr + hold_days, n_days - 1)
    exit_idx_arr = natural_exit_arr.copy()
    for k, (tk, sd) in enumerate(zip(s["ticker"].values, sig_dates_np)):
        ex_dates = exit_dates_per_ticker.get(tk)
        if ex_dates is None or len(ex_dates) == 0:
            continue
        # First exit-trigger signal STRICTLY after the entry signal date
        i = np.searchsorted(ex_dates, sd, side="right")
        if i >= len(ex_dates):
            continue
        early_sig_date = ex_dates[i]
        cand_exit_idx = np.searchsorted(trading_days_np, early_sig_date, side="right") + (entry_delay - 1)
        cand_exit_idx = min(cand_exit_idx, n_days - 1)
        if cand_exit_idx < exit_idx_arr[k]:
            exit_idx_arr[k] = cand_exit_idx

    s["trade_date"] = trading_days[entry_idx_arr]
    s["exit_date"] = trading_days[exit_idx_arr]

    # Position matrix held[d, t]: per-signal, P&L on days (entry_idx, exit_idx]. Multiple signals
    # on the same ticker collapse via OR (binary basket membership, not stacked by count).
    held = np.zeros((n_days, prices.shape[1]), dtype=bool)
    for ti, ei, tk in zip(entry_idx_arr, exit_idx_arr, tk_col):
        if ei > ti:
            held[ti + 1: ei + 1, tk] = True

    daily_ret_mat = prices.pct_change(fill_method=None).to_numpy()
    held_ret = np.where(held, daily_ret_mat, np.nan)
    # nanmean warns on all-NaN rows (no positions held that day) — that's expected, fall back to 0.
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", category=RuntimeWarning)
        basket_ret = np.nanmean(held_ret, axis=1)
    basket_ret = np.where(np.isnan(basket_ret), 0.0, basket_ret)
    short_book_daily = pd.Series(-basket_ret, index=trading_days)

    if "SPY" in col_idx:
        spy_ret = prices["SPY"].pct_change(fill_method=None).fillna(0).to_numpy()
        balanced_daily = pd.Series(
            balanced_weight * short_book_daily.to_numpy() + (1 - balanced_weight) * spy_ret,
            index=trading_days,
        )
    else:
        balanced_daily = short_book_daily.copy()

    # Per-trade view: compound stock-return between actual entry and actual exit
    # (early-exit-aware, so the per-trade table reflects what really happened).
    px_vals = prices.to_numpy()
    p_entry = px_vals[entry_idx_arr, tk_col]
    p_exit = px_vals[exit_idx_arr, tk_col]
    s["stock_ret"] = (p_exit - p_entry) / p_entry
    s["short_ret"] = -s["stock_ret"]
    s = s.drop(columns=["_factor"]).dropna(subset=["stock_ret"])

    return s, short_book_daily, balanced_daily


# -----------------------------------------------------------------------------
# Risk metrics
# -----------------------------------------------------------------------------

def portfolio_turnover_annualized(s_trades: pd.DataFrame, trading_days: pd.DatetimeIndex,
                                  periods_per_year: int = 252) -> float:
    """Annualized one-way turnover (%) of the equal-weight short basket.

    Uses the SEC mutual-fund convention: ``turnover = min(buys, sells) / avg_NAV``, annualized.
    For our equal-weight binary basket, "1 unit of NAV" ↔ "1 name", so:
      - buys per period  = total entries (names added vs the prior day)
      - sells per period = total exits   (names dropped vs the prior day)
      - avg_NAV per period = avg basket size on active days

    The ``min(buys, sells)`` makes the initial-allocation entry not count as turnover
    (buy-and-hold has 1 entry, 0 exits → min = 0 → turnover = 0). In steady state where
    entries ≈ exits, the result matches the average formula.

    Reference: a strategy with ``hold_days = 40`` in steady state has ≈ 252/40 = 6.3
    round-trips per name per year, i.e. ~630% annualized one-way turnover.
    Returns 0.0 if the book never holds anything.
    """
    if s_trades.empty:
        return 0.0
    starts = s_trades["trade_date"].to_numpy()
    ends = s_trades["exit_date"].to_numpy()
    tickers = s_trades["ticker"].to_numpy()
    td = trading_days.to_numpy()

    prev_set: set = set()
    n_active = 0
    total_entries = 0
    total_exits = 0
    sum_basket = 0
    for d in td:
        active = (d > starts) & (d <= ends)
        cur = set(np.unique(tickers[active])) if active.any() else set()
        if cur or prev_set:
            n_active += 1
            sum_basket += len(cur)
            total_entries += len(cur - prev_set)
            total_exits += len(prev_set - cur)
        prev_set = cur

    if n_active == 0 or sum_basket == 0:
        return 0.0
    avg_basket = sum_basket / n_active
    period_turnover = min(total_entries, total_exits) / avg_basket
    return period_turnover * (periods_per_year / n_active) * 100.0


def basket_size_daily(s_trades: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Number of unique tickers in the short book on each trading day. A ticker counts once per day
    even if multiple of its signals are active simultaneously."""
    if s_trades.empty:
        return pd.Series(0, index=trading_days, dtype=int)
    td_arr = trading_days.to_numpy()
    starts = s_trades["trade_date"].to_numpy()
    ends = s_trades["exit_date"].to_numpy()
    tickers = s_trades["ticker"].to_numpy()
    sizes = np.zeros(len(td_arr), dtype=int)
    for i in range(len(td_arr)):
        active = (td_arr[i] > starts) & (td_arr[i] <= ends)
        if active.any():
            sizes[i] = np.unique(tickers[active]).size
    return pd.Series(sizes, index=trading_days)


def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized Sharpe assuming each datapoint is one trade-day (independence assumed).
    Returns NaN if series is empty or std is 0."""
    if returns.std(ddof=1) == 0 or len(returns) < 2:
        return float("nan")
    return returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year)


def max_drawdown(cum_pct: pd.Series) -> float:
    """Max drawdown of a cumulative-percent series. Returns 0 if series is empty/single point."""
    if len(cum_pct) < 2:
        return 0.0
    return float((cum_pct - cum_pct.cummax()).min())
