"""Unit tests for the short-signal backtest helpers in strategy.py.

Run from SeronaBackTest/ with:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategy import (  # noqa: E402
    STRATEGIES, factor_for_condition, normalize_manu,
    simulate, sharpe, max_drawdown, basket_size_daily,
    portfolio_turnover_annualized, validate_prices, clean_prices,
)


# -----------------------------------------------------------------------------
# normalize_manu
# -----------------------------------------------------------------------------

def test_normalize_manu_basics():
    assert normalize_manu("Boston Scientific Corporation") == "BOSTON SCIENTIFIC"
    assert normalize_manu("ABBOTT, INC.") == "ABBOTT"
    assert normalize_manu("  Olympus Co., Ltd.  ") == "OLYMPUS"


def test_normalize_manu_legal_forms():
    # All these should boil down to the same thing
    forms = ["Acme Inc", "Acme, INC.", "ACME LLC", "Acme GmbH", "Acme S.A.", "Acme Limited"]
    normed = {normalize_manu(f) for f in forms}
    assert normed == {"ACME"}


def test_normalize_manu_unicode_and_punct():
    assert normalize_manu("Sanofi-Aventis Söder AB") == "SANOFI AVENTIS SODER"
    assert normalize_manu("(JJGC) S.A.") == "JJGC"


def test_normalize_manu_handles_none_and_empty():
    assert normalize_manu(None) is None
    assert normalize_manu("") is None
    assert normalize_manu(123) is None
    assert normalize_manu("INC.") is None  # legal form only → empty after strip


# -----------------------------------------------------------------------------
# STRATEGIES + factor_for_condition (note: short fires when prob > threshold)
# -----------------------------------------------------------------------------

@pytest.fixture
def small_sig():
    return pd.DataFrame({
        "ticker":         ["A", "A", "B", "B"],
        "signal_date":    pd.to_datetime(["2025-01-02"] * 4),
        "prob_class_0":   [0.10, 0.90, 0.50, 0.20],
        "prob_class_1":   [0.20, 0.05, 0.30, 0.95],
        "prob_class_2":   [0.70, 0.05, 0.20, 0.85],
    })


def test_strategy_single_prob(small_sig):
    p0 = small_sig["prob_class_0"].to_numpy()
    p1 = small_sig["prob_class_1"].to_numpy()
    p2 = small_sig["prob_class_2"].to_numpy()
    np.testing.assert_array_equal(STRATEGIES["p0 > t"](p0, p1, p2, 0.5), [False, True, False, False])
    np.testing.assert_array_equal(STRATEGIES["p1 > t"](p0, p1, p2, 0.5), [False, False, False, True])
    np.testing.assert_array_equal(STRATEGIES["p2 > t"](p0, p1, p2, 0.5), [True, False, False, True])


def test_strategy_or_conditions(small_sig):
    p0 = small_sig["prob_class_0"].to_numpy()
    p1 = small_sig["prob_class_1"].to_numpy()
    p2 = small_sig["prob_class_2"].to_numpy()
    # any prob > 0.6 → row 0 (p2=0.70), row 1 (p0=0.90), row 3 (p1=0.95, p2=0.85)
    np.testing.assert_array_equal(
        STRATEGIES["any prob > t"](p0, p1, p2, 0.6),
        [True, True, False, True],
    )


def test_factor_for_condition_single(small_sig):
    pd.testing.assert_series_equal(
        factor_for_condition(small_sig, "p0 > t"),
        small_sig["prob_class_0"],
    )


def test_factor_for_condition_or_uses_max(small_sig):
    # any-prob condition: factor = max(p0, p1, p2). (factor > t) must equal (any prob > t).
    factor = factor_for_condition(small_sig, "any prob > t").to_numpy()
    p0 = small_sig["prob_class_0"].to_numpy()
    p1 = small_sig["prob_class_1"].to_numpy()
    p2 = small_sig["prob_class_2"].to_numpy()
    for t in [0.1, 0.3, 0.5, 0.8]:
        np.testing.assert_array_equal(factor > t, (p0 > t) | (p1 > t) | (p2 > t))


def test_factor_for_condition_unknown_raises(small_sig):
    with pytest.raises(ValueError):
        factor_for_condition(small_sig, "bogus")


# -----------------------------------------------------------------------------
# simulate — synthetic prices, hand-checkable arithmetic
# -----------------------------------------------------------------------------

@pytest.fixture
def synthetic_prices():
    """10 trading days, ticker A drifts +1%/day, B flat, SPY +0.1%/day."""
    dates = pd.bdate_range("2025-01-02", periods=10)
    a = 100 * (1.01 ** np.arange(10))
    b = np.full(10, 50.0)
    spy = 400 * (1.001 ** np.arange(10))
    return pd.DataFrame({"A": a, "B": b, "SPY": spy}, index=dates)


def _signal(ticker, date, p0=0.0, p1=0.0, p2=0.99):
    return {"ticker": ticker, "signal_date": pd.Timestamp(date),
            "prob_class_0": p0, "prob_class_1": p1, "prob_class_2": p2}


def test_simulate_single_signal_daily_marks(synthetic_prices):
    """Signal fires on day 0 with entry_delay=1, hold=1.
    Position is short A on day 2 only (P&L from close-of-day-1 to close-of-day-2).
    A drifts +1%/day, so short_book on day 2 = -0.01. All other days are 0."""
    sig = pd.DataFrame([_signal("A", "2025-01-02")])
    s, short_book, _ = simulate(sig, synthetic_prices, "p2 > t", 0.5, 1, 1)
    assert len(s) == 1
    # Per-trade compound return (matches old test): same 1-day case
    assert s["short_ret"].iloc[0] == pytest.approx(-0.01, abs=1e-9)
    # Daily series spans all trading days
    assert len(short_book) == len(synthetic_prices)
    # Only day 2 has non-zero P&L
    nonzero = short_book[short_book != 0]
    assert len(nonzero) == 1
    assert nonzero.iloc[0] == pytest.approx(-0.01, abs=1e-9)
    assert nonzero.index[0] == synthetic_prices.index[2]


def test_simulate_balanced_is_50_50_spy_plus_short(synthetic_prices):
    """Balanced = 0.5*short_book + 0.5*SPY. SPY drifts +0.1%/day.
    On the held day: 0.5*(-0.01) + 0.5*0.001 = -0.0045.
    On other days: 0.5*0 + 0.5*0.001 = +0.0005."""
    sig = pd.DataFrame([_signal("A", "2025-01-02")])
    _, short_book, balanced = simulate(sig, synthetic_prices, "p2 > t", 0.5, 1, 1)
    # Day 2 (held)
    expected_held = 0.5 * (-0.01) + 0.5 * 0.001
    assert balanced.iloc[2] == pytest.approx(expected_held, abs=1e-9)
    # Day 1 (no short, just SPY)
    assert balanced.iloc[1] == pytest.approx(0.5 * 0.001, abs=1e-9)


def test_simulate_multi_ticker_basket_is_equal_weight_average(synthetic_prices):
    """A and B both shorted on day 2. A daily_ret=+1%, B daily_ret=0. Basket = avg = 0.005,
    short_book = -0.005."""
    sigs = pd.DataFrame([
        _signal("A", "2025-01-02"),
        _signal("B", "2025-01-02"),
    ])
    _, short_book, _ = simulate(sigs, synthetic_prices, "p2 > t", 0.5, 1, 1)
    assert short_book.iloc[2] == pytest.approx(-0.005, abs=1e-9)


def test_simulate_holding_period_marks_daily(synthetic_prices):
    """hold_days=5: position active days 2..6. Each day's basket return = -1% (A drifts +1%/day).
    Sum of daily returns = -0.05. Per-trade compound stock_ret = 1.01^5 - 1."""
    sig = pd.DataFrame([_signal("A", "2025-01-02")])
    s, short_book, _ = simulate(sig, synthetic_prices, "p2 > t", 0.5, 1, 5)
    nonzero = short_book[short_book != 0]
    assert len(nonzero) == 5
    assert all(v == pytest.approx(-0.01, abs=1e-12) for v in nonzero.values)
    assert short_book.sum() == pytest.approx(-0.05, abs=1e-9)
    # Per-trade view keeps the compound figure
    assert s["short_ret"].iloc[0] == pytest.approx(-(1.01 ** 5 - 1), abs=1e-9)


def test_simulate_no_signals_returns_empty(synthetic_prices):
    sig = pd.DataFrame([_signal("A", "2025-01-02", p2=0.1)])
    s, daily, daily_bal = simulate(sig, synthetic_prices, "p2 > t", 0.5, 1, 1)
    assert s.empty and daily.empty and daily_bal.empty


def test_simulate_nan_entry_does_not_poison_basket(synthetic_prices):
    """If a ticker's price is NaN at entry, the position contributes NaN that day; nanmean treats
    it as 'no observation' so the basket return falls back to 0 (no spurious P&L). The per-trade
    table drops the NaN row, but the daily series still spans all trading days."""
    px = synthetic_prices.copy()
    px.loc[px.index[1], "A"] = np.nan
    sig = pd.DataFrame([_signal("A", "2025-01-02")])
    s, short_book, _ = simulate(sig, px, "p2 > t", 0.5, 1, 1)
    assert s.empty                             # NaN trade dropped from per-trade table
    assert len(short_book) == len(px)          # daily series still full length
    assert short_book.sum() == pytest.approx(0.0, abs=1e-12)


def test_simulate_overlapping_signals_collapse_to_binary_basket(synthetic_prices):
    """Two signals for the SAME ticker on the same day shouldn't double-count: the basket is binary
    membership, not stacked by signal count."""
    sigs = pd.DataFrame([
        _signal("A", "2025-01-02"),
        _signal("A", "2025-01-02"),
    ])
    _, short_book_dup, _ = simulate(sigs, synthetic_prices, "p2 > t", 0.5, 1, 1)
    sig_one = pd.DataFrame([_signal("A", "2025-01-02")])
    _, short_book_one, _ = simulate(sig_one, synthetic_prices, "p2 > t", 0.5, 1, 1)
    pd.testing.assert_series_equal(short_book_dup, short_book_one)


# -----------------------------------------------------------------------------
# Strategy-rule tests — explicitly verifying the stated behaviour:
#   "If a condition is met for ticker T on day d, short T from d+entry_delay
#    to d+entry_delay+hold_days. The position is held even if the condition
#    becomes FALSE on subsequent days. Each firing creates its own hold window
#    [t, t+hold_days]; overlapping firings extend the active window. Short-book
#    return = −mean(stock returns of held tickers). Balanced = 0.5 × short_book
#    + 0.5 × SPY_return."
# -----------------------------------------------------------------------------

def test_rule_position_holds_full_window_after_a_one_off_signal(synthetic_prices):
    """A single signal on day 0 → position active EXACTLY on days [trade_date+1, trade_date+hold_days].
    Days outside that window contribute 0 (no condition active = no position)."""
    sig = pd.DataFrame([_signal("A", "2025-01-02", p2=0.99)])  # only one firing
    _, short_book, _ = simulate(sig, synthetic_prices, "p2 > t", 0.5, entry_delay=1, hold_days=5)

    nz = short_book[short_book != 0]
    # Active days = [day_2, day_3, day_4, day_5, day_6]  (entry day 1 → P&L on days 2..6)
    assert list(nz.index) == [synthetic_prices.index[i] for i in (2, 3, 4, 5, 6)]
    # And those are the ONLY non-zero days — confirming no position before entry, none after exit
    assert short_book.iloc[0] == 0 and short_book.iloc[1] == 0
    assert all(v == 0 for v in short_book.iloc[7:].values)


def test_rule_subsequent_low_prob_does_not_break_the_hold(synthetic_prices):
    """If on the day AFTER the firing the prob falls below threshold (no new firing), the position
    must still be held for the full window. We model this by providing only one signal row at d=0
    and no rows for subsequent days; simulate has no post-entry condition check, so the position
    persists for the full hold_days as required."""
    sigs = pd.DataFrame([
        _signal("A", "2025-01-02", p2=0.99),  # day 0: condition TRUE → enter day 1, held days 2..4
        # No rows for days 1, 2 — i.e. the condition is implicitly FALSE on those days.
    ])
    _, short_book, _ = simulate(sigs, synthetic_prices, "p2 > t", 0.5, entry_delay=1, hold_days=3)
    nz = short_book[short_book != 0]
    assert len(nz) == 3
    assert nz.index[0] == synthetic_prices.index[2]
    assert nz.index[-1] == synthetic_prices.index[4]


def test_rule_consecutive_firings_extend_the_active_window(synthetic_prices):
    """Two firings for the same ticker on consecutive days. Each creates its own [t+1, t+hold] window,
    and the basket aggregates by union (binary membership)."""
    sigs = pd.DataFrame([
        _signal("A", "2025-01-02"),  # entry day 1 → active days 2,3,4   (hold=3)
        _signal("A", "2025-01-03"),  # entry day 2 → active days 3,4,5
    ])
    _, short_book, _ = simulate(sigs, synthetic_prices, "p2 > t", 0.5, entry_delay=1, hold_days=3)
    nz = short_book[short_book != 0]
    # union = days 2,3,4,5  →  4 active days (NOT 6 = 3 + 3)
    assert list(nz.index) == [synthetic_prices.index[i] for i in (2, 3, 4, 5)]


def test_rule_short_book_is_neg_mean_of_held_stock_returns(synthetic_prices):
    """short_book[d] = -1 * mean(daily return of each held ticker that day).
    On day 2: A held (r=+1%) and B held (r=0). mean=+0.5% → short_book=-0.5%."""
    sigs = pd.DataFrame([
        _signal("A", "2025-01-02"),
        _signal("B", "2025-01-02"),
    ])
    _, short_book, _ = simulate(sigs, synthetic_prices, "p2 > t", 0.5, entry_delay=1, hold_days=1)
    a_ret = synthetic_prices["A"].iloc[2] / synthetic_prices["A"].iloc[1] - 1
    b_ret = synthetic_prices["B"].iloc[2] / synthetic_prices["B"].iloc[1] - 1
    assert short_book.iloc[2] == pytest.approx(-(a_ret + b_ret) / 2, abs=1e-12)


def test_rule_balanced_is_half_short_plus_half_spy_every_day(synthetic_prices):
    """For every trading day d: balanced[d] == 0.5 * short_book[d] + 0.5 * SPY_daily_return[d].
    Holds even on days where short_book=0 (then balanced is just half SPY's return)."""
    sigs = pd.DataFrame([_signal("A", "2025-01-02")])
    _, short_book, balanced = simulate(sigs, synthetic_prices, "p2 > t", 0.5, entry_delay=1, hold_days=2)
    spy_ret = synthetic_prices["SPY"].pct_change(fill_method=None).fillna(0)
    expected = 0.5 * short_book + 0.5 * spy_ret
    pd.testing.assert_series_equal(balanced, expected, check_names=False, atol=1e-12)


def test_rule_no_signal_after_window_means_zero_pnl(synthetic_prices):
    """After the last position exits, the short book carries no P&L — strict zero on every day past
    the last hold's exit, irrespective of any future market moves."""
    sigs = pd.DataFrame([_signal("A", "2025-01-02")])
    _, short_book, _ = simulate(sigs, synthetic_prices, "p2 > t", 0.5, entry_delay=1, hold_days=2)
    # Position exits at trade_date+hold = day 1+2 = day 3. Days 4 onward must be exactly 0.
    assert all(v == 0.0 for v in short_book.iloc[4:].values)


def test_exit_threshold_validation_must_be_below_entry(synthetic_prices):
    sig = pd.DataFrame([_signal("A", "2025-01-02")])
    with pytest.raises(ValueError, match="exit_threshold"):
        simulate(sig, synthetic_prices, "p2 > t", 0.5, 1, 5, exit_threshold=0.5)
    with pytest.raises(ValueError, match="exit_threshold"):
        simulate(sig, synthetic_prices, "p2 > t", 0.5, 1, 5, exit_threshold=0.6)


def test_exit_threshold_zero_is_a_no_op(synthetic_prices):
    """exit_threshold=0 must yield identical results to the original (no early exit) since
    probabilities are non-negative — `factor < 0` can never fire."""
    sig = pd.DataFrame([_signal("A", "2025-01-02")])
    _, sb_no_exit, _ = simulate(sig, synthetic_prices, "p2 > t", 0.5, 1, 5)
    _, sb_zero_exit, _ = simulate(sig, synthetic_prices, "p2 > t", 0.5, 1, 5, exit_threshold=0.0)
    pd.testing.assert_series_equal(sb_no_exit, sb_zero_exit)


def test_exit_threshold_closes_position_early(synthetic_prices):
    """Entry at signal day 0 (p2=0.99); on signal day 2 the prob drops below exit_threshold (p2=0.05).
    Natural exit would be entry_idx (1) + hold (10) = day 11 — but synthetic_prices has 10 days, and we
    want a clear early-exit case. Use hold_days=8 so natural exit is index 9; early exit triggers at
    day 2 + entry_delay 1 = index 3, well before. Active days [2, 3]."""
    sigs = pd.DataFrame([
        _signal("A", "2025-01-02", p2=0.99),  # ENTRY: factor 0.99 > 0.5
        _signal("A", "2025-01-06", p2=0.05),  # EXIT TRIGGER on day 2: factor 0.05 < exit_thresh 0.10
    ])
    s, short_book, _ = simulate(
        sigs, synthetic_prices, "p2 > t", threshold=0.5, entry_delay=1, hold_days=8,
        exit_threshold=0.10,
    )
    # Per-trade exit_date should be the early-exit trading day, not the natural one
    # (entry signal at index 0 → trade_date = index 1; exit signal at index 2 → exit_date = index 3)
    assert s["trade_date"].iloc[0] == synthetic_prices.index[1]
    assert s["exit_date"].iloc[0] == synthetic_prices.index[3]
    # Short book active on days 2 and 3 only (P&L from close-of-1 to close-of-3 is 2 days)
    nz = short_book[short_book != 0]
    assert list(nz.index) == [synthetic_prices.index[i] for i in (2, 3)]


def test_exit_threshold_after_natural_exit_does_nothing(synthetic_prices):
    """If the exit-trigger signal arrives AFTER the natural exit, the position closes at hold_days
    as normal (no shortening)."""
    sigs = pd.DataFrame([
        _signal("A", "2025-01-02", p2=0.99),  # entry
        _signal("A", "2025-01-09", p2=0.01),  # exit trigger ARRIVES after natural exit (day 5+)
    ])
    s, short_book, _ = simulate(
        sigs, synthetic_prices, "p2 > t", threshold=0.5, entry_delay=1, hold_days=3,
        exit_threshold=0.10,
    )
    # Natural exit at index 1 + 3 = 4. Active days [2, 3, 4].
    assert s["exit_date"].iloc[0] == synthetic_prices.index[4]
    nz = short_book[short_book != 0]
    assert list(nz.index) == [synthetic_prices.index[i] for i in (2, 3, 4)]


def test_exit_threshold_only_first_trigger_after_entry_matters(synthetic_prices):
    """Multiple later signals that are below exit_threshold — only the FIRST one (chronologically)
    after the entry signal triggers the early exit."""
    sigs = pd.DataFrame([
        _signal("A", "2025-01-02", p2=0.99),  # entry day 0
        _signal("A", "2025-01-03", p2=0.05),  # exit trigger day 1 (uses this one)
        _signal("A", "2025-01-06", p2=0.02),  # later trigger — ignored
    ])
    s, short_book, _ = simulate(
        sigs, synthetic_prices, "p2 > t", threshold=0.5, entry_delay=1, hold_days=8,
        exit_threshold=0.10,
    )
    # First trigger at signal index 1 → exit at trading index 1 + 1 = 2
    assert s["exit_date"].iloc[0] == synthetic_prices.index[2]
    # Active days: only index 2 (entered close-of-1, exited close-of-2 → 1 day P&L)
    nz = short_book[short_book != 0]
    assert list(nz.index) == [synthetic_prices.index[2]]


def test_exit_threshold_per_signal_independent_windows(synthetic_prices):
    """Two entry signals on the same ticker: an exit trigger between them closes the FIRST
    position; the SECOND entry creates a fresh independent position."""
    sigs = pd.DataFrame([
        _signal("A", "2025-01-02", p2=0.99),  # entry #1 at signal index 0 → trade index 1
        _signal("A", "2025-01-03", p2=0.05),  # exit trigger for #1 (first exit > 0)
        _signal("A", "2025-01-08", p2=0.99),  # entry #2 at signal index 4 → trade index 5
    ])
    s, short_book, _ = simulate(
        sigs, synthetic_prices, "p2 > t", threshold=0.5, entry_delay=1, hold_days=2,
        exit_threshold=0.10,
    )
    # Two trade rows
    assert len(s) == 2
    # Trade 1: entry at idx 1, exit at idx 2 (early exit). Active day: 2.
    # Trade 2: entry at idx 5, exit at idx 7 (natural). Active days: 6, 7.
    nz = short_book[short_book != 0]
    assert list(nz.index) == [synthetic_prices.index[i] for i in (2, 6, 7)]


def test_rule_entry_delay_aligns_to_signal_plus_n_trading_days(synthetic_prices):
    """Spec: 'go short at close of d + entry_delay trading days. Exit at close of d + entry_delay + hold_days.'

    Verify with entry_delay=3, hold_days=2:
    Signal_date = 2025-01-02 (= synthetic_prices.index[0]).
    Trade open at close of index[0+3] = index[3]. Position active days [4, 5] (hold=2).
    Exit at close of index[3+2] = index[5].
    """
    sigs = pd.DataFrame([_signal("A", "2025-01-02")])
    s, short_book, _ = simulate(sigs, synthetic_prices, "p2 > t", 0.5, entry_delay=3, hold_days=2)
    # Per-trade view: trade_date = signal + entry_delay trading days
    assert s["trade_date"].iloc[0] == synthetic_prices.index[3]
    # Daily marks: P&L active on days 4 and 5 only
    nz = short_book[short_book != 0]
    assert list(nz.index) == [synthetic_prices.index[i] for i in (4, 5)]
    # Sum across active days = sum of -1 * 1d returns (A drifts +1%/day → each day -0.01)
    assert short_book.sum() == pytest.approx(-0.02, abs=1e-12)


def test_simulate_invalid_args_raise(synthetic_prices):
    sig = pd.DataFrame([_signal("A", "2025-01-02")])
    with pytest.raises(ValueError):
        simulate(sig, synthetic_prices, "p2 > t", 0.5, 0, 1)
    with pytest.raises(ValueError):
        simulate(sig, synthetic_prices, "p2 > t", 0.5, 1, 0)


# -----------------------------------------------------------------------------
# validate_prices / clean_prices
# -----------------------------------------------------------------------------

def test_validate_prices_clean(synthetic_prices):
    """The fixture has a flat ticker B by design (for the multi-ticker average test),
    so we don't assert stale_run_count == 0 here — just no spike-reverts or bad data."""
    issues = validate_prices(synthetic_prices)
    assert issues["all_nan_tickers"] == []
    assert issues["partial_nan_tickers"] == {}
    assert issues["non_positive_obs"] == 0
    assert issues["spike_revert_count"] == 0


def test_validate_prices_detects_spike_revert():
    dates = pd.bdate_range("2025-01-02", periods=6)
    df = pd.DataFrame({"X": [100, 100, 200, 100, 100, 100]}, index=dates)
    issues = validate_prices(df)
    assert issues["spike_revert_count"] >= 1


def test_validate_prices_detects_all_nan_and_partial():
    dates = pd.bdate_range("2025-01-02", periods=5)
    df = pd.DataFrame({
        "GOOD": [10, 10.5, 11, 11.5, 12],
        "GAP":  [10, np.nan, 11, np.nan, 12],
        "DEAD": [np.nan] * 5,
    }, index=dates)
    issues = validate_prices(df)
    assert issues["all_nan_tickers"] == ["DEAD"]
    assert "GAP" in issues["partial_nan_tickers"]


def test_validate_prices_rejects_unsorted_index():
    dates = pd.to_datetime(["2025-01-03", "2025-01-02"])
    df = pd.DataFrame({"X": [1.0, 2.0]}, index=dates)
    with pytest.raises(ValueError):
        validate_prices(df)


def test_clean_prices_drops_all_nan_and_ffills_gaps():
    dates = pd.bdate_range("2025-01-02", periods=5)
    df = pd.DataFrame({
        "GOOD": [10, 10.5, 11, 11.5, 12],
        "GAP":  [10, np.nan, 11, np.nan, 12],
        "DEAD": [np.nan] * 5,
    }, index=dates)
    cleaned = clean_prices(df, max_ffill=2)
    assert "DEAD" not in cleaned.columns
    # ffill should have filled GAP
    assert cleaned["GAP"].notna().all()


# -----------------------------------------------------------------------------
# sharpe, max_drawdown
# -----------------------------------------------------------------------------

def test_sharpe_positive():
    r = pd.Series([0.01, 0.02, -0.005, 0.015, 0.0])
    sr = sharpe(r)
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert sr == pytest.approx(expected, rel=1e-9)


def test_sharpe_zero_vol_returns_nan():
    assert np.isnan(sharpe(pd.Series([0.01, 0.01, 0.01])))


def test_sharpe_short_series_returns_nan():
    assert np.isnan(sharpe(pd.Series([0.01])))
    assert np.isnan(sharpe(pd.Series(dtype=float)))


def test_max_drawdown_known():
    """Cumulative peaks at 10, troughs at -5 → drawdown = -5 - 10 = -15."""
    cum = pd.Series([0, 5, 10, 4, -5, 0])
    assert max_drawdown(cum) == -15.0


def test_max_drawdown_monotonic_zero():
    assert max_drawdown(pd.Series([0, 1, 2, 3, 4])) == 0.0


def test_max_drawdown_empty_series():
    assert max_drawdown(pd.Series(dtype=float)) == 0.0


# -----------------------------------------------------------------------------
# basket_size_daily
# -----------------------------------------------------------------------------

def test_basket_size_daily_counts_unique_tickers(synthetic_prices):
    """A ticker counts once per day even if it has multiple overlapping signals."""
    sigs = pd.DataFrame([
        _signal("A", "2025-01-02"),
        _signal("A", "2025-01-02"),  # duplicate firing
        _signal("B", "2025-01-02"),
    ])
    s, _, _ = simulate(sigs, synthetic_prices, "p2 > t", 0.5, 1, 2)
    sizes = basket_size_daily(s, synthetic_prices.index)
    # Active days for both tickers: 2 and 3 (entry idx 1, hold 2). Each has 2 unique tickers.
    assert sizes.iloc[2] == 2
    assert sizes.iloc[3] == 2
    # Days 0, 1, 4+ should have 0
    assert sizes.iloc[0] == 0 and sizes.iloc[1] == 0
    assert sizes.iloc[4] == 0


def test_basket_size_daily_empty_returns_zeros(synthetic_prices):
    empty = pd.DataFrame(columns=["ticker", "trade_date", "exit_date"])
    sizes = basket_size_daily(empty, synthetic_prices.index)
    assert (sizes == 0).all()
    assert len(sizes) == len(synthetic_prices)


# -----------------------------------------------------------------------------
# portfolio_turnover_annualized
# -----------------------------------------------------------------------------

def test_turnover_zero_when_no_trades(synthetic_prices):
    empty = pd.DataFrame(columns=["ticker", "trade_date", "exit_date"])
    assert portfolio_turnover_annualized(empty, synthetic_prices.index) == 0.0


def test_turnover_buy_and_hold_is_zero(synthetic_prices):
    """One trade entered on day 0, held through the end of the window — 1 entry, 0 exits.
    Under the SEC `min(buys, sells)` convention the initial allocation doesn't count as
    turnover, so the result is exactly 0%."""
    s = pd.DataFrame([{
        "ticker": "A",
        "trade_date": synthetic_prices.index[0],
        "exit_date": synthetic_prices.index[-1],
    }])
    assert portfolio_turnover_annualized(s, synthetic_prices.index) == 0.0


def test_turnover_steady_state_matches_252_over_hold():
    """Simulated steady state: a new 5-day position opens each day, indefinitely.
    Expect annualized one-way turnover ≈ 252/hold × 100% = 5040%."""
    dates = pd.bdate_range("2025-01-02", periods=120)
    # Open A from day i to day i+5 for i in 0..100 — but use distinct tickers so each is its own slot
    rows = []
    for i in range(100):
        rows.append({"ticker": f"T{i:03d}",
                     "trade_date": dates[i],
                     "exit_date": dates[i + 5]})
    s = pd.DataFrame(rows)
    turnover = portfolio_turnover_annualized(s, dates)
    # Wide tolerance: hand-computation is sensitive to edge effects (warmup + cooldown)
    assert 3000 < turnover < 7000
