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
    STRATEGIES, factor_for_condition, factor_for_top_k, normalize_manu,
    simulate, simulate_top_k, sharpe, max_drawdown, basket_size_daily,
    portfolio_turnover_daily, validate_prices, clean_prices,
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
# factor_for_top_k — multi-class rules use MAX (same direction as the threshold
# strategy). Top-K shorts the K names with the HIGHEST factor per signal_date.
# -----------------------------------------------------------------------------

def test_factor_for_top_k_single_class_columns(small_sig):
    """Single-class rules return the corresponding probability column."""
    pd.testing.assert_series_equal(
        factor_for_top_k(small_sig, "p0"), small_sig["prob_class_0"], check_names=False,
    )
    pd.testing.assert_series_equal(
        factor_for_top_k(small_sig, "p1"), small_sig["prob_class_1"], check_names=False,
    )
    pd.testing.assert_series_equal(
        factor_for_top_k(small_sig, "p2"), small_sig["prob_class_2"], check_names=False,
    )


def test_factor_for_top_k_max_rules(small_sig):
    """Multi-class rules take the MAX across the listed columns. Top-K shorts the HIGHEST factor."""
    p0 = small_sig["prob_class_0"]
    p1 = small_sig["prob_class_1"]
    p2 = small_sig["prob_class_2"]
    pd.testing.assert_series_equal(
        factor_for_top_k(small_sig, "max(p0, p1)"),
        pd.concat([p0, p1], axis=1).max(axis=1),
    )
    pd.testing.assert_series_equal(
        factor_for_top_k(small_sig, "max(p0, p1, p2)"),
        pd.concat([p0, p1, p2], axis=1).max(axis=1),
    )


def test_factor_for_top_k_unknown_raises(small_sig):
    with pytest.raises(ValueError):
        factor_for_top_k(small_sig, "bogus")


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


def test_simulate_balanced_is_50_50_spy_plus_short_only_when_position_held(synthetic_prices):
    """Balanced book = 0.5*short + 0.5*SPY ONLY on days a short position is live.
    On no-position days both legs are flat — balanced = 0. SPY drifts +0.1%/day in
    this fixture, but the long leg is gated by basket activity so it only contributes
    on the day a short is actually held."""
    sig = pd.DataFrame([_signal("A", "2025-01-02")])
    _, short_book, balanced = simulate(sig, synthetic_prices, "p2 > t", 0.5, 1, 1)
    # Day 2 (held): 0.5 * (-0.01) + 0.5 * 0.001 = -0.0045
    expected_held = 0.5 * (-0.01) + 0.5 * 0.001
    assert balanced.iloc[2] == pytest.approx(expected_held, abs=1e-9)
    # Day 1 (no short): both legs at 0 → balanced = 0 (NOT 0.5 * SPY return).
    assert balanced.iloc[1] == pytest.approx(0.0, abs=1e-12)
    # Day 3+ (after exit): same — no position, no hedge, flat.
    for i in range(3, len(balanced)):
        assert balanced.iloc[i] == pytest.approx(0.0, abs=1e-12), f"day {i}"


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


def test_rule_balanced_gated_by_basket_activity(synthetic_prices):
    """Balanced = 0.5 * short_book + 0.5 * SPY ONLY on days the short book is live.
    Days with no held names → balanced = 0 (both legs flat, no exposure)."""
    sigs = pd.DataFrame([_signal("A", "2025-01-02")])
    s, short_book, balanced = simulate(sigs, synthetic_prices, "p2 > t", 0.5,
                                       entry_delay=1, hold_days=2)
    # Days the basket is active: held window is [trade_date+1, exit_date].
    sizes = basket_size_daily(s, synthetic_prices.index)
    active_mask = (sizes > 0).to_numpy()
    spy_ret = synthetic_prices["SPY"].pct_change(fill_method=None).fillna(0).to_numpy()
    expected = np.where(
        active_mask,
        0.5 * short_book.to_numpy() + 0.5 * spy_ret,
        0.0,
    )
    np.testing.assert_allclose(balanced.to_numpy(), expected, atol=1e-12)
    # Sanity: at least one active day, at least one inactive day in this scenario.
    assert active_mask.any() and (~active_mask).any()


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
# simulate_top_k — short the K names with the HIGHEST factor each signal_date
# -----------------------------------------------------------------------------

def test_top_k_picks_highest_factor_per_signal_date(synthetic_prices):
    """Three tickers signal on the same day with distinct p0 values. With K=2 and rule
    `p0`, top-K should pick the two HIGHEST p0 — C (0.90) and B (0.20), NOT A (0.10)."""
    px = synthetic_prices.copy()
    px["C"] = 80 * (1.005 ** np.arange(len(px)))   # gently drifting third name
    sigs = pd.DataFrame([
        _signal("A", "2025-01-02", p0=0.10, p1=0.45, p2=0.45),
        _signal("B", "2025-01-02", p0=0.20, p1=0.40, p2=0.40),
        _signal("C", "2025-01-02", p0=0.90, p1=0.05, p2=0.05),
    ])
    s, _, _ = simulate_top_k(sigs, px, "p0", k=2, entry_delay=1, hold_days=1)
    chosen = set(s["ticker"])
    assert chosen == {"C", "B"}


def test_top_k_holds_full_window_no_early_exit(synthetic_prices):
    """Top-K has no early-exit logic. A position entered on day 0 with hold=5 must
    contribute P&L on EXACTLY days 2..6 (entry day 1 → P&L on days 2..6)."""
    sigs = pd.DataFrame([_signal("A", "2025-01-02", p0=0.95)])
    _, short_book, _ = simulate_top_k(sigs, synthetic_prices, "p0",
                                         k=1, entry_delay=1, hold_days=5)
    nz = short_book[short_book != 0]
    assert list(nz.index) == [synthetic_prices.index[i] for i in (2, 3, 4, 5, 6)]


def test_top_k_uses_max_for_multi_class_rule(synthetic_prices):
    """Under `max(p0, p1)` the factor is the per-row max. Three tickers with distinct maxes:
    A max=0.95, B max=0.40, C max=0.95. With K=2 we expect the TWO highest maxes → {A, C}
    (ties broken by ticker name, both tied at 0.95)."""
    px = synthetic_prices.copy()
    px["C"] = 80 * (1.005 ** np.arange(len(px)))
    sigs = pd.DataFrame([
        _signal("A", "2025-01-02", p0=0.10, p1=0.95, p2=0.0),  # max = 0.95
        _signal("B", "2025-01-02", p0=0.40, p1=0.30, p2=0.0),  # max = 0.40
        _signal("C", "2025-01-02", p0=0.05, p1=0.95, p2=0.0),  # max = 0.95
    ])
    s, _, _ = simulate_top_k(sigs, px, "max(p0, p1)",
                             k=2, entry_delay=1, hold_days=1)
    assert set(s["ticker"]) == {"A", "C"}


def test_top_k_per_date_independent(synthetic_prices):
    """K is applied PER signal_date. Two days, three tickers each, K=1 → 2 trades total
    (one per day, the HIGHEST-factor name on each day — different ticker per day)."""
    px = synthetic_prices.copy()
    px["C"] = 80 * (1.005 ** np.arange(len(px)))
    sigs = pd.DataFrame([
        _signal("A", "2025-01-02", p0=0.10),
        _signal("B", "2025-01-02", p0=0.30),
        _signal("C", "2025-01-02", p0=0.50),   # day-1 winner
        _signal("A", "2025-01-03", p0=0.95),   # day-2 winner
        _signal("B", "2025-01-03", p0=0.05),
        _signal("C", "2025-01-03", p0=0.30),
    ])
    s, _, _ = simulate_top_k(sigs, px, "p0", k=1, entry_delay=1, hold_days=1)
    assert len(s) == 2
    chosen_by_date = dict(zip(s["signal_date"], s["ticker"]))
    assert chosen_by_date[pd.Timestamp("2025-01-02")] == "C"   # highest p0 = 0.50
    assert chosen_by_date[pd.Timestamp("2025-01-03")] == "A"   # highest p0 = 0.95


def test_top_k_balanced_gated_by_basket_activity(synthetic_prices):
    """Same gating rule as the threshold simulator: balanced[d] = 0.5 * short + 0.5 * SPY
    only on days the short book is live; otherwise 0."""
    sigs = pd.DataFrame([_signal("A", "2025-01-02", p0=0.95)])
    s, short_book, balanced = simulate_top_k(sigs, synthetic_prices, "p0",
                                             k=1, entry_delay=1, hold_days=1)
    sizes = basket_size_daily(s, synthetic_prices.index)
    active_mask = (sizes > 0).to_numpy()
    spy_ret = synthetic_prices["SPY"].pct_change(fill_method=None).fillna(0).to_numpy()
    expected = np.where(
        active_mask,
        0.5 * short_book.to_numpy() + 0.5 * spy_ret,
        0.0,
    )
    np.testing.assert_allclose(balanced.to_numpy(), expected, atol=1e-12)


def test_balanced_book_is_strictly_zero_when_basket_empty(synthetic_prices):
    """The strict invariant: on any day with NO held names, the balanced book MUST
    return exactly 0 — both legs flat, no exposure, no drift from SPY. Verifies the
    "no recall position → no hedge" semantic explicitly across a multi-window run."""
    # Short A for days 2–3 (hold=2). Days 0,1,4,5,...,9 have no positions.
    sigs = pd.DataFrame([_signal("A", "2025-01-02", p2=0.99)])
    s, short_book, balanced = simulate(sigs, synthetic_prices, "p2 > t", 0.5,
                                       entry_delay=1, hold_days=2)
    sizes = basket_size_daily(s, synthetic_prices.index).to_numpy()
    inactive = sizes == 0
    # Every inactive day: balanced is strictly 0 (not 0.5 * SPY).
    assert inactive.any(), "test scenario must include some no-position days"
    np.testing.assert_array_equal(balanced.to_numpy()[inactive], 0.0)
    # And the short book is also 0 on those days (defensive — no leak).
    np.testing.assert_array_equal(short_book.to_numpy()[inactive], 0.0)


def test_top_k_raises_on_invalid_args(synthetic_prices):
    sigs = pd.DataFrame([_signal("A", "2025-01-02")])
    with pytest.raises(ValueError, match="entry_delay"):
        simulate_top_k(sigs, synthetic_prices, "p0", k=1, entry_delay=0, hold_days=1)
    with pytest.raises(ValueError, match="hold_days"):
        simulate_top_k(sigs, synthetic_prices, "p0", k=1, entry_delay=1, hold_days=0)
    with pytest.raises(ValueError, match="k"):
        simulate_top_k(sigs, synthetic_prices, "p0", k=0, entry_delay=1, hold_days=1)
    with pytest.raises(ValueError, match="balanced_weight"):
        simulate_top_k(sigs, synthetic_prices, "p0", k=1, entry_delay=1, hold_days=1,
                       balanced_weight=1.5)


def test_top_k_empty_when_no_signals(synthetic_prices):
    """Sig table missing required tickers → empty result."""
    sigs = pd.DataFrame(columns=["ticker", "signal_date",
                                 "prob_class_0", "prob_class_1", "prob_class_2"])
    s, daily, daily_bal = simulate_top_k(sigs, synthetic_prices, "p0",
                                         k=1, entry_delay=1, hold_days=1)
    assert s.empty and daily.empty and daily_bal.empty


def test_top_k_pools_ae_dates_that_share_an_entry_day():
    """When multiple AE dates roll forward to the same entry trading day (e.g. AE
    dates over a weekend all rolling to Monday), top-K must pick K UNIQUE names
    from the COMBINED pool — not K per AE date.

    Setup: bdate_range starting Thu Jan 2 makes
        dates[1] = Fri Jan 3,   dates[2] = Mon Jan 6.
    With entry_delay=1, all of {Fri, Sat, Sun} → entry_idx = 2 (Mon trade day).
    Six tickers spread over Fri/Sat/Sun signals; combined ranking by p0 picks the
    K=2 HIGHEST globally, not 2 per date.
    """
    dates = pd.bdate_range("2025-01-02", periods=10)
    fri = pd.Timestamp("2025-01-03")  # trading day, dates[1]
    sat = pd.Timestamp("2025-01-04")  # weekend → rolls to Mon
    sun = pd.Timestamp("2025-01-05")  # weekend → rolls to Mon

    sigs = pd.DataFrame([
        _signal("A", fri, p0=0.10),
        _signal("B", fri, p0=0.20),
        _signal("C", sat, p0=0.05),
        _signal("D", sat, p0=0.15),
        _signal("E", sun, p0=0.25),
        _signal("F", sun, p0=0.95),  # global highest
    ])
    # Combined ranking by p0 (descending): F(0.95) > E(0.25) > B(0.20) > D(0.15) > A(0.10) > C(0.05).
    # With K=2 we expect {F, E}, NOT 2 per AE-date which would give 6 trades.

    prices = pd.DataFrame(
        {tk: [100.0] * 10 for tk in ["A", "B", "C", "D", "E", "F", "SPY"]},
        index=dates,
    )
    s, _, _ = simulate_top_k(
        sigs, prices, "p0", k=2, entry_delay=1, hold_days=1,
    )
    assert len(s) == 2
    assert set(s["ticker"]) == {"F", "E"}


def test_top_k_dedupes_ticker_within_entry_day_group():
    """If the same ticker appears in TWO AE-date signals that both roll to one
    entry day, it should count once toward K — keeping the HIGHEST-factor row."""
    dates = pd.bdate_range("2025-01-02", periods=10)
    fri = pd.Timestamp("2025-01-03")
    sat = pd.Timestamp("2025-01-04")  # rolls to Mon = same entry as Fri

    # A appears twice (Fri and Sat) with different p0; B and C appear once each.
    sigs = pd.DataFrame([
        _signal("A", fri, p0=0.10),
        _signal("A", sat, p0=0.95),  # A's HIGHEST factor across the pool
        _signal("B", fri, p0=0.20),
        _signal("C", sat, p0=0.30),
    ])
    # Combined unique-ticker ranking (descending): A(0.95), C(0.30), B(0.20).
    # K=2 → {A, C}, with A's representative row carrying p0=0.95 (the higher one).

    prices = pd.DataFrame(
        {tk: [100.0] * 10 for tk in ["A", "B", "C", "SPY"]},
        index=dates,
    )
    s, _, _ = simulate_top_k(
        sigs, prices, "p0", k=2, entry_delay=1, hold_days=1,
    )
    assert len(s) == 2
    assert set(s["ticker"]) == {"A", "C"}
    a_row = s[s["ticker"] == "A"].iloc[0]
    assert a_row["prob_class_0"] == pytest.approx(0.95)


def test_top_k_skips_tickers_not_in_prices(synthetic_prices):
    """A ticker not in the price table cannot be traded; top-K must ignore it even if
    it would otherwise rank into the K highest."""
    sigs = pd.DataFrame([
        _signal("MISSING", "2025-01-02", p0=0.99),  # would rank highest, but no prices
        _signal("A",       "2025-01-02", p0=0.40),
        _signal("B",       "2025-01-02", p0=0.20),
    ])
    s, _, _ = simulate_top_k(sigs, synthetic_prices, "p0",
                             k=1, entry_delay=1, hold_days=1)
    assert len(s) == 1
    assert s["ticker"].iloc[0] == "A"


# -----------------------------------------------------------------------------
# validate_prices / clean_prices
# -----------------------------------------------------------------------------

def test_validate_prices_clean(synthetic_prices):
    """The fixture has a flat ticker B by design (for the multi-ticker average test),
    so we don't assert stale_run_count == 0 here — just no NaNs or non-positive obs."""
    issues = validate_prices(synthetic_prices)
    assert issues["all_nan_tickers"] == []
    assert issues["partial_nan_tickers"] == {}
    assert issues["non_positive_obs"] == 0


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
# portfolio_turnover_daily
#
# Convention: daily turnover = Σ_i |w_i(t) − w_i(t−1)|. 100% means the L1 weight
# change equals 1 on a typical day (one full GMV's worth of trading). A complete
# same-day book swap has Σ|Δw| = 2 → 200%.
# -----------------------------------------------------------------------------

def test_turnover_zero_when_no_trades(synthetic_prices):
    empty = pd.DataFrame(columns=["ticker", "trade_date", "exit_date"])
    assert portfolio_turnover_daily(empty, synthetic_prices.index) == 0.0


def test_turnover_buy_and_hold_only_initial_allocation():
    """One name held throughout the window: only the ∅→{A} transition contributes.
    On that single day Σ|Δw| = 1.0 (the new weight). Every other day contributes 0.
    Mean over (N−1) active-pair days = 1.0/(N−1)."""
    dates = pd.bdate_range("2025-01-02", periods=20)
    s = pd.DataFrame([{"ticker": "A",
                       "trade_date": dates[0],
                       "exit_date": dates[-1]}])
    daily = portfolio_turnover_daily(s, dates)
    expected_pct = 1.0 / (len(dates) - 1) * 100
    assert daily == pytest.approx(expected_pct, rel=1e-9)


def test_turnover_once_per_week_rotation():
    """A 5-name book that swaps 1-in / 1-out per day has steady-state Σ|Δw| = 2/5 = 0.4
    (40%). Construction: trade i opens day i, exits day i+5. After warm-up the basket
    holds 5 names with one entry and one exit per day."""
    dates = pd.bdate_range("2025-01-02", periods=400)
    rows = [{"ticker": f"T{i:04d}",
             "trade_date": dates[i],
             "exit_date": dates[i + 5]} for i in range(390)]
    s = pd.DataFrame(rows)
    daily = portfolio_turnover_daily(s, dates)
    # Steady state = 40%; ramp + cooldown of ~10 days out of 400 negligible.
    assert 39.0 < daily < 41.0


def test_turnover_steady_state_2_over_hold_days():
    """For a constant-size basket with 1-in/1-out per day and hold=H, steady-state daily
    turnover is 2/H (one entry of weight 1/H + one exit of weight 1/H = Σ|Δw| = 2/H).
    Verify across H ∈ {5, 10, 20, 40}, allowing a small ramp tolerance."""
    for hold in [5, 10, 20, 40]:
        n_days = 600
        n_trades = n_days - hold
        dates = pd.bdate_range("2025-01-02", periods=n_days)
        rows = [{"ticker": f"T{i:04d}",
                 "trade_date": dates[i],
                 "exit_date": dates[i + hold]} for i in range(n_trades)]
        s = pd.DataFrame(rows)
        daily = portfolio_turnover_daily(s, dates) / 100  # back to fraction
        target = 2.0 / hold
        assert abs(daily - target) < 0.04, f"hold={hold}: got {daily:.4f}, expected ~{target:.4f}"


def test_turnover_static_two_name_book():
    """Two names entered together on day 0, held throughout. Only one transition has
    nonzero L1: ∅ → {A,B}, with each new weight = 0.5, so Σ|Δw| = 1.0. All subsequent
    days have Σ|Δw| = 0. Mean over 9 active-pair days = 1.0/9."""
    dates = pd.bdate_range("2025-01-02", periods=10)
    s = pd.DataFrame([
        {"ticker": "A", "trade_date": dates[0], "exit_date": dates[-1]},
        {"ticker": "B", "trade_date": dates[0], "exit_date": dates[-1]},
    ])
    daily = portfolio_turnover_daily(s, dates)
    assert daily == pytest.approx(1.0 / 9 * 100, rel=1e-9)


def test_turnover_swap_full_book_in_one_day():
    """Book {A,B} on day 1, then {C,D} on day 2 — a complete swap of the equal-weight basket.
    Day 1 (∅→{A,B}): Σ|Δw| = 1.0. Day 2 (full swap): Σ|Δw| = 4·0.5 = 2.0 → 200%.
    Day 3 ({C,D}→∅): Σ|Δw| = 1.0. Mean of [1.0, 2.0, 1.0] = 4/3 → 133.33%."""
    dates = pd.bdate_range("2025-01-02", periods=4)
    s = pd.DataFrame([
        {"ticker": "A", "trade_date": dates[0], "exit_date": dates[1]},
        {"ticker": "B", "trade_date": dates[0], "exit_date": dates[1]},
        {"ticker": "C", "trade_date": dates[1], "exit_date": dates[2]},
        {"ticker": "D", "trade_date": dates[1], "exit_date": dates[2]},
    ])
    daily = portfolio_turnover_daily(s, dates)
    assert daily == pytest.approx(400.0 / 3, rel=1e-9)


# -----------------------------------------------------------------------------
# Defensive raise paths and edge-case branches
# -----------------------------------------------------------------------------

def test_validate_prices_raises_on_non_datetime_index():
    df = pd.DataFrame({"X": [1, 2, 3]}, index=[0, 1, 2])
    with pytest.raises(ValueError, match="DatetimeIndex"):
        validate_prices(df)


def test_validate_prices_raises_on_unsorted_index():
    dates = pd.DatetimeIndex(["2025-01-03", "2025-01-02", "2025-01-04"])
    df = pd.DataFrame({"X": [1, 2, 3]}, index=dates)
    with pytest.raises(ValueError, match="sorted ascending"):
        validate_prices(df)


def test_simulate_raises_on_invalid_balanced_weight(synthetic_prices):
    sig = pd.DataFrame([_signal("A", "2025-01-02")])
    with pytest.raises(ValueError, match="balanced_weight"):
        simulate(sig, synthetic_prices, "p2 > t", 0.5, 1, 1, balanced_weight=1.5)
    with pytest.raises(ValueError, match="balanced_weight"):
        simulate(sig, synthetic_prices, "p2 > t", 0.5, 1, 1, balanced_weight=-0.1)


def test_simulate_raises_on_invalid_entry_delay(synthetic_prices):
    sig = pd.DataFrame([_signal("A", "2025-01-02")])
    with pytest.raises(ValueError, match="entry_delay"):
        simulate(sig, synthetic_prices, "p2 > t", 0.5, 0, 1)


def test_simulate_raises_on_invalid_hold_days(synthetic_prices):
    sig = pd.DataFrame([_signal("A", "2025-01-02")])
    with pytest.raises(ValueError, match="hold_days"):
        simulate(sig, synthetic_prices, "p2 > t", 0.5, 1, 0)


def test_simulate_no_spy_uses_cash_benchmark(synthetic_prices):
    """When SPY is missing, the long leg yields cash (=0). With balanced_weight=0.5,
    balanced[d] should equal 0.5 * short_book[d] (not equal to short_book itself, which
    was the old behavior that silently ignored balanced_weight)."""
    px = synthetic_prices.drop(columns=["SPY"])
    sig = pd.DataFrame([_signal("A", "2025-01-02")])
    _, short_book, balanced = simulate(sig, px, "p2 > t", 0.5, 1, 1, balanced_weight=0.5)
    np.testing.assert_allclose(balanced.to_numpy(), 0.5 * short_book.to_numpy(), atol=1e-12)
    # And the short_book itself reflects the held day (-1% on day 2)
    assert short_book.iloc[2] == pytest.approx(-0.01, abs=1e-9)
    assert balanced.iloc[2] == pytest.approx(-0.005, abs=1e-9)


def test_simulate_signals_all_past_window_returns_empty(synthetic_prices):
    """If every signal's entry date falls past the end of the trading window, no trade can
    accumulate P&L. simulate returns an empty trade table and empty daily series."""
    # Signal on the second-to-last day of the window with entry_delay=10 → entry index >> last day.
    late_date = synthetic_prices.index[-2].strftime("%Y-%m-%d")
    sig = pd.DataFrame([_signal("A", late_date)])
    s, daily_short, daily_bal = simulate(sig, synthetic_prices, "p2 > t", 0.5,
                                         entry_delay=10, hold_days=1)
    assert s.empty
    assert daily_short.empty
    assert daily_bal.empty


def test_portfolio_turnover_daily_empty_directly():
    """Direct call with an empty trade table — should short-circuit and return 0."""
    dates = pd.bdate_range("2025-01-02", periods=10)
    empty = pd.DataFrame(columns=["ticker", "trade_date", "exit_date"])
    assert portfolio_turnover_daily(empty, dates) == 0.0


def test_sharpe_handles_short_or_constant_input():
    """Edge-case inputs all return NaN cleanly (no warnings, no crashes)."""
    assert pd.isna(sharpe(pd.Series([], dtype=float)))         # empty
    assert pd.isna(sharpe(pd.Series([0.01])))                  # length 1
    assert pd.isna(sharpe(pd.Series([0.0, 0.0, 0.0])))         # zero std


def test_max_drawdown_short_input():
    """Empty / single-point series have undefined drawdown; we return 0.0 by convention."""
    assert max_drawdown(pd.Series([], dtype=float)) == 0.0
    assert max_drawdown(pd.Series([5.0])) == 0.0


# -----------------------------------------------------------------------------
# End-to-end synthetic validation
#
# These two tests construct a controlled price path AND a fixed signal calendar
# so every active position, every basket-membership transition, and every daily
# return value is knowable by hand. We then assert the simulator's output against
# those values. They are the canonical "the strategy code does what we say it
# does" tests.
# -----------------------------------------------------------------------------

def _prices_from_returns(returns_per_ticker: dict, dates: pd.DatetimeIndex,
                         start: float = 100.0) -> pd.DataFrame:
    """Build a wide price DataFrame from per-ticker daily-return lists.

    ``returns_per_ticker[t][i]`` is the return ON day i (price[i] / price[i-1] - 1).
    The first entry is ignored (used as a placeholder for the no-prior-day return).
    """
    out = {}
    for ticker, rets in returns_per_ticker.items():
        prices = [start]
        for r in rets[1:]:
            prices.append(prices[-1] * (1.0 + r))
        out[ticker] = prices
    return pd.DataFrame(out, index=dates)


def test_threshold_strategy_full_synthetic_validation():
    """End-to-end synthetic validation of the **Threshold** strategy.

    Spec we're verifying:
      - When a ticker's factor exceeds entry_threshold, go SHORT at close of
        signal_date + entry_delay trading days.
      - The position is held for at most hold_days trading days (natural exit).
      - Early exit fires if a SUBSEQUENT signal for the same ticker has factor <
        exit_threshold; the position closes at close of (that signal_date +
        entry_delay), provided that's BEFORE the natural exit.
      - A repeat entry signal while the ticker is already in the book starts a
        FRESH overlapping trade (extending the membership window via OR).
      - All shorts are equal-weighted: short_book[d] = -mean(daily_ret) over the
        held tickers on day d.
      - Balanced book: balanced[d] = 0.5 * short_book[d] + 0.5 * SPY_ret[d].
    """
    dates = pd.bdate_range("2025-01-02", periods=12)

    # Engineered daily returns. ON DAY i, return goes from price[i-1] to price[i].
    a_rets   = [None, 0.0,   0.04,  0.0,   -0.02,  0.0,   0.06,  0.0, 0.0, 0.0, 0.0, 0.0]
    b_rets   = [None, 0.0,   0.0,   0.02,   0.04, -0.02,  0.0,   0.0, 0.0, 0.0, 0.0, 0.0]
    spy_rets = [None] + [0.0] * 11
    prices = _prices_from_returns(
        {"A": a_rets, "B": b_rets, "SPY": spy_rets}, dates,
    )

    # Signal calendar — see expected trade table below.
    sigs = pd.DataFrame([
        _signal("A", dates[0], p2=0.90),  # A entry #1
        _signal("B", dates[1], p2=0.70),  # B entry
        _signal("A", dates[2], p2=0.70),  # A entry #2 (extends membership window)
        _signal("B", dates[4], p2=0.05),  # B early-exit trigger
        _signal("A", dates[5], p2=0.05),  # A early-exit trigger (closes both A trades)
    ])

    s, short_book, balanced = simulate(
        sigs, prices, "p2 > t", threshold=0.5, entry_delay=1, hold_days=4,
        exit_threshold=0.20,
    )

    # ---- Per-trade table -----------------------------------------------------
    # A trade #1 (signal d0): entry_idx = 1, natural exit = 5,
    #   first exit-trig after d0 is d5 → cand_exit_idx = 6, min(5,6) = 5 → exit=5.
    # A trade #2 (signal d2): entry_idx = 3, natural exit = 7,
    #   first exit-trig after d2 is d5 → cand_exit_idx = 6, min(7,6) = 6 → exit=6.
    # B trade    (signal d1): entry_idx = 2, natural exit = 6,
    #   first exit-trig after d1 is d4 → cand_exit_idx = 5, min(6,5) = 5 → exit=5.
    assert len(s) == 3
    a_trades = (s[s["ticker"] == "A"]
                .sort_values("trade_date").reset_index(drop=True))
    b_trades = (s[s["ticker"] == "B"]
                .sort_values("trade_date").reset_index(drop=True))
    assert a_trades["trade_date"].tolist() == [dates[1], dates[3]]
    assert a_trades["exit_date"].tolist()  == [dates[5], dates[6]]
    assert b_trades["trade_date"].tolist() == [dates[2]]
    assert b_trades["exit_date"].tolist()  == [dates[5]]

    # ---- Basket composition per day -----------------------------------------
    # held[ti+1 : ei+1] for each trade:
    #   A#1: held days 2..5
    #   A#2: held days 4..6
    #   B  : held days 3..5
    # Combined (binary OR):
    #   d0,d1: ∅;  d2: {A};  d3: {A,B};  d4: {A,B};  d5: {A,B};  d6: {A};  d7+: ∅
    bsz = basket_size_daily(s, dates)
    assert list(bsz.values) == [0, 0, 1, 2, 2, 2, 1, 0, 0, 0, 0, 0]

    # ---- Daily short_book returns -------------------------------------------
    # short_book[d] = -mean(daily_returns of held tickers on day d).
    expected_short = [
        0.0,                    # d0  ∅
        0.0,                    # d1  ∅
        -0.04,                  # d2  {A}: -A_ret = -0.04
        -(0.0 + 0.02) / 2,      # d3  {A,B}: -mean(0, 0.02) = -0.01
        -(-0.02 + 0.04) / 2,    # d4  {A,B}: -mean(-0.02, 0.04) = -0.01
        -(0.0 + -0.02) / 2,     # d5  {A,B}: -mean(0, -0.02) = +0.01
        -0.06,                  # d6  {A}: -A_ret = -0.06
        0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    np.testing.assert_allclose(short_book.values, expected_short, atol=1e-12)

    # ---- Balanced book: 0.5 * short + 0.5 * SPY (SPY flat → 0.5 * short) ----
    expected_bal = [0.5 * v for v in expected_short]
    np.testing.assert_allclose(balanced.values, expected_bal, atol=1e-12)


def test_top_k_strategy_full_synthetic_validation():
    """End-to-end synthetic validation of the **Top-K** strategy.

    Spec we're verifying:
      - On each signal_date, rank tickers by factor_for_top_k(condition) and
        SHORT the K names with the HIGHEST factor.
      - Each pick enters at close of signal_date + entry_delay trading days and
        is held for exactly hold_days trading days. NO thresholds, NO early exit.
      - Same ticker can be picked on consecutive signal_dates; each pick spawns
        an independent overlapping trade (basket aggregates via OR).
      - All shorts equal-weighted: short_book[d] = -mean(daily_ret) over held set.
      - Balanced: balanced[d] = 0.5 * short_book[d] + 0.5 * SPY_ret[d].
    """
    dates = pd.bdate_range("2025-01-02", periods=10)

    a_rets   = [None, 0.0,  0.06, 0.0,   0.0,  0.0, 0.0, 0.0, 0.0, 0.0]
    b_rets   = [None, 0.0,  0.04, 0.0,   0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
    c_rets   = [None, 0.0,  0.0,  0.0,   0.06, 0.0, 0.0, 0.0, 0.0, 0.0]
    d_rets   = [None] + [0.0] * 9   # D never moves; included so it can be ranked
    spy_rets = [None] + [0.0] * 9
    prices = _prices_from_returns(
        {"A": a_rets, "B": b_rets, "C": c_rets, "D": d_rets, "SPY": spy_rets},
        dates,
    )

    # On d0: rank by p0 → A=0.95, B=0.85, D=0.20, C=0.10.  Top-2 = {A, B}.
    # On d2: rank by p0 → B=0.95, C=0.85, A=0.10, D=0.20.  Top-2 = {B, C}.
    sigs = pd.DataFrame([
        _signal("A", dates[0], p0=0.95),
        _signal("B", dates[0], p0=0.85),
        _signal("C", dates[0], p0=0.10),
        _signal("D", dates[0], p0=0.20),
        _signal("A", dates[2], p0=0.10),
        _signal("B", dates[2], p0=0.95),
        _signal("C", dates[2], p0=0.85),
        _signal("D", dates[2], p0=0.20),
    ])

    s, short_book, balanced = simulate_top_k(
        sigs, prices, "p0", k=2, entry_delay=1, hold_days=2,
    )

    # ---- Per-trade table -----------------------------------------------------
    # d0 cohort: A & B selected.  entry_idx = 1, exit_idx = 3.
    # d2 cohort: B & C selected.  entry_idx = 3, exit_idx = 5.
    assert len(s) == 4
    d0_cohort = s[s["signal_date"] == dates[0]]
    d2_cohort = s[s["signal_date"] == dates[2]]
    assert set(d0_cohort["ticker"]) == {"A", "B"}
    assert set(d2_cohort["ticker"]) == {"B", "C"}
    assert all(td == dates[1] for td in d0_cohort["trade_date"])
    assert all(ed == dates[3] for ed in d0_cohort["exit_date"])
    assert all(td == dates[3] for td in d2_cohort["trade_date"])
    assert all(ed == dates[5] for ed in d2_cohort["exit_date"])

    # ---- Basket composition per day -----------------------------------------
    # d0 cohort: held days 2, 3 for A and B
    # d2 cohort: held days 4, 5 for B and C
    #   d0,d1: ∅;  d2: {A,B};  d3: {A,B};  d4: {B,C};  d5: {B,C};  d6+: ∅
    bsz = basket_size_daily(s, dates)
    assert list(bsz.values) == [0, 0, 2, 2, 2, 2, 0, 0, 0, 0]

    # ---- Daily short_book returns -------------------------------------------
    # d2 {A,B}: -mean(A_ret=0.06, B_ret=0.04) = -0.05
    # d3 {A,B}: -mean(0, 0)                   = 0
    # d4 {B,C}: -mean(B_ret=0.02, C_ret=0.06) = -0.04
    # d5 {B,C}: -mean(0, 0)                   = 0
    expected_short = [0.0, 0.0, -0.05, 0.0, -0.04, 0.0, 0.0, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(short_book.values, expected_short, atol=1e-12)

    # ---- Balanced book: 0.5 * short + 0.5 * SPY (SPY flat → 0.5 * short) ----
    expected_bal = [0.5 * v for v in expected_short]
    np.testing.assert_allclose(balanced.values, expected_bal, atol=1e-12)
