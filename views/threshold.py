"""Threshold Backtest page.

Each AE-date signal that triggers the rule (factor > entry_threshold) opens a short.
Optional early exit closes a position when the factor drops below exit_threshold on a
later signal_date. All knobs live in the sidebar; the strategy rules expander explains
the mechanics in detail.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from strategy import (
    STRATEGIES, simulate, sharpe, max_drawdown,
    basket_size_daily, portfolio_turnover_daily,
)
from app_common import (
    BRAND_BLUE, BRAND_AMBER, BRAND_NAVY, BRAND_SLATE,
    load_signals, load_prices, show_data_health,
    base_layout, style_axes, trim_to_active_window,
)


show_data_health()

sig = load_signals()
prices = load_prices()


# ---------------------------------------------------------------------------
# Sidebar knobs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Threshold Knobs")
    st.html("<br/>")
    strategy_keys = list(STRATEGIES.keys())
    cond = st.selectbox(
        "Short Trigger Rule (short fires on condition)", strategy_keys,
        index=strategy_keys.index("p0 > t OR p1 > t"),
        help="OR rules use max(...) for ranking — short when any of the listed probabilities "
             "exceeds the entry threshold.",
    )
    threshold = st.slider("Entry threshold", 0.0, 1.0, 0.60, 0.01,
                          help="Position opens when the condition's factor > this value.")
    exit_threshold = st.slider(
        "Exit threshold (early exit if p < threshold)", 0.0, 1.0, 0.40, 0.01,
        help="If a later ticker-day shows factor < this value, close the position early at the "
             "exit signal's trade date. 0 disables early exit. Must be strictly < entry threshold.",
    )
    if exit_threshold >= threshold:
        st.error(f"Exit threshold ({exit_threshold:.2f}) must be < entry threshold "
                 f"({threshold:.2f}). Lower the exit slider.")
        st.stop()
    entry_delay = st.number_input("Entry delay (trading days after AE date)", 1, 60, 20, 1)
    hold_days = st.number_input("Holding period (trading days)", 1, 120, 20, 1)


# ---------------------------------------------------------------------------
# Strategy rules (specific to this backtest)
# ---------------------------------------------------------------------------
with st.expander("Strategy rules", expanded=False):
    st.markdown(
        """
**Probability classes** — `p0` = probability of a recall within **30 days**,
`p1` = within **60 days**, `p2` = within **90 days** of the AE date.

**Entry**

- When `condition(p0, p1, p2)`'s factor exceeds `entry_threshold` for ticker `T` on day `d`
  (the AE date), go **short** at close of `d + entry_delay` trading days.
- Natural exit: close of `d + entry_delay + hold_days`.

**Early exit (optional)**

- If a later ticker-day for `T` has `factor < exit_threshold` (must be strictly less than
  the entry threshold), the position closes at close of `(that_signal_date + entry_delay)`,
  whichever comes first relative to the natural exit.
- Setting `exit_threshold = 0` disables early exit (probabilities are non-negative).
- Each entry signal owns its own window — a later re-entry signal opens a fresh independent position.

**Daily portfolio return** (true daily marks, no compounding)

- `held(d)` = set of tickers with at least one active short on day `d`.
- `short_book[d] = −mean( r[d, t] for t in held(d) )`.
- `balanced[d] = 0.5 × short_book[d] + 0.5 × SPY_return[d]`.
        """
    )


# ---------------------------------------------------------------------------
# Run the backtest
# ---------------------------------------------------------------------------
s, daily_short, daily_bal = simulate(
    sig, prices, cond, threshold, entry_delay, hold_days,
    exit_threshold=exit_threshold,
)

if daily_short.empty:
    st.warning("No signals fired with these parameters.")
    st.stop()

# Trim to the active window (first → last day the basket actually held names).
basket_full = basket_size_daily(s, daily_short.index)
daily_short, daily_bal, basket_size_series = trim_to_active_window(
    daily_short, daily_bal, basket_full,
)

cum_short = daily_short.cumsum() * 100
cum_bal = daily_bal.cumsum() * 100

sharpe_bal = sharpe(daily_bal)
ann_ret_bal = daily_bal.mean() * 252 * 100
ann_vol_bal = daily_bal.std(ddof=1) * np.sqrt(252) * 100
dd_bal = max_drawdown(cum_bal)
turnover_daily_pct = portfolio_turnover_daily(s, daily_short.index)

active_basket = basket_size_series[basket_size_series > 0]
avg_basket = active_basket.mean() if len(active_basket) else 0.0
universe_size = sig["ticker"].nunique()


# ---------------------------------------------------------------------------
# Section header — period range + strategy pill
# ---------------------------------------------------------------------------
period_str = (f"{daily_short.index[0].strftime('%b %d, %Y')} → "
              f"{daily_short.index[-1].strftime('%b %d, %Y')}")
st.markdown(
    f"""
    <div class="section-meta">
      Threshold strategy · {period_str}
      <span class="pill">balanced book · 50% short / 50% SPY</span>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Port Sharpe", f"{sharpe_bal:+.2f}")
c2.metric("Active trading days", f"{len(daily_short):,}")
c3.metric("Avg # stocks in short book", f"{int(round(avg_basket)):d}",
          help="Mean unique tickers held per day, averaged over days the book is active.")
c4.metric("Universe size (recall-prob tickers)", f"{universe_size:,}",
          help="Unique tickers with at least one model prediction.")

d1, d2, d3, d4 = st.columns(4)
d1.metric("Ann. return", f"{ann_ret_bal:+.1f}%",
          help="mean(daily_balanced) × 252.")
d2.metric("Ann. volatility", f"{ann_vol_bal:.1f}%",
          help="std(daily_balanced) × √252.")
d3.metric("Max drawdown", f"{dd_bal:+.1f}%",
          help="Largest peak-to-trough drop on the balanced cumulative-return curve.")
d4.metric("Avg daily turnover", f"{turnover_daily_pct:.1f}%",
          help="Avg over active days of Σ|Δw|. 100% means one full GMV's worth of trading "
               "per day. Steady-state shorthand: 2/hold_days.")


# ---------------------------------------------------------------------------
# Cumulative return + drawdown chart
# ---------------------------------------------------------------------------
fig = make_subplots(rows=2, cols=1, row_heights=[0.66, 0.34],
                    vertical_spacing=0.22,
                    subplot_titles=("Cumulative return (%)", "Drawdown (pp)"))
fig.add_trace(go.Scatter(x=cum_short.index, y=cum_short.values,
                         name="Pure short basket",
                         line=dict(width=2.0, color=BRAND_BLUE),
                         hovertemplate="%{x|%b %d, %Y}<br>%{y:+.1f}%<extra></extra>"),
              row=1, col=1)
fig.add_trace(go.Scatter(x=cum_bal.index, y=cum_bal.values,
                         name="Balanced (0.5 short + 0.5 SPY)",
                         line=dict(width=2.0, color=BRAND_AMBER),
                         hovertemplate="%{x|%b %d, %Y}<br>%{y:+.1f}%<extra></extra>"),
              row=1, col=1)
fig.add_hline(y=0, line=dict(color="#94A3B8", width=0.6, dash="dot"), row=1, col=1)

dd_s = (cum_short - cum_short.cummax()).values
dd_b = (cum_bal - cum_bal.cummax()).values
fig.add_trace(go.Scatter(x=cum_short.index, y=dd_s, name="Pure short DD",
                         fill="tozeroy", mode="lines",
                         line=dict(color=BRAND_BLUE, width=1.0),
                         fillcolor="rgba(30, 64, 175, 0.20)", showlegend=False,
                         hovertemplate="%{x|%b %d, %Y}<br>%{y:.1f}pp<extra></extra>"),
              row=2, col=1)
fig.add_trace(go.Scatter(x=cum_bal.index, y=dd_b, name="Balanced DD",
                         fill="tozeroy", mode="lines",
                         line=dict(color=BRAND_AMBER, width=1.0),
                         fillcolor="rgba(217, 119, 6, 0.18)", showlegend=False,
                         hovertemplate="%{x|%b %d, %Y}<br>%{y:.1f}pp<extra></extra>"),
              row=2, col=1)
fig.add_hline(y=0, line=dict(color="#94A3B8", width=0.6, dash="dot"), row=2, col=1)
style_axes(fig)
fig.update_xaxes(title_text=None, row=2, col=1)
fig.update_yaxes(title_text="cum return (%)", row=1, col=1,
                 title_font=dict(color=BRAND_SLATE, size=12))
fig.update_yaxes(title_text="drawdown (pp)", row=2, col=1,
                 title_font=dict(color=BRAND_SLATE, size=12))
fig.update_layout(
    **base_layout(
        f"Strategy returns — {cond}, threshold {threshold:.2f}, hold {hold_days}td",
        height=640, top_margin=150,
    ),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0,
                bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
)
for ann in fig["layout"]["annotations"]:
    ann["font"] = dict(color=BRAND_NAVY, size=13, family="-apple-system, sans-serif")
    ann["x"] = 0
    ann["xanchor"] = "left"
st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Daily basket returns (in expander)
# ---------------------------------------------------------------------------
with st.expander("Daily trade returns"):
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=daily_short.index, y=daily_short.values * 100,
                          name="daily short basket return (%)",
                          marker=dict(color=BRAND_BLUE),
                          hovertemplate="%{x|%b %d, %Y}<br>%{y:+.1f}%<extra></extra>"))
    fig2.add_hline(y=0, line=dict(color="#94A3B8", width=0.6, dash="dot"))
    style_axes(fig2)
    fig2.update_layout(
        **base_layout(
            f"Daily short-basket returns — {cond}, threshold {threshold:.2f}",
            height=340, top_margin=80,
        ),
        showlegend=False,
        xaxis_title="trade date",
        yaxis_title="daily short basket return (%)",
        xaxis_title_font=dict(color=BRAND_SLATE, size=12),
        yaxis_title_font=dict(color=BRAND_SLATE, size=12),
    )
    st.plotly_chart(fig2, width="stretch")


# ---------------------------------------------------------------------------
# Per-trade & per-ticker tables (in expanders)
# ---------------------------------------------------------------------------
with st.expander("Per-trade table"):
    cols = ["ticker", "signal_date", "trade_date", "exit_date",
            "prob_class_0", "prob_class_1", "prob_class_2",
            "stock_ret", "short_ret"]
    cols = [c for c in cols if c in s.columns]
    st.dataframe(
        s[cols].sort_values("trade_date").reset_index(drop=True),
        width="stretch",
    )

with st.expander("Per-ticker contribution"):
    by_tk = (
        s.groupby("ticker")
         .agg(n_signals=("ticker", "size"),
              avg_short_ret=("short_ret", "mean"),
              total_short_ret=("short_ret", "sum"))
         .sort_values("total_short_ret", ascending=False)
    )
    st.dataframe(by_tk, width="stretch")


