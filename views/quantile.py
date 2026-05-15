"""Quantile Long-Short Backtest page.

Each AE date, rank tickers by ``factor_for_top_k(condition)`` and bucket them into
``n_quantiles`` equal-population quantiles. Long Q1 (lowest recall probability),
short Qn (highest). Displays per-quantile cumulative returns plus the LS spread.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from strategy import (
    TOP_K_RULES, simulate_quantile_ls, sharpe, max_drawdown,
)
from app_common import (
    BRAND_BLUE, BRAND_AMBER, BRAND_NAVY, BRAND_SLATE,
    QUANTILE_PALETTE,
    load_signals, load_prices,
    base_layout, chart_title, style_axes,
)


sig = load_signals()
prices = load_prices()


# ---------------------------------------------------------------------------
# Sidebar knobs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Quantile L/S Knobs")
    st.html("<br/>")
    rule_keys = list(TOP_K_RULES.keys())
    cond = st.selectbox(
        "Short Trigger Rule (ranking factor)", rule_keys,
        index=rule_keys.index("p2"),
        help="Single-class rules (p0, p1, p2) rank by that probability column. "
             "Multi-class rules (max(p0, p1), max(p0, p1, p2)) take the max across "
             "the listed columns.",
    )
    n_quantiles = st.number_input(
        "Number of quantiles (N)",
        min_value=2, max_value=10, value=4, step=1,
        help="Tickers are bucketed each AE date into N equal-population quantiles by "
             "the trigger rule's factor. Q1 = lowest factor (low recall prob, LONG); "
             "Qn = highest factor (high recall prob, SHORT).",
    )
    entry_delay = st.number_input("Entry delay (trading days after AE date)", 1, 60, 20, 1)
    hold_days = st.number_input(
        "Holding period (trading days)",
        min_value=1, max_value=120, value=60, step=1,
        help="Each AE date's quartile cohort is held for this many trading days. "
             "Cohorts from different AE dates overlap: on any day, the per-quantile "
             "basket is the equal-weighted mean of every currently-held name's daily "
             "return. hold_days=1 recovers daily rebucketing (no overlap).",
    )


# ---------------------------------------------------------------------------
# Strategy rules
# ---------------------------------------------------------------------------
with st.expander("Strategy rules", expanded=False):
    st.markdown(
        f"""
**Probability classes** — `p0` = probability of a recall within **30 days**,
`p1` = within **60 days**, `p2` = within **90 days** of the AE date.

**Bucketing (per AE date)**

- For each `signal_date`, compute the factor across all tickers with predictions.
  Single-class rules (`p0`, `p1`, `p2`) use that probability column directly.
  Multi-class rules (`max(p0, p1)`, `max(p0, p1, p2)`) take the **max** across the
  listed columns.
- Rank the pool and split into **{int(n_quantiles)}** equal-population quantiles.
  Q1 = lowest factor (least likely to be recalled). Q{int(n_quantiles)} = highest
  (most likely).

**Long-short construction**

- **Long** the Q1 basket (low recall prob → expected outperformer).
- **Short** the Q{int(n_quantiles)} basket (high recall prob → expected underperformer).
- Each pick enters at close of `signal_date + entry_delay` trading days and is held
  for **{int(hold_days)} trading day{'s' if int(hold_days) != 1 else ''}**.
- **Overlapping cohorts (Jegadeesh-Titman)**: when `hold_days > 1`, baskets from
  different AE dates overlap. Each cohort keeps its own quantile assignments;
  portfolio Q_q daily return is the **average of cohort-level Q_q daily returns**.
  A ticker assigned Q1 by 12 active cohorts and Q{int(n_quantiles)} by 8 contributes
  12× to Q1 and 8× to Q{int(n_quantiles)} — relative frequency of assignment is
  preserved. (A naive per-name pool would mark the ticker held in both buckets
  equally and wash out the signal in this small-universe / long-hold regime.)
- Daily LS return = `Q1_return − Q{int(n_quantiles)}_return`.

**No external benchmark** — the long-short pair is self-funding (the long leg IS
the hedge for the short leg). Days with no positions in any quantile leave all
quantile and LS returns at 0.
        """
    )


# ---------------------------------------------------------------------------
# Run the backtest
# ---------------------------------------------------------------------------
quantile_returns, long_short = simulate_quantile_ls(
    sig, prices, cond,
    n_quantiles=int(n_quantiles), entry_delay=int(entry_delay),
    hold_days=int(hold_days),
)

if long_short.eq(0).all():
    st.warning("No trades generated with these parameters.")
    st.stop()

# Trim to first → last active day for chart cleanliness.
active = (quantile_returns != 0).any(axis=1)
first_active = active.idxmax() if active.any() else quantile_returns.index[0]
last_active = active[::-1].idxmax() if active.any() else quantile_returns.index[-1]
quantile_returns = quantile_returns.loc[first_active:last_active]
long_short = long_short.loc[first_active:last_active]

cum_quantiles = quantile_returns.cumsum() * 100
cum_ls = long_short.cumsum() * 100

ls_sharpe = sharpe(long_short)
ls_ann_ret = long_short.mean() * 252 * 100
ls_ann_vol = long_short.std(ddof=1) * np.sqrt(252) * 100
ls_max_dd = max_drawdown(cum_ls)


# ---------------------------------------------------------------------------
# Section header
# ---------------------------------------------------------------------------
period_str = (f"{quantile_returns.index[0].strftime('%b %d, %Y')} → "
              f"{quantile_returns.index[-1].strftime('%b %d, %Y')}")
st.markdown(
    f"""
    <div class="section-meta">
      Quantile L/S strategy · N = {int(n_quantiles)} · {period_str}
      <span class="pill">Long Q1 · Short Q{int(n_quantiles)}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("L/S Sharpe", f"{ls_sharpe:+.2f}",
          help="Annualized Sharpe of the (Q1 − Qn) daily-return series.")
c2.metric("L/S ann. return", f"{ls_ann_ret:+.1f}%",
          help="mean(daily_ls) × 252 — additive annualization.")
c3.metric("L/S ann. volatility", f"{ls_ann_vol:.1f}%",
          help="std(daily_ls) × √252.")
c4.metric("L/S max drawdown", f"{ls_max_dd:+.1f}%",
          help="Largest peak-to-trough drop on the cumulative L/S curve.")


# ---------------------------------------------------------------------------
# Chart 1: cumulative returns of each quantile
# ---------------------------------------------------------------------------
_hold_label = "daily rebucket" if int(hold_days) == 1 else f"hold {int(hold_days)}d"
chart_title(
    "Cumulative quantile returns",
    f"{cond} · N = {int(n_quantiles)} · entry +{int(entry_delay)}d · {_hold_label}",
)
fig_q = go.Figure()
n_cols = len(cum_quantiles.columns)
for i, col in enumerate(cum_quantiles.columns):
    # Q1 = lowest factor → typically expected outperformer (gets the "high" end of
    # the palette so it stands out); Qn = highest factor → expected underperformer.
    palette_idx = (i * len(QUANTILE_PALETTE)) // n_cols
    palette_idx = max(0, min(palette_idx, len(QUANTILE_PALETTE) - 1))
    color = QUANTILE_PALETTE[palette_idx]
    # Only the extremes (Q1 and Qn) are visible by default — the middle quantiles
    # are rendered as "legend only" so users can toggle them on when wanted.
    is_extreme = (i == 0) or (i == n_cols - 1)
    fig_q.add_trace(go.Scatter(
        x=cum_quantiles.index, y=cum_quantiles[col].values,
        name=col, mode="lines",
        line=dict(width=1.8, color=color),
        visible=True if is_extreme else "legendonly",
        hovertemplate=f"{col} · %{{x|%b %d, %Y}}<br>%{{y:+.1f}}%<extra></extra>",
    ))
fig_q.add_hline(y=0, line=dict(color="#94A3B8", width=0.6, dash="dot"))
style_axes(fig_q)
fig_q.update_layout(
    **base_layout(height=440, top_margin=80),
    hovermode="x unified",
    xaxis_title="date",
    yaxis_title="cumulative return — sum (%)",
    xaxis_title_font=dict(color=BRAND_SLATE, size=12),
    yaxis_title_font=dict(color=BRAND_SLATE, size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0,
                bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=BRAND_NAVY)),
)
st.plotly_chart(fig_q, width="stretch")


# ---------------------------------------------------------------------------
# Chart 2: cumulative L/S spread (Q1 − Qn)
# ---------------------------------------------------------------------------
chart_title(
    "Long-short spread",
    f"Long Q1 / Short Q{int(n_quantiles)} · {cond} · entry +{int(entry_delay)}d · {_hold_label}",
)
fig_ls = go.Figure()
fig_ls.add_trace(go.Scatter(
    x=cum_ls.index, y=cum_ls.values,
    name=f"Q1 − Q{int(n_quantiles)}",
    line=dict(width=2.0, color=BRAND_BLUE),
    fill="tozeroy", fillcolor="rgba(30, 64, 175, 0.10)",
    hovertemplate="%{x|%b %d, %Y}<br>%{y:+.1f}%<extra></extra>",
))
fig_ls.add_hline(y=0, line=dict(color="#94A3B8", width=0.6, dash="dot"))
style_axes(fig_ls)
fig_ls.update_layout(
    **base_layout(height=340, top_margin=70),
    hovermode="x unified",
    showlegend=False,
    xaxis_title="date",
    yaxis_title=f"Q1 − Q{int(n_quantiles)} cum return (%)",
    xaxis_title_font=dict(color=BRAND_SLATE, size=12),
    yaxis_title_font=dict(color=BRAND_SLATE, size=12),
)
st.plotly_chart(fig_ls, width="stretch")

st.caption(
    f"**Interpretation:** if the recall-probability factor has predictive power, Q1 "
    f"(low recall prob) should outperform Q{int(n_quantiles)} (high recall prob) — "
    f"so the Q1 − Q{int(n_quantiles)} spread should drift upward over time."
)
