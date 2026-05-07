"""Short-signal backtest dashboard.

Run with:  streamlit run streamlit_app.py
Knobs: condition, probability threshold, entry delay (trading days after AE date),
holding period (trading days). Uses sum-of-returns aggregation.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import alphalens as al

from strategy import (
    STRATEGIES, factor_for_condition, normalize_manu,
    simulate, sharpe, max_drawdown, basket_size_daily,
    portfolio_turnover_annualized, portfolio_turnover_daily,
    validate_prices, clean_prices,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MAPPING_CSV = ROOT / "ticker_mapping" / "fuzzy_matching" / "combined_mappings.csv"


@st.cache_data(show_spinner="Loading + joining predictions and tickers...")
def load_signals():
    cc = pd.read_csv(DATA_DIR / "CoreCoverage.tsv", sep="\t")
    pred = pd.read_parquet(DATA_DIR / "260420_preds_and_recalls_stored.parquet")
    mp = pd.read_csv(MAPPING_CSV)

    core = set(cc["Ticker"].str.upper())
    mp = mp[mp["ticker"].isin(core)][["normalized_manufacturer", "ticker"]].drop_duplicates()

    pred = pred.assign(_norm=pred["device_manufacturer_d_name"].map(normalize_manu))
    j = pred.merge(mp, left_on="_norm", right_on="normalized_manufacturer", how="inner")
    j["signal_date"] = pd.to_datetime(j["unified_ae_date"]).dt.normalize()

    sig = (
        j.groupby(["ticker", "signal_date"])[["prob_class_0", "prob_class_1", "prob_class_2"]]
         .mean()
         .reset_index()
    )
    return sig


@st.cache_data(show_spinner="Loading prices...")
def load_prices():
    px = pd.read_parquet(DATA_DIR / "yf_adj_close_2025.parquet")
    if "date" in px.columns:
        px = px.set_index("date")
    px.index = pd.DatetimeIndex(px.index).tz_localize(None).normalize()
    px.index.name = "date"
    return px.sort_index()


st.set_page_config(page_title="Serona Data Recall Dataset Backtest", layout="wide",
                   initial_sidebar_state="expanded")

# --- Brand palette and typography --------------------------------------------
BRAND_TEAL   = "#0F766E"   # primary
BRAND_NAVY   = "#0F172A"   # text / dark accent
BRAND_AMBER  = "#D97706"   # warm accent for negatives / alerts
BRAND_SLATE  = "#475569"   # secondary text
BRAND_MIST   = "#F1F5F9"   # subtle background

st.markdown(
    """
    <style>
      /* Hide Streamlit chrome that screams "demo" */
      #MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden; height: 0;}

      /* Tight, focused page padding */
      .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px;}

      /* Hero band */
      .serona-hero {
          display: flex; align-items: baseline; justify-content: space-between;
          padding: 0.4rem 0 1.0rem 0;
          border-bottom: 1px solid #E2E8F0;
          margin-bottom: 1.4rem;
      }
      .serona-wordmark {
          font-size: 1.9rem; font-weight: 700; letter-spacing: -0.025em;
          color: #0F172A;
      }
      .serona-wordmark .accent {color: #0F766E;}
      .serona-tag {
          color: #475569; font-size: 0.95rem; font-weight: 500;
      }

      /* Metric card polish */
      [data-testid="stMetric"] {
          background: #FFFFFF;
          border: 1px solid #E2E8F0;
          border-radius: 12px;
          padding: 1.0rem 1.2rem 0.85rem 1.2rem;
          box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
          transition: box-shadow 120ms ease;
      }
      [data-testid="stMetric"]:hover {
          box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
      }
      [data-testid="stMetricLabel"] {
          color: #475569; font-weight: 500; font-size: 0.85rem; text-transform: uppercase;
          letter-spacing: 0.04em;
      }
      [data-testid="stMetricValue"] {
          color: #0F172A; font-weight: 700; font-size: 1.85rem; letter-spacing: -0.01em;
      }

      /* Tab list */
      .stTabs [data-baseweb="tab-list"] {gap: 0.25rem; border-bottom: 1px solid #E2E8F0;}
      .stTabs [data-baseweb="tab"] {
          padding: 0.6rem 1.1rem; border-radius: 8px 8px 0 0;
          font-weight: 500; color: #475569;
      }
      .stTabs [aria-selected="true"] {color: #0F766E;}

      /* Sidebar polish */
      [data-testid="stSidebar"] {background: #F8FAFC; border-right: 1px solid #E2E8F0;}
      [data-testid="stSidebar"] h2 {
          color: #0F172A; font-size: 1.1rem; font-weight: 700;
          margin-top: 0; padding-bottom: 0.25rem; border-bottom: 1px solid #E2E8F0;
      }

      /* Section subheaders */
      h2, h3 {color: #0F172A; letter-spacing: -0.015em;}
      h3 {font-size: 1.15rem; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="serona-hero">
      <div class="serona-wordmark">Serona Data<span class="accent"> - </span>Recall Dataset Backtest</div>
      <div class="serona-tag">FDA medical-device adverse-event predictions</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Strategy rules", expanded=False):
    st.markdown(
        """
**Entry**

- When the chosen `condition(p0, p1, p2)` evaluates `factor > entry_threshold` for ticker `T`
  on day `d` (the AE date), go **short** at close of `d + entry_delay` trading days.
- Natural exit: close of `d + entry_delay + hold_days`.

**Early exit (optional)**

- If a later ticker-day for `T` has `factor < exit_threshold` (must be strictly less than the
  entry threshold), the position closes at close of `(that_signal_date + entry_delay)`,
  whichever comes first relative to the natural exit.
- Setting `exit_threshold = 0` disables early exit (probabilities are non-negative).
- Each entry signal owns its own window — a later re-entry signal opens a fresh independent position.

**Daily portfolio return** (true daily marks, no compounding)

- `held(d)` = set of tickers with at least one active short on day `d`.
- `short_book[d] = −mean( r[d, t]  for t in held(d) )`, where `r[d, t]` is the 1-day
  adjusted-close return of ticker `t`.
- `balanced[d] = 0.5 × short_book[d] + 0.5 × SPY_return[d]` (long SPY + short the recall basket).

**Cumulative & risk**

- Cumulative return = `cumsum(daily)` (additive, no compounding).
- Sharpe = `mean(daily) / std(daily) × √252`, over every trading day (0 on flat days).
- Max drawdown computed on the cumulative-percent curve.

**Data**: yfinance adjusted close (`auto_adjust=True`); SWAV dropped (delisted);
HOLX 5% NaN forward-filled up to 2 days. No stale-price anomalies detected.
        """
    )

sig = load_signals()
prices_raw = load_prices()
prices = clean_prices(prices_raw)
price_issues = validate_prices(prices)
n_dropped = prices_raw.shape[1] - prices.shape[1]
if n_dropped or price_issues["non_positive_obs"]:
    print(
        f"Price audit: dropped {n_dropped} all-NaN ticker(s) "
        f"({', '.join(price_issues['all_nan_tickers']) or '—'}), "
        f"non-positive obs: {price_issues['non_positive_obs']}."
    )

with st.sidebar:
    st.header("Strategy Knobs")
    st.html("<br/>")
    strategy_keys = list(STRATEGIES.keys())
    cond = st.selectbox("Short Trigger Rule (short fires on condition)", strategy_keys,
                        index=strategy_keys.index("p0 > t OR p1 > t"))
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
    hold_days = st.number_input("Holding period (trading days)", 1, 120, 40, 1)

@st.cache_data(show_spinner="Building factor data…")
def build_factor_data(_sig, _prices, condition, entry_delay, periods=(1, 5, 10), quantiles=5):
    """Aligns the continuous factor implied by `condition` to the entry trade date,
    then runs alphalens.get_clean_factor_and_forward_returns. _sig and _prices are
    underscore-prefixed so streamlit doesn't try to hash them — they're stable for the session."""
    sig, prices = _sig, _prices
    factor_vals = factor_for_condition(sig, condition)
    df = sig[["ticker", "signal_date"]].copy()
    df["factor"] = factor_vals.values
    df = df[df["ticker"].isin(prices.columns)]

    trading_days = prices.index
    pos = trading_days.searchsorted(df["signal_date"].values, side="right") + (entry_delay - 1)
    valid = pos < len(trading_days) - max(periods)
    df = df.iloc[valid].copy()
    df["trade_date"] = trading_days[pos[valid]]
    factor_series = df.groupby(["trade_date", "ticker"])["factor"].mean()

    return al.utils.get_clean_factor_and_forward_returns(
        factor=factor_series, prices=prices, periods=periods, quantiles=quantiles,
    )


tab_strategy, tab_factor = st.tabs(["Strategy backtest", "Factor analysis (alphalens)"])

# -------- Tab 1: Strategy backtest --------------------------------------------
with tab_strategy:
    s, daily_short, daily_bal = simulate(
        sig, prices, cond, threshold, entry_delay, hold_days,
        exit_threshold=exit_threshold,
    )

    if daily_short.empty:
        st.warning("No signals fired with these parameters.")
    else:
        # Trim the active window to the last day the pure-short basket actually had P&L
        # (otherwise the chart drags out a flat tail past the last position's exit).
        nonzero = daily_short[daily_short != 0]
        if len(nonzero):
            last_active = nonzero.index[-1]
            daily_short = daily_short.loc[:last_active]
            daily_bal = daily_bal.loc[:last_active]

        cum_short = daily_short.cumsum() * 100
        cum_bal = daily_bal.cumsum() * 100

        sharpe_bal = sharpe(daily_bal)
        # Annualized return / vol on the BALANCED portfolio (mean·252 / std·√252,
        # consistent with sum-of-returns aggregation and the Sharpe formula).
        ann_ret_bal = daily_bal.mean() * 252 * 100
        ann_vol_bal = daily_bal.std(ddof=1) * np.sqrt(252) * 100
        dd_bal = max_drawdown(cum_bal)
        turnover_daily_pct = portfolio_turnover_daily(s, daily_short.index)
        turnover_annual_pct = portfolio_turnover_annualized(s, daily_short.index)

        # Daily basket size (within the trimmed active window) — average across days that
        # actually had a position (gives "when the book is on, how big is it on average").
        basket_size_series = basket_size_daily(s, daily_short.index)
        active_basket = basket_size_series[basket_size_series > 0]
        avg_basket = active_basket.mean() if len(active_basket) else 0.0

        # Universe size = unique tickers with any prediction in the joined sig table
        # (i.e. the pool the strategy can choose from before any threshold is applied).
        universe_size = sig["ticker"].nunique()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Balanced Sharpe (ann.)", f"{sharpe_bal:+.2f}")
        c2.metric("Active trading days", f"{len(daily_short):,}")
        c3.metric("Avg # stocks in short book",
                  f"{int(round(avg_basket)):d}",
                  help="Mean of unique tickers held per day, averaged over days the book is active.")
        c4.metric("Universe size (recall-prob tickers)",
                  f"{universe_size:,}",
                  help="Unique tickers with at least one model prediction — the pool the "
                       "strategy can short before threshold gating.")

        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Balanced ann. return",
                  f"{ann_ret_bal:+.2f}%",
                  help="mean(daily_balanced) × 252 — additive (sum-of-returns) annualization.")
        d2.metric("Balanced ann. volatility",
                  f"{ann_vol_bal:.2f}%",
                  help="std(daily_balanced) × √252.")
        d3.metric("Balanced max drawdown",
                  f"{dd_bal:+.2f}%",
                  help="Largest peak-to-trough drop on the balanced cumulative-return curve.")
        d4.metric("Daily turnover",
                  f"{turnover_daily_pct:.2f}%",
                  help="Avg over active days of ½·Σ|Δw|. Reference: 20% means the book "
                       "turns over once a week (≈ 1/hold_days in steady state).")
        d5.metric("Annualized turnover",
                  f"{turnover_annual_pct:,.0f}%",
                  help="Daily turnover × 252. Reference: ~252/hold_days × 100% in steady state.")

        # Cumulative return + drawdown — plotly, interactive
        fig = make_subplots(rows=2, cols=1, row_heights=[0.66, 0.34],
                            vertical_spacing=0.10,
                            subplot_titles=("Cumulative return (%)", "Drawdown (pp)"))
        fig.add_trace(go.Scatter(x=cum_short.index, y=cum_short.values,
                                 name="Pure short basket",
                                 line=dict(width=2.0, color=BRAND_TEAL),
                                 hovertemplate="%{x|%b %d, %Y}<br>%{y:+.2f}%<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=cum_bal.index, y=cum_bal.values,
                                 name="Balanced (0.5 short + 0.5 SPY)",
                                 line=dict(width=2.0, color=BRAND_AMBER),
                                 hovertemplate="%{x|%b %d, %Y}<br>%{y:+.2f}%<extra></extra>"),
                      row=1, col=1)
        fig.add_hline(y=0, line=dict(color="#94A3B8", width=0.6, dash="dot"), row=1, col=1)

        dd_s = (cum_short - cum_short.cummax()).values
        dd_b = (cum_bal - cum_bal.cummax()).values
        fig.add_trace(go.Scatter(x=cum_short.index, y=dd_s, name="Pure short DD",
                                 fill="tozeroy", mode="lines",
                                 line=dict(color=BRAND_TEAL, width=1.0),
                                 fillcolor="rgba(15, 118, 110, 0.22)", showlegend=False,
                                 hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}pp<extra></extra>"),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=cum_bal.index, y=dd_b, name="Balanced DD",
                                 fill="tozeroy", mode="lines",
                                 line=dict(color=BRAND_AMBER, width=1.0),
                                 fillcolor="rgba(217, 119, 6, 0.20)", showlegend=False,
                                 hovertemplate="%{x|%b %d, %Y}<br>%{y:.2f}pp<extra></extra>"),
                      row=2, col=1)
        fig.add_hline(y=0, line=dict(color="#94A3B8", width=0.6, dash="dot"), row=2, col=1)
        fig.update_xaxes(showgrid=False, showline=True, linecolor="#E2E8F0",
                         tickfont=dict(color=BRAND_SLATE))
        fig.update_yaxes(gridcolor="#EEF2F7", zeroline=False,
                         tickfont=dict(color=BRAND_SLATE))
        fig.update_xaxes(title_text=None, row=2, col=1)
        fig.update_yaxes(title_text="cum return (%)", row=1, col=1,
                         title_font=dict(color=BRAND_SLATE, size=12))
        fig.update_yaxes(title_text="drawdown (pp)", row=2, col=1,
                         title_font=dict(color=BRAND_SLATE, size=12))
        fig.update_layout(
            height=520, hovermode="x unified",
            margin=dict(t=50, b=30, l=60, r=20),
            template="plotly_white",
            font=dict(family="-apple-system, BlinkMacSystemFont, Inter, sans-serif",
                      color=BRAND_NAVY, size=12),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0,
                        bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        for ann in fig["layout"]["annotations"]:
            ann["font"] = dict(color=BRAND_NAVY, size=13, family="-apple-system, sans-serif")
            ann["x"] = 0  # left-align subplot titles
            ann["xanchor"] = "left"
        st.plotly_chart(fig, width="stretch")

        # Quantile forward returns (depends only on condition + entry_delay)
        period_palette = ["#0F766E", "#0EA5E9", "#D97706"]  # teal / sky / amber
        try:
            fd_q = build_factor_data(sig, prices, cond, entry_delay)
            mq, _ = al.performance.mean_return_by_quantile(fd_q)
            mq_bps = mq * 1e4
            fig_q = go.Figure()
            for i, col in enumerate(mq_bps.columns):
                fig_q.add_trace(go.Bar(
                    x=[f"Q{int(q)}" for q in mq_bps.index],
                    y=mq_bps[col].values, name=str(col),
                    marker=dict(color=period_palette[i % len(period_palette)]),
                    hovertemplate=f"{col} · Q%{{x}}<br>%{{y:+.1f}} bps<extra></extra>",
                ))
            fig_q.add_hline(y=0, line=dict(color="#94A3B8", width=0.6, dash="dot"))
            fig_q.update_xaxes(showgrid=False, tickfont=dict(color=BRAND_SLATE))
            fig_q.update_yaxes(gridcolor="#EEF2F7", zeroline=False,
                               tickfont=dict(color=BRAND_SLATE))
            fig_q.update_layout(
                barmode="group", height=340, template="plotly_white",
                title=dict(text=f"Quantile forward returns — {cond}, entry +{entry_delay}td",
                           x=0, xanchor="left", font=dict(color=BRAND_NAVY, size=14)),
                xaxis_title="factor quantile (5 = highest factor / shorted bucket)",
                yaxis_title="mean fwd return (bps)",
                xaxis_title_font=dict(color=BRAND_SLATE, size=12),
                yaxis_title_font=dict(color=BRAND_SLATE, size=12),
                legend=dict(title="period", orientation="h",
                            yanchor="bottom", y=1.04, x=1, xanchor="right",
                            bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=60, b=40, l=60, r=20),
                font=dict(family="-apple-system, BlinkMacSystemFont, Inter, sans-serif",
                          color=BRAND_NAVY, size=12),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig_q, width="stretch")
        except Exception as e:
            st.caption(f"(could not compute quantile chart: {e})")

        with st.expander("Daily trade returns"):
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=daily_short.index, y=daily_short.values * 100,
                                  name="daily short basket return (%)",
                                  marker=dict(color=BRAND_TEAL),
                                  hovertemplate="%{x|%b %d, %Y}<br>%{y:+.2f}%<extra></extra>"))
            fig2.add_hline(y=0, line=dict(color="#94A3B8", width=0.6, dash="dot"))
            fig2.update_xaxes(showgrid=False, tickfont=dict(color=BRAND_SLATE))
            fig2.update_yaxes(gridcolor="#EEF2F7", zeroline=False,
                              tickfont=dict(color=BRAND_SLATE))
            fig2.update_layout(height=300, showlegend=False, template="plotly_white",
                               xaxis_title="trade date",
                               yaxis_title="daily short basket return (%)",
                               xaxis_title_font=dict(color=BRAND_SLATE, size=12),
                               yaxis_title_font=dict(color=BRAND_SLATE, size=12),
                               font=dict(family="-apple-system, sans-serif",
                                         color=BRAND_NAVY, size=12),
                               margin=dict(t=20, b=40, l=60, r=20),
                               plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig2, width="stretch")

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


# -------- Tab 2: Factor analysis ----------------------------------------------
with tab_factor:
    st.markdown(
        f"""
**Factor analysis** &nbsp;|&nbsp; condition: `{cond}` &nbsp;|&nbsp; alignment: `signal_date + {entry_delay} trading days`

- Continuous **factor** = the probability (or `max(...)` for OR conditions) underlying the chosen condition.
- Quantile bins (1–5) are built from the full factor distribution; **Q5** is the bucket the strategy actually shorts.
- Forward returns at **1D / 5D / 10D** are measured from the aligned trade date.
- **Threshold** and **holding period** do **not** affect this view — alphalens evaluates the
  factor's predictive power across *all* rows, regardless of where you put the cutoff.
        """
    )

    try:
        fd = build_factor_data(sig, prices, cond, entry_delay)
    except Exception as e:
        st.error(f"Could not build factor data: {e}")
        st.stop()

    # IC summary
    ic = al.performance.factor_information_coefficient(fd)
    n = len(ic)
    ic_summary = pd.DataFrame({
        "IC mean":   ic.mean(),
        "IC std":    ic.std(),
        "t-stat":    ic.mean() / (ic.std() / np.sqrt(n)),
        "Ann. IR":   ic.mean() / ic.std() * np.sqrt(252),
        "% positive": (ic > 0).mean(),
    }).round(4)
    st.subheader("Spearman rank IC (factor → forward return)")
    st.dataframe(ic_summary, width="stretch")
    st.caption(
        "Sign convention: short fires when factor is HIGH (factor > threshold). "
        "Negative IC means high factor → lower forward return — i.e. the strategy works."
    )

    # Brand-coordinated palette for tab-2 charts (period bars + quantile lines)
    period_palette_t2 = ["#0F766E", "#0EA5E9", "#D97706"]   # teal / sky / amber
    quantile_palette = ["#7F1D1D", "#DC2626", "#94A3B8", "#10B981", "#0F766E"]  # red→teal Q1→Q5

    # Mean return by quantile (basis points)
    mq, _ = al.performance.mean_return_by_quantile(fd)
    mq_bps = mq * 1e4
    qcol, ccol = st.columns(2)
    with qcol:
        st.subheader("Mean forward return by quantile (bps)")
        fig_q2 = go.Figure()
        for i, col in enumerate(mq_bps.columns):
            fig_q2.add_trace(go.Bar(
                x=[f"Q{int(q)}" for q in mq_bps.index],
                y=mq_bps[col].values, name=str(col),
                marker=dict(color=period_palette_t2[i % len(period_palette_t2)]),
                hovertemplate=f"{col} · Q%{{x}}<br>%{{y:+.1f}} bps<extra></extra>",
            ))
        fig_q2.add_hline(y=0, line=dict(color="#94A3B8", width=0.6, dash="dot"))
        fig_q2.update_xaxes(showgrid=False, tickfont=dict(color=BRAND_SLATE))
        fig_q2.update_yaxes(gridcolor="#EEF2F7", zeroline=False, tickfont=dict(color=BRAND_SLATE))
        fig_q2.update_layout(barmode="group", height=380, template="plotly_white",
                             xaxis_title="factor quantile",
                             yaxis_title="mean fwd return (bps)",
                             xaxis_title_font=dict(color=BRAND_SLATE, size=12),
                             yaxis_title_font=dict(color=BRAND_SLATE, size=12),
                             legend=dict(title="period", bgcolor="rgba(0,0,0,0)"),
                             font=dict(family="-apple-system, sans-serif",
                                       color=BRAND_NAVY, size=12),
                             margin=dict(t=20, b=40, l=60, r=20),
                             plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_q2, width="stretch")

    # Cumulative return by quantile (1D forward)
    quant_daily, _ = al.performance.mean_return_by_quantile(fd, by_date=True)
    with ccol:
        st.subheader("Cumulative quantile returns (1D)")
        cum_q = quant_daily["1D"].unstack("factor_quantile").sort_index().fillna(0).cumsum() * 100
        fig_c = go.Figure()
        for q in cum_q.columns:
            color = quantile_palette[int(q) - 1] if 1 <= int(q) <= 5 else BRAND_SLATE
            fig_c.add_trace(go.Scatter(x=cum_q.index, y=cum_q[q].values,
                                       name=f"Q{int(q)}", mode="lines",
                                       line=dict(width=1.8, color=color),
                                       hovertemplate="%{x|%b %d, %Y}<br>%{y:+.2f}%<extra></extra>"))
        fig_c.add_hline(y=0, line=dict(color="#94A3B8", width=0.6, dash="dot"))
        fig_c.update_xaxes(showgrid=False, tickfont=dict(color=BRAND_SLATE))
        fig_c.update_yaxes(gridcolor="#EEF2F7", zeroline=False, tickfont=dict(color=BRAND_SLATE))
        fig_c.update_layout(height=380, hovermode="x unified", template="plotly_white",
                            xaxis_title="date",
                            yaxis_title="cumulative 1D return — sum (%)",
                            xaxis_title_font=dict(color=BRAND_SLATE, size=12),
                            yaxis_title_font=dict(color=BRAND_SLATE, size=12),
                            font=dict(family="-apple-system, sans-serif",
                                      color=BRAND_NAVY, size=12),
                            legend=dict(bgcolor="rgba(0,0,0,0)"),
                            margin=dict(t=20, b=40, l=60, r=20),
                            plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_c, width="stretch")

    # Top − Bottom spread, one panel per period
    st.subheader("Top quantile (Q5) minus bottom quantile (Q1) — spread in bps")
    periods = ["1D", "5D", "10D"]
    fig_sp = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                           subplot_titles=[f"{p} spread" for p in periods])
    for i, period in enumerate(periods, start=1):
        wide = quant_daily[period].unstack("factor_quantile").sort_index()
        if 5 in wide.columns and 1 in wide.columns:
            tmb = (wide[5] - wide[1]) * 1e4
            ma = tmb.rolling(21).mean()
            fig_sp.add_trace(go.Scatter(x=tmb.index, y=tmb.values, name="spread",
                                        line=dict(color=BRAND_TEAL, width=0.7),
                                        opacity=0.55, showlegend=(i == 1),
                                        hovertemplate="%{x|%b %d}<br>%{y:+.1f} bps<extra></extra>"),
                             row=i, col=1)
            fig_sp.add_trace(go.Scatter(x=ma.index, y=ma.values, name="21d MA",
                                        line=dict(color=BRAND_AMBER, width=1.8),
                                        showlegend=(i == 1),
                                        hovertemplate="%{x|%b %d}<br>%{y:+.1f} bps (21d MA)<extra></extra>"),
                             row=i, col=1)
            fig_sp.add_hline(y=0, line=dict(color="#94A3B8", width=0.6, dash="dot"), row=i, col=1)
            fig_sp.update_yaxes(title_text=f"{period} bps", row=i, col=1,
                                title_font=dict(color=BRAND_SLATE, size=12))
    fig_sp.update_xaxes(title_text=None, showgrid=False, tickfont=dict(color=BRAND_SLATE))
    fig_sp.update_yaxes(gridcolor="#EEF2F7", zeroline=False, tickfont=dict(color=BRAND_SLATE))
    fig_sp.update_layout(height=620, hovermode="x unified", template="plotly_white",
                         legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0,
                                     bgcolor="rgba(0,0,0,0)"),
                         font=dict(family="-apple-system, sans-serif",
                                   color=BRAND_NAVY, size=12),
                         margin=dict(t=50, b=40, l=60, r=20),
                         plot_bgcolor="white", paper_bgcolor="white")
    for ann in fig_sp["layout"]["annotations"]:
        ann["font"] = dict(color=BRAND_NAVY, size=12, family="-apple-system, sans-serif")
        ann["x"] = 0
        ann["xanchor"] = "left"
    st.plotly_chart(fig_sp, width="stretch")
