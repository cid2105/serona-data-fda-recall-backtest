"""Overview page — short intro to the dataset and the two backtests."""

import streamlit as st

from app_common import (
    BRAND_BLUE, BRAND_BLUE_LIGHT, BRAND_BLUE_TINT, BRAND_SLATE,
    load_signals, load_prices, show_data_health,
)

show_data_health()

sig = load_signals()
prices = load_prices()

universe_size = sig["ticker"].nunique()
n_signals = len(sig)
sig_first = sig["signal_date"].min()
sig_last = sig["signal_date"].max()
n_days = len(prices.index)
years_history = (sig_last - sig_first).days / 365.25

# ---- High-level KPIs about the underlying dataset (NOT a backtest) ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tickers in universe", f"{universe_size:,}",
          help="Unique CoreCoverage tickers with at least one model prediction.")
c2.metric("AE signal rows", f"{n_signals:,}",
          help="Distinct (ticker, AE date) signal rows after joining predictions to tickers.")
c3.metric("AE date range",
          f"{sig_first.strftime('%b %Y')} → {sig_last.strftime('%b %Y')}",
          help=f"Earliest: {sig_first.strftime('%b %d, %Y')} · "
               f"Latest: {sig_last.strftime('%b %d, %Y')}.")
c4.metric("Years of history", f"{years_history:.1f} yrs",
          help=f"Span from earliest to latest AE date. Price panel covers "
               f"{n_days:,} trading days for backtest runway on both ends.")


# ---- What the probability classes mean ------------------------------------
st.markdown(
    f"""
    <div style="margin: 1.6rem 0 0.4rem 0; color: {BRAND_SLATE};
                font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em;
                text-transform: uppercase;">
      What the probability classes mean
    </div>
    <div style="border: 1px solid #DAE6F5; border-radius: 12px;
                padding: 0.95rem 1.2rem; background: {BRAND_BLUE_TINT}33;
                color: #334155; font-size: 0.92rem; line-height: 1.55;">
      Each <code>(ticker, AE-date)</code> signal carries three model-predicted probabilities
      that a recall will follow within a given horizon:
      <ul style="margin: 0.4rem 0 0.2rem 1.1rem;">
        <li><code style="color: {BRAND_BLUE}; font-weight: 600;">p0</code> — probability of a recall within <strong>30 days</strong></li>
        <li><code style="color: {BRAND_BLUE}; font-weight: 600;">p1</code> — probability of a recall within <strong>60 days</strong></li>
        <li><code style="color: {BRAND_BLUE}; font-weight: 600;">p2</code> — probability of a recall within <strong>90 days</strong></li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    f"""
    <div style="margin: 1.6rem 0 0.4rem 0; color: {BRAND_SLATE};
                font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em;
                text-transform: uppercase;">
      Two backtests in this app
    </div>
    """,
    unsafe_allow_html=True,
)

c_left, c_right = st.columns(2)
_card_style = (
    f"border: 1px solid #DAE6F5; border-radius: 12px; padding: 1.1rem 1.3rem; "
    f"background: linear-gradient(180deg, #FFFFFF 0%, {BRAND_BLUE_TINT}55 100%);"
    f"box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);"
)
with c_left:
    st.markdown(
        f"""
        <div style="{_card_style}">
          <div style="font-weight: 700; font-size: 1.1rem; color: {BRAND_BLUE}; margin-bottom: 0.4rem;">
            <a href="/threshold" target="_self">Threshold Backtest</a>
          </div>
          <div style="color: #334155; font-size: 0.92rem; line-height: 1.5;">
            Each AE-date signal whose factor exceeds the entry threshold opens a short
            position; an optional exit threshold closes the position early when the factor
            drops back below it. Tunes <em>entry threshold</em>, <em>exit threshold</em>,
            <em>entry delay</em>, <em>holding period</em>, and the <em>trigger rule</em>
            (which probability classes count).
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c_right:
    st.markdown(
        f"""
        <div style="{_card_style}">
          <div style="font-weight: 700; font-size: 1.1rem; color: {BRAND_BLUE_LIGHT}; margin-bottom: 0.4rem;">
            <a href="/bottom-k" target="_self">Bottom-K Backtest</a>
          </div>
          <div style="color: #334155; font-size: 0.92rem; line-height: 1.5;">
            Each AE date, rank tickers by the trigger-rule factor (using <em>min</em> for
            multi-class rules) and short the K names with the lowest factor. No thresholds —
            pure mechanical bottom-K selection. Tunes <em>K</em>, <em>entry delay</em>,
            <em>holding period</em>, and the <em>trigger rule</em>.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
