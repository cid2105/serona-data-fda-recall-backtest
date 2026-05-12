"""Shared chrome, brand palette, and cached data loaders for the Serona Data
Recall Dataset Backtest streamlit app.

Each page imports the helpers it needs; ``inject_chrome()`` is called once from
``streamlit_app.py`` so the hero + global CSS render on every page navigation.
"""

import hmac
from pathlib import Path

import pandas as pd
import streamlit as st

from strategy import normalize_manu, clean_prices

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MAPPING_CSV = ROOT / "ticker_mapping" / "fuzzy_matching" / "combined_mappings.csv"

# ---------------------------------------------------------------------------
# Brand palette — Serona blue
# ---------------------------------------------------------------------------
BRAND_BLUE       = "#1E40AF"   # primary
BRAND_BLUE_LIGHT = "#3B82F6"   # accent
BRAND_NAVY       = "#0F172A"   # text / dark
BRAND_AMBER      = "#D97706"   # warm contrast (losses, benchmark)
BRAND_SLATE      = "#475569"   # secondary text
BRAND_MIST       = "#F1F5F9"   # subtle bg
BRAND_BLUE_TINT  = "#DBEAFE"   # very light blue (pill bg, subtle highlight)

# Used for the alphalens forward-period grouping + quantile lines
PERIOD_PALETTE   = [BRAND_BLUE, BRAND_BLUE_LIGHT, BRAND_AMBER]
QUANTILE_PALETTE = ["#7F1D1D", "#DC2626", "#94A3B8", BRAND_BLUE_LIGHT, BRAND_BLUE]

PLOTLY_FONT = dict(
    family="-apple-system, BlinkMacSystemFont, Inter, sans-serif",
    color=BRAND_NAVY, size=12,
)


# ---------------------------------------------------------------------------
# Page chrome — set_page_config + CSS + hero
# ---------------------------------------------------------------------------

_CSS = f"""
<style>
  /* Hide demo-y Streamlit chrome but keep the header visible (sidebar toggle + Deploy live there). */
  #MainMenu, footer {{visibility: hidden; height: 0;}}
  header[data-testid="stHeader"] {{background: rgba(255,255,255,0); height: 2.5rem;}}

  /* Tight, focused page padding */
  .block-container {{padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px;}}

  /* Hero band — logo + subtitle on the left, descriptive tag on the right */
  .serona-hero {{
      display: flex; align-items: center; justify-content: space-between;
      gap: 1.4rem;
      padding: 0.4rem 0 1.0rem 0;
      border-bottom: 2px solid {BRAND_BLUE};
      margin-bottom: 1.4rem;
  }}
  .serona-brand {{display: flex; align-items: center; gap: 0.9rem;}}
  .serona-logo {{height: 38px; width: auto; display: block;}}
  .serona-title {{
      font-size: 1.15rem; font-weight: 600; letter-spacing: -0.01em;
      color: {BRAND_NAVY}; line-height: 1.1;
  }}
  .serona-title .accent {{color: {BRAND_BLUE};}}
  .serona-tag {{
      color: {BRAND_SLATE}; font-size: 0.9rem; font-weight: 500; text-align: right;
  }}

  /* Section header above metrics — period range + strategy pill */
  .section-meta {{
      color: {BRAND_SLATE}; font-size: 0.78rem; font-weight: 600;
      letter-spacing: 0.08em; text-transform: uppercase;
      margin: 0.6rem 0 0.9rem 0;
  }}
  .section-meta .pill {{
      display: inline-block; background: {BRAND_BLUE_TINT}; color: {BRAND_BLUE};
      padding: 2px 10px; border-radius: 999px; margin-left: 0.6rem;
      letter-spacing: 0.04em; font-size: 0.72rem;
  }}

  /* Metric card polish */
  [data-testid="stMetric"] {{
      background: #FFFFFF;
      border: 1px solid #E2E8F0;
      border-radius: 12px;
      padding: 1.0rem 1.2rem 0.85rem 1.2rem;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
      transition: box-shadow 120ms ease, border-color 120ms ease;
  }}
  [data-testid="stMetric"]:hover {{
      box-shadow: 0 4px 12px rgba(30, 64, 175, 0.10);
      border-color: {BRAND_BLUE_LIGHT};
  }}
  [data-testid="stMetricLabel"] {{
      color: {BRAND_SLATE}; font-weight: 500; font-size: 0.85rem; text-transform: uppercase;
      letter-spacing: 0.04em;
  }}
  [data-testid="stMetricValue"] {{
      color: {BRAND_NAVY}; font-weight: 700; font-size: 1.85rem; letter-spacing: -0.01em;
  }}

  /* Sidebar polish — subtle blue tint to nod at the brand */
  [data-testid="stSidebar"] {{
      background: #F4F8FF; border-right: 1px solid #DAE6F5;
  }}
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {{
      color: {BRAND_NAVY}; font-size: 1.05rem; font-weight: 700;
      margin-top: 0.3rem; padding-bottom: 0.25rem;
      border-bottom: 1px solid #DAE6F5;
  }}
  /* Widget labels (selectbox, slider, number_input titles) — navy for legibility
     against the tinted-blue sidebar background. */
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
  [data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
  [data-testid="stSidebar"] label p {{
      color: {BRAND_NAVY} !important; font-weight: 500;
  }}

  /* Section subheaders */
  h2, h3 {{color: {BRAND_NAVY}; letter-spacing: -0.015em;}}
  h3 {{font-size: 1.15rem; font-weight: 600;}}

  /* Page-nav (sidebar) — non-active links navy, active link brand blue.
     Span selectors override any inline color Streamlit injects on the link text. */
  [data-testid="stSidebarNav"] a,
  [data-testid="stSidebarNav"] a span {{
      color: {BRAND_NAVY} !important; font-weight: 500;
  }}
  [data-testid="stSidebarNav"] a:hover,
  [data-testid="stSidebarNav"] a:hover span {{
      color: {BRAND_BLUE} !important;
  }}
  [data-testid="stSidebarNav"] a[aria-current="page"],
  [data-testid="stSidebarNav"] a[aria-current="page"] span {{
      color: {BRAND_BLUE} !important; font-weight: 600;
  }}

  /* Chart title — HTML header used in place of inline Plotly title.
     Brand-blue left bar accent + bold navy primary + lighter slate params. */
  .chart-title {{
      display: flex; align-items: baseline; flex-wrap: wrap;
      gap: 0.6rem 0.8rem;
      margin: 1.2rem 0 0.5rem 0;
      padding-left: 0.7rem;
      border-left: 3px solid {BRAND_BLUE};
      line-height: 1.25;
  }}
  .chart-title-name {{
      color: {BRAND_NAVY}; font-size: 1.05rem; font-weight: 700;
      letter-spacing: -0.01em;
  }}
  .chart-title-meta {{
      color: {BRAND_SLATE}; font-size: 0.88rem; font-weight: 500;
      letter-spacing: 0;
  }}
</style>
"""

_HERO_HTML = """
<div class="serona-hero">
  <div class="serona-brand">
    <img class="serona-logo"
         src="https://seronadata.com/assets/serona-logo-full-BhluY9ja.png"
         alt="Serona Data" />
    <div class="serona-title">Recall Dataset Backtest</div>
  </div>
  <div class="serona-tag">FDA medical-device adverse-event predictions</div>
</div>
"""


def inject_chrome():
    """Set page config + global CSS + render the hero. Call once at the top of the entry script."""
    st.set_page_config(
        page_title="Serona Data — Recall Dataset Backtest",
        layout="wide", initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(_HERO_HTML, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Password gate
# ---------------------------------------------------------------------------

def _is_localhost() -> bool:
    """True iff the app is being served from localhost — i.e. local dev, not the
    public Streamlit Cloud deployment. Used to skip the password gate during dev."""
    try:
        host = (st.context.headers.get("Host") or "").split(":")[0].lower()
    except Exception:
        return False
    return host in ("localhost", "127.0.0.1", "0.0.0.0", "::1")


def check_password() -> bool:
    """Block the app until the shared password (``st.secrets['app_password']``)
    is entered correctly. Returns True once authenticated; renders an input form
    and returns False otherwise — the caller should ``st.stop()`` in that case.

    The gate is **bypassed automatically on localhost** so ``streamlit run`` on
    your laptop never prompts. The deployed app on ``*.streamlit.app`` still
    requires the password.

    Configure the password by adding to the Streamlit Cloud Secrets pane:

        app_password = "your-shared-password"

    Authentication persists for the session via ``st.session_state`` — users
    aren't re-prompted when navigating between pages.
    """
    if _is_localhost():
        return True

    if st.session_state.get("_password_correct"):
        return True

    # Read the expected password; handle both "file missing" and "key missing".
    # st.secrets[...] raises StreamlitSecretNotFoundError if the secrets file
    # doesn't exist (different exception type than a plain KeyError), so catch
    # broadly here — either way it's a user-config issue with the same fix.
    try:
        expected_password = st.secrets["app_password"]
    except Exception:
        st.error(
            "App password not configured. Add `app_password = \"...\"` to "
            "the Streamlit Cloud Secrets pane (Settings → Secrets)."
        )
        return False

    def _on_submit():
        if hmac.compare_digest(
            st.session_state.get("_password_input", ""),
            expected_password,
        ):
            st.session_state["_password_correct"] = True
            # Don't keep the cleartext password around in session state.
            st.session_state.pop("_password_input", None)
        else:
            st.session_state["_password_correct"] = False

    st.markdown(
        f"""
        <div style="max-width: 420px; margin: 3rem auto 1rem auto;
                    padding: 1.6rem 1.8rem 1.2rem 1.8rem;
                    border: 1px solid #DAE6F5; border-radius: 14px;
                    background: linear-gradient(180deg, #FFFFFF 0%, {BRAND_BLUE_TINT}33 100%);
                    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06);">
          <div style="font-weight: 700; color: {BRAND_NAVY}; font-size: 1.15rem;
                      letter-spacing: -0.01em; margin-bottom: 0.35rem;">
            Restricted access
          </div>
          <div style="color: {BRAND_SLATE}; font-size: 0.92rem; line-height: 1.5;">
            Enter the access password to view the backtest dashboard.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Center the input under the card via columns trick.
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.text_input(
            "Password",
            type="password",
            key="_password_input",
            on_change=_on_submit,
            label_visibility="collapsed",
            placeholder="Password",
        )
        if st.session_state.get("_password_correct") is False:
            st.error("Incorrect password.")
    return False


# ---------------------------------------------------------------------------
# Cached data loaders (shared across pages)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading + joining predictions and tickers...")
def load_signals():
    cc = pd.read_csv(DATA_DIR / "CoreCoverage.tsv", sep="\t")
    pred = pd.read_parquet(DATA_DIR / "predictions_with_base.parquet")
    mp = pd.read_csv(MAPPING_CSV)

    core = set(cc["Ticker"].str.upper())
    mp = mp[mp["ticker"].isin(core)][["normalized_manufacturer", "ticker"]].drop_duplicates()

    pred = pred.assign(_norm=pred["device_manufacturer_d_name"].map(normalize_manu))
    j = pred.merge(mp, left_on="_norm", right_on="normalized_manufacturer", how="inner")
    j["signal_date"] = pd.to_datetime(j["unified_ae_date"]).dt.normalize()

    sig = (j.groupby(["ticker", "signal_date"])[
              ["prob_class_0", "prob_class_1", "prob_class_2"]]
              .mean().reset_index())
    return sig


@st.cache_data(show_spinner="Loading prices...")
def _load_prices_raw():
    px = pd.read_parquet(DATA_DIR / "yf_adj_close.parquet")
    if "date" in px.columns:
        px = px.set_index("date")
    px.index = pd.DatetimeIndex(px.index).tz_localize(None).normalize()
    px.index.name = "date"
    return px.sort_index()


@st.cache_data(show_spinner=False)
def load_prices():
    """Returns cleaned prices (forward-fills short NaN runs, drops all-NaN tickers)."""
    return clean_prices(_load_prices_raw())


def show_data_health():
    """Surface SPY data-quality issues at the top of strategy pages."""
    prices = load_prices()
    if "SPY" in prices.columns:
        spy_nan_dates = prices.index[prices["SPY"].isna()]
        if len(spy_nan_dates):
            st.caption(
                f":warning: SPY benchmark has {len(spy_nan_dates)} missing day(s) — "
                f"treating those as 0% return. Affected: "
                f"{', '.join(d.strftime('%Y-%m-%d') for d in spy_nan_dates[:5])}"
                f"{' …' if len(spy_nan_dates) > 5 else ''}"
            )


# ---------------------------------------------------------------------------
# Plotly layout helpers — keep the look consistent across pages
# ---------------------------------------------------------------------------

def base_layout(*, height: int = 380, top_margin: int = 50):
    """Return common Plotly ``update_layout`` kwargs with brand styling.

    Chart titles are rendered as HTML headers via :func:`chart_title` instead of
    Plotly's built-in title — keeps the chart config purely about data, frees up
    the top margin, and gives us full CSS control over the heading look.
    """
    return dict(
        height=height, template="plotly_white",
        font=PLOTLY_FONT,
        margin=dict(t=top_margin, b=40, l=60, r=20),
        # Subtle off-white plot area so the (white-bg, blue-bordered) legend stands out
        # against it. paper_bgcolor stays pure white so titles + margins feel clean.
        plot_bgcolor="#F8FAFC", paper_bgcolor="white",
    )


def chart_title(name: str, subtitle: str | None = None) -> None:
    """Render a brand-styled HTML header above a chart. Use this in place of the
    Plotly inline title for a cleaner, more striking look.

    Args:
        name: Primary title (bold navy).
        subtitle: Optional metadata string (e.g. parameter values), rendered in
            lighter slate after the primary title.
    """
    parts = [f'<span class="chart-title-name">{name}</span>']
    if subtitle:
        parts.append(f'<span class="chart-title-meta">{subtitle}</span>')
    st.markdown(
        f'<div class="chart-title">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def style_axes(fig, *, percent_axis: bool = True):
    """Apply consistent axis styling to a Plotly figure."""
    fig.update_xaxes(showgrid=False, showline=True, linecolor="#E2E8F0",
                     tickfont=dict(color=BRAND_SLATE))
    yaxis_kwargs = dict(gridcolor="#EEF2F7", zeroline=False,
                        tickfont=dict(color=BRAND_SLATE))
    if percent_axis:
        yaxis_kwargs["tickformat"] = ".0f"
    fig.update_yaxes(**yaxis_kwargs)


# ---------------------------------------------------------------------------
# Active-window helper — shared by both backtest pages
# ---------------------------------------------------------------------------

def trim_to_active_window(daily_short, daily_bal, basket_full):
    """Trim daily series to the first→last day the basket actually held names. Avoids
    leading/trailing zero days biasing Sharpe / vol toward 0 and stretching charts.

    Returns (daily_short_trimmed, daily_bal_trimmed, basket_size_trimmed).
    """
    active_dates = basket_full[basket_full > 0].index
    if not len(active_dates):
        return daily_short, daily_bal, basket_full
    first_active, last_active = active_dates[0], active_dates[-1]
    return (
        daily_short.loc[first_active:last_active],
        daily_bal.loc[first_active:last_active],
        basket_full.loc[first_active:last_active],
    )


__all__ = [
    "BRAND_BLUE", "BRAND_BLUE_LIGHT", "BRAND_NAVY", "BRAND_AMBER",
    "BRAND_SLATE", "BRAND_MIST", "BRAND_BLUE_TINT",
    "PERIOD_PALETTE", "QUANTILE_PALETTE", "PLOTLY_FONT",
    "inject_chrome", "check_password",
    "load_signals", "load_prices", "show_data_health",
    "base_layout", "chart_title", "style_axes", "trim_to_active_window",
]
