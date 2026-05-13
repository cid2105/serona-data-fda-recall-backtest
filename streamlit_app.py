"""Serona Data — Recall Dataset Backtest.

Entry script: applies the global page config, brand CSS, and hero, then routes
to one of the views via ``st.navigation``.

Run with:  streamlit run streamlit_app.py
"""

import warnings

import streamlit as st

from app_common import check_password, inject_chrome

warnings.filterwarnings("ignore")

inject_chrome()

# Password gate — hero is already rendered above, so the gate shows under the brand.
if not check_password():
    st.stop()

home = st.Page("views/home.py",      title="Overview",          icon=":material/home:",        default=True)
threshold = st.Page("views/threshold.py", title="Threshold Backtest", icon=":material/tune:",        url_path="threshold")
top_k = st.Page("views/top_k.py",  title="Top-K Backtest",  icon=":material/leaderboard:", url_path="top-k")

pg = st.navigation([home, threshold, top_k])
pg.run()
