"""Serona Data — Recall Dataset Backtest.

Entry script: applies the global page config, brand CSS, and hero, then routes
to one of the views via ``st.navigation``.

Run with:  streamlit run streamlit_app.py
"""

import warnings

import streamlit as st

from app_common import inject_chrome

warnings.filterwarnings("ignore")

inject_chrome()

home = st.Page("views/home.py",      title="Overview",          icon=":material/home:",        default=True)
threshold = st.Page("views/threshold.py", title="Threshold Backtest", icon=":material/tune:",        url_path="threshold")
bottom_k = st.Page("views/bottom_k.py",  title="Bottom-K Backtest",  icon=":material/leaderboard:", url_path="bottom-k")

pg = st.navigation([home, threshold, bottom_k])
pg.run()
