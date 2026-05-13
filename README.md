# Serona Data — Recall Dataset Backtest

**Live app:** [serona-data-fda-recall-backtest.streamlit.app](https://serona-data-fda-recall-backtest.streamlit.app)

Streamlit app that backtests two short strategies on FDA medical-device adverse-event predictions, using the model's per-event probabilities of a recall within 30 / 60 / 90 days (`p0` / `p1` / `p2`).

- **Threshold Backtest** — short any ticker whose factor exceeds an entry threshold; hold for `hold_days` or until factor drops below an exit threshold.
- **Top-K Backtest** — each AE date, short the K names with the highest factor (max-style ranking, same direction as the threshold strategy); hold for `hold_days`. No thresholds.

Both run a 50/50 market-neutral book (short basket + long SPY).

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Tests

```bash
pip install pandas numpy pytest
python -m pytest tests/ -q
```

CI runs the same tests on every push to `main` and every PR — see [.github/workflows/tests.yml](.github/workflows/tests.yml).

## Layout

```
streamlit_app.py        # entry — page config + nav
app_common.py           # shared chrome, palette, cached data loaders
strategy.py             # backtest engines + risk metrics (no IO)
views/
  home.py               # overview / dataset summary
  threshold.py          # Threshold Backtest page
  top_k.py              # Top-K Backtest page
data/                   # CoreCoverage tickers, predictions parquet, prices parquet
ticker_mapping/         # manufacturer → ticker mapping
tests/test_strategy.py  # unit + end-to-end synthetic validation
```
