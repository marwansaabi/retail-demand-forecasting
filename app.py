"""Streamlit front-end for the retail demand forecasting backtest."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.backtest import rolling_origin_backtest
from utils.features import period_index

BASE = Path(__file__).parent
DATA = BASE / "data"
RESULTS = BASE / "results"

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#2563eb"
MODEL_COLORS = {
    "LightGBM": ACCENT,
    "Moving average (13w)": "#94a3b8",
    "Moving average (4w)": "#cbd5e1",
    "Naive (last week)": "#e2e8f0",
    "Seasonal naive (52w)": "#f1f5f9",
}


@st.cache_data(show_spinner=False)
def load_panel() -> pd.DataFrame:
    return pd.read_csv(DATA / "weekly_demand.csv", parse_dates=["date"])


@st.cache_data(show_spinner=False)
def load_cached_results():
    try:
        results = pd.read_csv(RESULTS / "backtest_results.csv", parse_dates=["date", "origin"])
        summary = pd.read_csv(RESULTS / "backtest_summary.csv")
        importance = pd.read_csv(RESULTS / "feature_importance.csv")
        return results, summary, importance
    except FileNotFoundError:
        return None, None, None


@st.cache_data(show_spinner="Running rolling-origin backtest...")
def run_backtest(horizon: int, n_folds: int):
    panel = load_panel()
    results, summary, model = rolling_origin_backtest(
        panel, horizon=horizon, n_folds=n_folds
    )
    return results, summary, model.feature_importance()


panel = load_panel()

st.sidebar.title("📦 Demand Forecasting")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "Results",
    ["Cached backtest", "Re-run backtest"],
    help="The cached run is the one reported in the README. Re-running lets you "
         "change the horizon and the number of folds.",
)

if mode == "Re-run backtest":
    horizon = st.sidebar.slider("Forecast horizon (weeks)", 1, 8, 4)
    n_folds = st.sidebar.slider("Backtest folds", 3, 12, 8)
    results, summary, importance = run_backtest(horizon, n_folds)
else:
    results, summary, importance = load_cached_results()
    horizon = 4
    if results is None:
        st.error("No cached results found. Run `python run_backtest.py` first.")
        st.stop()

st.sidebar.markdown("---")
st.sidebar.caption(
    f"{panel['StockCode'].nunique()} SKUs · "
    f"{panel['date'].nunique()} weeks · "
    f"{panel['date'].min():%b %Y} – {panel['date'].max():%b %Y}"
)

st.title("Retail Demand Forecasting")
st.caption(
    "Weekly SKU-level demand forecasting on real e-commerce transactions, "
    "benchmarked against the baselines that actually matter."
)

best = summary.iloc[0]
best_baseline = summary[summary["model"] != "LightGBM"].iloc[0]
lgbm = summary[summary["model"] == "LightGBM"].iloc[0]
lift = (best_baseline["wMAPE"] - lgbm["wMAPE"]) / best_baseline["wMAPE"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Best model", best["model"])
c2.metric("wMAPE", f"{lgbm['wMAPE']:.1%}")
c3.metric("MASE", f"{lgbm['MASE']:.3f}", help="Below 1 beats the in-sample naive forecast")
c4.metric("vs best baseline", f"{lift:+.1%}", help=f"against {best_baseline['model']}")

st.markdown("---")

tab_models, tab_sku, tab_signals, tab_method = st.tabs(
    ["📊 Model comparison", "🔍 Per-SKU forecasts", "🧠 What drives it", "📐 Method"]
)

with tab_models:
    st.subheader("Backtest results")
    st.caption(
        "Averaged over every fold and SKU. Lower is better on all four metrics."
    )

    show = summary.copy()
    show["wMAPE"] = (show["wMAPE"] * 100).round(1)
    show[["MAE", "RMSE"]] = show[["MAE", "RMSE"]].round(1)
    show["MASE"] = show["MASE"].round(3)
    st.dataframe(
        show.rename(columns={"wMAPE": "wMAPE (%)"}),
        use_container_width=True,
        hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            summary.sort_values("wMAPE"), x="wMAPE", y="model", orientation="h",
            template="simple_white", color="model", color_discrete_map=MODEL_COLORS,
            title="wMAPE by model (lower is better)",
        )
        fig.update_layout(showlegend=False, height=380, xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        per_fold = (
            results.assign(abs_err=lambda d: (d["prediction"] - d["units"]).abs())
            .groupby(["model", "fold"])
            .apply(lambda g: g["abs_err"].sum() / g["units"].sum(), include_groups=False)
            .rename("wMAPE").reset_index()
        )
        fig = px.line(
            per_fold, x="fold", y="wMAPE", color="model", markers=True,
            template="simple_white", color_discrete_map=MODEL_COLORS,
            title="Stability across folds",
        )
        fig.update_layout(height=380, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "The gap is real but modest, and that is the honest shape of this problem: "
        "a 13-week moving average is a strong opponent when demand is mostly a "
        "slowly-drifting level plus bulk-order noise."
    )

with tab_sku:
    st.subheader("Forecast vs actual, one SKU at a time")

    descriptions = panel.groupby("StockCode")["description"].last()
    volumes = panel.groupby("StockCode")["units"].sum().sort_values(ascending=False)
    options = list(volumes.index)
    labels = {s: f"{s} · {descriptions.get(s, '')[:40]}" for s in options}

    c1, c2 = st.columns([2, 1])
    sku = c1.selectbox("SKU", options, format_func=lambda s: labels[s])
    fold = c2.selectbox("Backtest fold", sorted(results["fold"].unique()))

    hist = panel[panel["StockCode"] == sku].sort_values("date")
    fold_res = results[(results["StockCode"] == sku) & (results["fold"] == fold)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["units"], name="Actual demand",
        line=dict(color="#0f172a", width=1.5),
    ))
    if len(fold_res):
        origin = fold_res["origin"].iloc[0]
        # add_vline() does arithmetic on the x value internally, which recent
        # pandas refuses to do on a Timestamp - add_shape with an ISO string
        # sidesteps it and renders identically.
        fig.add_shape(
            type="line", x0=origin.isoformat(), x1=origin.isoformat(),
            y0=0, y1=1, yref="paper",
            line=dict(color="#64748b", dash="dot"),
        )
        fig.add_annotation(
            x=origin.isoformat(), y=1.02, yref="paper", showarrow=False,
            text="forecast origin", font=dict(color="#64748b", size=11),
        )
        for model, g in fold_res.groupby("model"):
            g = g.sort_values("date")
            fig.add_trace(go.Scatter(
                x=g["date"], y=g["prediction"], name=model, mode="lines+markers",
                line=dict(color=MODEL_COLORS.get(model, "#999"),
                          dash="solid" if model == "LightGBM" else "dash"),
            ))
        window_start = origin - pd.Timedelta(weeks=26)
        fig.update_xaxes(range=[window_start, hist["date"].max()])

    fig.update_layout(template="simple_white", height=460,
                      yaxis_title="Units", legend_title="")
    st.plotly_chart(fig, use_container_width=True)

    if len(fold_res):
        err = (
            fold_res.assign(abs_err=lambda d: (d["prediction"] - d["units"]).abs())
            .groupby("model")
            .agg(MAE=("abs_err", "mean"), actual=("units", "sum"),
                 predicted=("prediction", "sum"))
            .round(1).reset_index()
        )
        st.dataframe(err, use_container_width=True, hide_index=True)

with tab_signals:
    st.subheader("Feature importance")
    st.caption("Split gain, from the model fitted on the final fold.")

    top = importance.head(20).sort_values("importance")
    fig = px.bar(top, x="importance", y="feature", orientation="h",
                 template="simple_white", color_discrete_sequence=[ACCENT])
    fig.update_layout(height=520, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "Recent rolling means carry most of the weight, which is why the moving "
        "average baselines are so hard to beat. The model's edge comes from the "
        "extras: the calendar position within the year, the price ratio that "
        "flags a SKU trading below its own norm, and the horizon step, which "
        "lets one model widen its own uncertainty as it forecasts further out."
    )

with tab_method:
    st.markdown(
        """
### Data

Two years of real transaction lines from a UK online retailer
([Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), UCI).
Cancellations, non-product codes and non-positive quantities are removed, the
remaining lines are aggregated to **weekly units per SKU**, and the 50 highest-
volume SKUs are kept.

Weeks with no invoice are explicit zeros, not gaps - a SKU that sold nothing
sold zero, and letting that go missing would quietly bias every average.

### Why weekly

This is a wholesale business. One customer ordering 80,000 units on a Tuesday
makes daily SKU demand close to unforecastable noise, and no model can time an
individual bulk order from sales history alone. Weekly buckets absorb that
timing jitter and match how replenishment is actually planned.

### Validation

Rolling-origin backtesting: 8 forecast origins, 4 weeks apart, each forecasting
4 weeks ahead. At every origin the model is retrained from scratch on data up to
that origin only.

Every feature is read **at the forecast origin**, never at the target week.
Calendar facts about the target are the one exception, because they genuinely
are known in advance.

The target transform (`log1p`) was chosen on a **separate development period**
(Nov 2010 - Mar 2011) so that the reported test period stayed untouched.

### Metrics

- **wMAPE** - total absolute error over total demand. What demand planners use;
  plain MAPE explodes on zero-demand weeks, and 12% of these weeks are zero.
- **MASE** - error scaled by the in-sample naive forecast. Below 1 means the
  model beats that naive rule; comparable across SKUs of very different volume.
- **MAE / RMSE** - absolute units, for reference. RMSE is dominated by a handful
  of bulk orders, which is exactly why it is not the headline.
        """
    )

st.markdown("---")
st.caption("Built by Marwan El Saabi · Data: UCI Online Retail II")
