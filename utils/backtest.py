"""
Rolling-origin backtesting.

A single train/test split on a time series tells you about one month's weather.
This walks several forecast origins forward through the calendar, retraining at
each one, so the reported error is an average over several genuinely
out-of-sample periods rather than one lucky window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_training_frame, period_index
from .models import BASELINES, LightGBMForecaster


def mase_scale(values: np.ndarray, m: int = 1) -> float:
    """
    Denominator for MASE: in-sample MAE of the m-step naive forecast.

    m=1 (last period) rather than m=52: with only two years of history the
    year-over-year differences are too few to give a stable scale.
    """
    v = np.asarray(values, dtype=float)
    if len(v) <= m:
        return np.nan
    scale = float(np.mean(np.abs(v[m:] - v[:-m])))
    return scale if scale > 0 else np.nan


def score(results: pd.DataFrame, scales: dict[str, float]) -> pd.DataFrame:
    """
    MAE, RMSE, wMAPE and MASE per model.

    wMAPE (total absolute error / total actual demand) is the one demand
    planners actually use: plain MAPE explodes on the zero-demand weeks, and
    this dataset has plenty.
    """
    rows = []
    for model, g in results.groupby("model", sort=False):
        err = g["prediction"] - g["units"]
        abs_err = np.abs(err)

        per_sku = []
        for sku, gs in g.groupby("StockCode", sort=False):
            s = scales.get(sku, np.nan)
            if s and not np.isnan(s):
                per_sku.append(np.mean(np.abs(gs["prediction"] - gs["units"])) / s)

        rows.append(
            {
                "model": model,
                "MAE": float(abs_err.mean()),
                "RMSE": float(np.sqrt((err**2).mean())),
                "wMAPE": float(abs_err.sum() / g["units"].sum()),
                "MASE": float(np.mean(per_sku)) if per_sku else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("wMAPE").reset_index(drop=True)


def rolling_origin_backtest(
    panel: pd.DataFrame,
    horizon: int = 4,
    n_folds: int = 8,
    stride: int = 4,
    train_stride: int = 1,
    progress=None,
):
    """
    Evaluate every model at `n_folds` origins, each `stride` periods apart.

    At each origin the ML model is retrained from scratch on data up to that
    origin only, so nothing after it can leak backwards.

    Returns (tidy per-observation results, summary table, model fitted on the
    last fold - kept so the app can show its feature importances).
    """
    panel = panel.sort_values(["StockCode", "date"])
    dates = period_index(panel)
    last_pos = len(dates) - 1

    origin_positions = sorted(
        last_pos - horizon - i * stride for i in range(n_folds)
    )
    if origin_positions[0] < 30:
        raise ValueError("Not enough history for the requested number of folds")

    frames = []
    fitted = None

    for fold, pos in enumerate(origin_positions, start=1):
        origin_date = dates[pos]
        future_dates = dates[pos + 1: pos + 1 + horizon]
        if progress:
            progress(fold, len(origin_positions), origin_date)

        train_panel = panel[panel["date"] <= origin_date]
        actuals = panel[panel["date"].isin(future_dates)][["StockCode", "date", "units"]]

        for name, fn in BASELINES.items():
            pred = fn(train_panel, pos, horizon, future_dates)
            merged = actuals.merge(pred, on=["StockCode", "date"], how="inner")
            merged["model"] = name
            merged["fold"] = fold
            merged["origin"] = origin_date
            frames.append(merged)

        training_frame = build_training_frame(train_panel, horizon=horizon, stride=train_stride)
        fitted = LightGBMForecaster().fit(training_frame)
        pred = fitted.predict(train_panel, pos, horizon, future_dates)
        merged = actuals.merge(pred, on=["StockCode", "date"], how="inner")
        merged["model"] = "LightGBM"
        merged["fold"] = fold
        merged["origin"] = origin_date
        frames.append(merged)

    results = pd.concat(frames, ignore_index=True)
    scales = {
        sku: mase_scale(g.sort_values("date")["units"].to_numpy())
        for sku, g in panel.groupby("StockCode", sort=False)
    }
    return results, score(results, scales), fitted
