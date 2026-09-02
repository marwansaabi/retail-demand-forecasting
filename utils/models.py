"""
Forecasting models: four baselines and one gradient-boosted global model.

The baselines are not filler. "wMAPE 38%" means nothing on its own; "38% vs
47% for the four-week moving average" is a claim about skill, and it is the
only kind of claim worth putting on a chart.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import (
    FEATURE_COLUMNS,
    as_series_map,
    build_prediction_frame,
    period_index,
)


def _skeleton(skus, future_dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"StockCode": s, "date": d, "horizon_step": k + 1}
            for s in skus
            for k, d in enumerate(future_dates)
        ]
    )


def naive(panel, origin_pos: int, horizon: int, future_dates) -> pd.DataFrame:
    """Next week looks like this week."""
    sm = as_series_map(panel)
    out = _skeleton(sm.keys(), future_dates)
    last = {s: float(v.to_numpy()[origin_pos]) for s, v in sm.items()}
    out["prediction"] = out["StockCode"].map(last)
    return out


def seasonal_naive(panel, origin_pos: int, horizon: int, future_dates, m: int = 52):
    """
    Same week last year. The honest seasonal benchmark for retail, and a hard
    one to beat around Christmas. Falls back to the plain naive when there is
    not yet a year of history.
    """
    sm = as_series_map(panel)
    out = _skeleton(sm.keys(), future_dates)
    preds = []
    for _, row in out.iterrows():
        v = sm[row["StockCode"]].to_numpy(dtype=float)
        k = int(row["horizon_step"])
        idx = origin_pos + k - m
        preds.append(float(v[idx]) if idx >= 0 else float(v[origin_pos]))
    out["prediction"] = preds
    return out


def _moving_average(panel, origin_pos, horizon, future_dates, window):
    sm = as_series_map(panel)
    out = _skeleton(sm.keys(), future_dates)
    means = {
        s: float(v.to_numpy(dtype=float)[max(0, origin_pos - window + 1): origin_pos + 1].mean())
        for s, v in sm.items()
    }
    out["prediction"] = out["StockCode"].map(means)
    return out


def moving_average_4(panel, origin_pos, horizon, future_dates):
    return _moving_average(panel, origin_pos, horizon, future_dates, 4)


def moving_average_13(panel, origin_pos, horizon, future_dates):
    return _moving_average(panel, origin_pos, horizon, future_dates, 13)


class LightGBMForecaster:
    """
    One global model across all SKUs, with the horizon step as a feature.

    Global beats per-SKU here: each series is only ~100 weeks long, so pooling
    lets a quiet SKU borrow the seasonal shape learned from the busy ones. SKU
    identity stays available as a categorical feature.

    The target is modelled as log1p(units). Weekly demand is non-negative and
    heavily right-skewed - a handful of bulk orders sit two orders of magnitude
    above the median - and on the raw scale those few weeks dominate the loss,
    dragging every ordinary forecast upwards. Working in log space and
    inverting afterwards helped more than any amount of tuning; the
    development-period comparison that settled it is in the README.
    """

    def __init__(self, log_target: bool = True, **params):
        self.log_target = log_target
        self.params = {
            "objective": "regression",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_data_in_leaf": 30,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "n_estimators": 200,
            "lambda_l2": 1.0,
            "verbosity": -1,
            **params,
        }
        self.model: lgb.LGBMRegressor | None = None
        self._sku_dtype: pd.CategoricalDtype | None = None

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[FEATURE_COLUMNS].copy()
        X["sku"] = pd.Categorical(df["StockCode"], dtype=self._sku_dtype)
        return X

    def fit(self, training_frame: pd.DataFrame) -> "LightGBMForecaster":
        self._sku_dtype = pd.CategoricalDtype(sorted(training_frame["StockCode"].unique()))
        y = training_frame["units"]
        if self.log_target:
            y = np.log1p(y)
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(self._design(training_frame), y, categorical_feature=["sku"])
        return self

    def predict(self, panel, origin_pos: int, horizon: int, future_dates) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("fit() must be called before predict()")
        frame = build_prediction_frame(panel, origin_pos, horizon, future_dates)
        pred = self.model.predict(self._design(frame))
        if self.log_target:
            pred = np.expm1(pred)
        frame["prediction"] = np.clip(pred, 0, None)
        return frame[["StockCode", "date", "horizon_step", "prediction"]]

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("fit() must be called before feature_importance()")
        return (
            pd.DataFrame(
                {
                    "feature": self.model.feature_name_,
                    "importance": self.model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


BASELINES = {
    "Naive (last week)": naive,
    "Seasonal naive (52w)": seasonal_naive,
    "Moving average (4w)": moving_average_4,
    "Moving average (13w)": moving_average_13,
}
