"""
Feature construction for multi-horizon demand forecasting.

Two rules run through this file.

1. A feature for target period `t+k`, forecast from origin `t`, may only use
   information available at `t`. Calendar facts about the target are fair game
   (we know today that Christmas week is week 52); its sales are not. Lag
   features are therefore always read at the origin, never at the target.

2. Everything is indexed by *period position*, not by a number of days. The
   panel is weekly here, but nothing below assumes that.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

# Measured in periods (weeks) backwards from the forecast origin.
LAGS = (1, 2, 3, 4, 8, 13, 52)
ROLLING_WINDOWS = (4, 8, 13, 26)

_UK_HOLIDAYS = holidays.UnitedKingdom(years=range(2009, 2013))
_HOLIDAY_DATES = pd.to_datetime(sorted(_UK_HOLIDAYS.keys()))


def calendar_features(dates: pd.Series) -> pd.DataFrame:
    """Facts about the target week that are knowable arbitrarily far ahead."""
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    out = pd.DataFrame(index=d.index)
    out["month"] = d.dt.month
    out["quarter"] = d.dt.quarter
    out["week_of_year"] = d.dt.isocalendar().week.astype(int)
    out["week_of_month"] = ((d.dt.day - 1) // 7) + 1

    # A week is stamped with its end date; a holiday anywhere inside it counts.
    week_start = d - pd.Timedelta(days=6)
    out["n_holidays"] = [
        int(((_HOLIDAY_DATES >= s) & (_HOLIDAY_DATES <= e)).sum())
        for s, e in zip(week_start, d)
    ]

    # Christmas dominates this retailer's year. Stocking happens weeks earlier,
    # so distance to it is a stronger signal than the month alone.
    christmas = pd.to_datetime(dict(year=d.dt.year, month=12, day=25))
    weeks_to_christmas = ((christmas - d).dt.days / 7).round()
    next_christmas = pd.to_datetime(dict(year=d.dt.year + 1, month=12, day=25))
    weeks_to_next = ((next_christmas - d).dt.days / 7).round()
    out["weeks_to_christmas"] = np.where(
        weeks_to_christmas < 0, weeks_to_next, weeks_to_christmas
    )

    # Cyclical encodings so the model sees week 52 and week 1 as adjacent.
    out["woy_sin"] = np.sin(2 * np.pi * out["week_of_year"] / 52)
    out["woy_cos"] = np.cos(2 * np.pi * out["week_of_year"] / 52)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


EXOG_COLUMNS = ("avg_price", "n_invoices", "n_customers")


def history_features(history: np.ndarray, exog: dict | None = None) -> dict[str, float]:
    """
    Summarise one SKU's history up to and including the forecast origin.

    `history` is the demand series; `exog` maps each market signal (price,
    invoice count, customer count) to its own series. All must already be
    truncated at the origin - this function cannot tell whether it was handed
    future data.
    """
    v = np.asarray(history, dtype=float)
    feats: dict[str, float] = {}

    for lag in LAGS:
        feats[f"lag_{lag}"] = float(v[-lag]) if len(v) >= lag else np.nan

    for w in ROLLING_WINDOWS:
        recent = v[-w:]
        if len(recent) == 0:
            feats[f"roll_mean_{w}"] = np.nan
            feats[f"roll_std_{w}"] = np.nan
            feats[f"roll_zero_share_{w}"] = np.nan
            continue
        feats[f"roll_mean_{w}"] = float(recent.mean())
        feats[f"roll_std_{w}"] = float(recent.std())
        # Whether a SKU sells at all in a week is its own signal.
        feats[f"roll_zero_share_{w}"] = float((recent == 0).mean())

    # Trend: is the recent month above or below the recent quarter?
    m4 = feats.get("roll_mean_4") or 0.0
    m13 = feats.get("roll_mean_13") or 0.0
    feats["trend_4_over_13"] = float(m4 / m13) if m13 > 0 else np.nan

    # Last year, same period - the yearly seasonal anchor.
    if len(v) >= 52:
        feats["last_year_mean_3"] = float(v[-54:-51].mean()) if len(v) >= 54 else float(v[-52])
    else:
        feats["last_year_mean_3"] = np.nan

    nonzero = v[v > 0]
    feats["mean_nonzero_all"] = float(nonzero.mean()) if len(nonzero) else 0.0

    # Market signals. Price relative to its own recent average is the closest
    # thing this dataset has to a promotion flag: a SKU trading below its
    # 13-week norm is usually being pushed.
    for name in EXOG_COLUMNS:
        if exog is None or name not in exog:
            feats[f"{name}_last"] = np.nan
            feats[f"{name}_mean_4"] = np.nan
            feats[f"{name}_ratio_13"] = np.nan
            continue
        e = np.asarray(exog[name], dtype=float)
        last = float(e[-1]) if len(e) else np.nan
        mean_4 = float(e[-4:].mean()) if len(e) else np.nan
        mean_13 = float(e[-13:].mean()) if len(e) else np.nan
        feats[f"{name}_last"] = last
        feats[f"{name}_mean_4"] = mean_4
        feats[f"{name}_ratio_13"] = (
            float(last / mean_13) if mean_13 and mean_13 > 0 else np.nan
        )
    return feats


HISTORY_COLUMNS = (
    [f"lag_{l}" for l in LAGS]
    + [f"roll_mean_{w}" for w in ROLLING_WINDOWS]
    + [f"roll_std_{w}" for w in ROLLING_WINDOWS]
    + [f"roll_zero_share_{w}" for w in ROLLING_WINDOWS]
    + ["trend_4_over_13", "last_year_mean_3", "mean_nonzero_all"]
    + [f"{n}_last" for n in EXOG_COLUMNS]
    + [f"{n}_mean_4" for n in EXOG_COLUMNS]
    + [f"{n}_ratio_13" for n in EXOG_COLUMNS]
)

CALENDAR_COLUMNS = [
    "month", "quarter", "week_of_year", "week_of_month", "n_holidays",
    "weeks_to_christmas", "woy_sin", "woy_cos", "month_sin", "month_cos",
]

FEATURE_COLUMNS = HISTORY_COLUMNS + CALENDAR_COLUMNS + ["horizon_step"]


def as_series_map(panel: pd.DataFrame) -> dict[str, pd.Series]:
    """One date-indexed demand series per SKU, aligned on a common calendar."""
    return {
        sku: g.sort_values("date").set_index("date")["units"]
        for sku, g in panel.groupby("StockCode", sort=False)
    }


def as_exog_map(panel: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """Per-SKU arrays of the market signals, aligned with the demand series."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for sku, g in panel.groupby("StockCode", sort=False):
        g = g.sort_values("date")
        out[sku] = {
            c: g[c].to_numpy(dtype=float) for c in EXOG_COLUMNS if c in g.columns
        }
    return out


def period_index(panel: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(np.sort(panel["date"].unique()))


def build_training_frame(
    panel: pd.DataFrame,
    horizon: int,
    min_history: int = 26,
    stride: int = 1,
) -> pd.DataFrame:
    """
    Expand the (SKU, period) panel into supervised rows.

    Each row is one (origin, horizon step) pair: features known at the origin,
    target = units sold `step` periods later.
    """
    dates = period_index(panel)
    series_map = as_series_map(panel)
    rows = []

    exog_map = as_exog_map(panel)

    for sku, s in series_map.items():
        v = s.to_numpy(dtype=float)
        ex = exog_map.get(sku, {})
        for pos in range(min_history, len(v) - horizon, stride):
            hist_feats = history_features(
                v[: pos + 1], {k: a[: pos + 1] for k, a in ex.items()}
            )
            for step in range(1, horizon + 1):
                rows.append(
                    {
                        "StockCode": sku,
                        "origin": dates[pos],
                        "date": dates[pos + step],
                        "horizon_step": step,
                        "units": float(v[pos + step]),
                        **hist_feats,
                    }
                )

    df = pd.DataFrame(rows)
    return pd.concat([df, calendar_features(df["date"])], axis=1)


def build_prediction_frame(
    panel: pd.DataFrame, origin_pos: int, horizon: int, future_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Features for the `horizon` periods after `origin_pos`, for every SKU."""
    dates = period_index(panel)
    series_map = as_series_map(panel)
    rows = []

    exog_map = as_exog_map(panel)

    for sku, s in series_map.items():
        v = s.to_numpy(dtype=float)[: origin_pos + 1]
        if len(v) == 0:
            continue
        ex = exog_map.get(sku, {})
        hist_feats = history_features(
            v, {k: a[: origin_pos + 1] for k, a in ex.items()}
        )
        for step in range(1, horizon + 1):
            rows.append(
                {
                    "StockCode": sku,
                    "origin": dates[origin_pos],
                    "date": future_dates[step - 1],
                    "horizon_step": step,
                    **hist_feats,
                }
            )

    df = pd.DataFrame(rows)
    return pd.concat([df, calendar_features(df["date"])], axis=1)
