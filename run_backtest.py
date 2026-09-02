"""Run the full rolling-origin backtest and cache results for the app."""
import time
from pathlib import Path

import pandas as pd

from utils.backtest import rolling_origin_backtest

OUT = Path("results")
OUT.mkdir(exist_ok=True)


def main():
    panel = pd.read_csv("data/weekly_demand.csv", parse_dates=["date"])
    print(f"{panel['StockCode'].nunique()} SKUs, {len(panel):,} rows, "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}\n")

    t0 = time.time()
    results, summary, model = rolling_origin_backtest(
        panel, horizon=4, n_folds=8, stride=4,
        progress=lambda i, n, o: print(f"  fold {i}/{n}  origin={o.date()}"),
    )
    print(f"\ndone in {time.time() - t0:.1f}s\n")
    print(summary.to_string(index=False))

    results.to_csv(OUT / "backtest_results.csv", index=False)
    summary.to_csv(OUT / "backtest_summary.csv", index=False)
    model.feature_importance().to_csv(OUT / "feature_importance.csv", index=False)
    print(f"\nwrote {OUT}/")


if __name__ == "__main__":
    main()
