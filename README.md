# 📦 Retail Demand Forecasting

Weekly SKU-level demand forecasting on two years of real e-commerce
transactions, benchmarked against the baselines that actually matter.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-gradient%20boosting-green)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit)

---

## Results

Rolling-origin backtest: 8 forecast origins, 4 weeks apart, each forecasting
4 weeks ahead, 50 SKUs. Lower is better throughout.

| Model | MAE | RMSE | wMAPE | MASE |
|---|---|---|---|---|
| **LightGBM** | **219.5** | 456.2 | **55.8%** | **0.783** |
| Moving average (13w) | 230.6 | 449.4 | 58.6% | 0.829 |
| Moving average (4w) | 233.1 | 444.4 | 59.2% | 0.809 |
| Naive (last week) | 290.9 | 530.5 | 73.9% | 0.988 |
| Seasonal naive (52w) | 362.1 | 772.0 | 92.0% | 1.237 |

The model beats the best baseline by **4.8% on wMAPE**. That is a modest
margin, and it is the honest one: when demand is mostly a slowly-drifting level
plus bulk-order noise, a 13-week moving average is a genuinely strong opponent.
Any portfolio project claiming to crush baselines at demand forecasting has
usually leaked the future into its features.

RMSE is the one metric where the moving averages win, and that is a real
trade-off rather than a rounding error: modelling the target in log space
deliberately stops the model chasing the handful of enormous bulk orders, which
costs squared error and buys accuracy on the other 99% of weeks.

---

## The data

[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
(UCI) - 1,067,371 transaction lines from a UK online retailer, Dec 2009 to
Dec 2011.

Cleaning removes cancellations (invoices prefixed `C`), non-positive quantities
and non-product codes (postage, bank charges, manual adjustments). The
remaining lines are aggregated to weekly units per SKU, keeping the 50
highest-volume SKUs: **5,200 SKU-weeks over 104 complete weeks**.

Two decisions did more for the results than any modelling:

**Weeks with no invoice are explicit zeros, not gaps.** A SKU that sold nothing
sold zero. Leaving those weeks missing would quietly bias every rolling average
upwards. 12% of SKU-weeks are zeros.

**Daily was the wrong grain.** This is a wholesale business - one customer
ordering 80,000 units on a Tuesday makes daily SKU demand close to
unforecastable noise, and 0.4% of days carry 21% of all volume. No model can
time an individual bulk order from sales history alone. Weekly buckets absorb
that timing jitter and match how replenishment is actually planned.

---

## Method

### Features

Every feature is read **at the forecast origin**, never at the target week.
Calendar facts about the target week are the one exception, because they
genuinely are known in advance - we know today that Christmas falls in week 52.

| Group | Features |
|---|---|
| **Lags** | units at t-1, t-2, t-3, t-4, t-8, t-13, t-52 |
| **Rolling** | mean, std and zero-share over 4 / 8 / 13 / 26 weeks |
| **Trend** | 4-week mean over 13-week mean; same period last year |
| **Market signals** | price, invoice count, customer count - last value, 4-week mean, and ratio to the 13-week norm |
| **Calendar** | month, quarter, week of year and of month, UK holidays in the week, weeks to Christmas, cyclical sin/cos encodings |
| **Horizon** | the step being forecast (1-4), so one model can widen its own uncertainty further out |

The price-to-13-week-norm ratio is the closest thing this dataset has to a
promotion flag: a SKU trading below its own recent norm is usually being pushed.

### Model

A single **global LightGBM** across all SKUs, with SKU identity as a
categorical feature. Global beats per-SKU here because each series is only ~100
weeks long - pooling lets a quiet SKU borrow the seasonal shape learned from
the busy ones.

The target is modelled as `log1p(units)`. Weekly demand is non-negative and
heavily right-skewed, and on the raw scale a few bulk-order weeks dominate the
loss and drag every ordinary forecast upwards.

### Validation

**Rolling-origin backtesting.** A single train/test split on a time series
tells you about one month's weather. This walks 8 origins forward through the
calendar, retraining from scratch at each one on data up to that origin only.

**The target transform was chosen on a separate development period**
(Nov 2010 - Mar 2011) so the reported test period stayed untouched. On that
development period `log1p` scored 0.768 wMAPE against 0.803 for Tweedie, which
is what settled it. Tuning on the test folds would have been the subtler cousin
of training on the test set.

### Metrics

- **wMAPE** - total absolute error over total demand. What demand planners
  actually use; plain MAPE explodes on the zero-demand weeks, and 12% of these
  weeks are zero.
- **MASE** - error scaled by the in-sample naive forecast. Below 1 beats that
  naive rule, and it is comparable across SKUs of very different volume.
- **MAE / RMSE** - absolute units, for reference.

---

## Running it

```bash
pip install -r requirements.txt

# Rebuild the weekly panel from the raw extract (see data/README.md)
python data/prepare_data.py

# Reproduce the backtest table above
python run_backtest.py

# Explore it
streamlit run app.py
```

## Project structure

```
demand-forecasting/
├── app.py                  # Streamlit dashboard
├── run_backtest.py         # Reproduces the results table
├── data/
│   ├── prepare_data.py     # Raw transactions -> weekly panel
│   └── weekly_demand.csv   # 50 SKUs x 104 weeks
├── utils/
│   ├── features.py         # Leak-safe feature construction
│   ├── models.py           # Baselines + global LightGBM
│   └── backtest.py         # Rolling-origin evaluation, metrics
└── results/                # Cached backtest output
```

## What I would do next

- **Hierarchical reconciliation.** Total demand is far more forecastable than
  any single SKU; forecasting the top level and reconciling downwards usually
  beats bottom-up on noisy series.
- **Prediction intervals** rather than point forecasts. Replenishment decisions
  need a quantile, not a mean - LightGBM's quantile objective would give
  service-level-aware stock recommendations.
- **Intermittent-demand models** (Croston, TSB) for the long tail of SKUs that
  sell in only a handful of weeks.

---

## Author

**Marwan El Saabi** - MSc Bioinformatics candidate
[Portfolio](https://marwansaabi.github.io) · [GitHub](https://github.com/marwansaabi) · [LinkedIn](https://www.linkedin.com/in/marwansaabi/)
