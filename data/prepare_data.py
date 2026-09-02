"""
Turn the raw Online Retail II transaction log into a clean daily demand series.

The raw file is one row per invoice line. Forecasting needs one row per
(SKU, day) with the units sold that day - including the days where nothing
was sold, which the transaction log simply doesn't mention.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
RAW_XLSX = DATA_DIR / "online_retail_II.xlsx"
DAILY_CSV = DATA_DIR / "daily_demand.csv"
WEEKLY_CSV = DATA_DIR / "weekly_demand.csv"

# Keep the busiest SKUs: enough history to learn from, small enough to ship.
TOP_N_SKUS = 50


def load_raw() -> pd.DataFrame:
    sheets = ["Year 2009-2010", "Year 2010-2011"]
    frames = [pd.read_excel(RAW_XLSX, sheet_name=s) for s in sheets]
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Invoice"] = df["Invoice"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()

    # Invoices prefixed with C are cancellations - they carry negative
    # quantities and would otherwise cancel out real demand.
    df = df[~df["Invoice"].str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]

    # Non-product codes: postage, bank charges, samples, manual adjustments.
    non_products = {"POST", "D", "DOT", "M", "S", "AMAZONFEE", "BANK CHARGES",
                    "C2", "CRUK", "PADS", "B", "GIFT"}
    df = df[~df["StockCode"].isin(non_products)]
    # Real product codes start with digits; the rest are admin entries.
    df = df[df["StockCode"].str[0].str.isdigit()]

    df["date"] = pd.to_datetime(df["InvoiceDate"]).dt.normalize()
    return df


def to_daily_demand(df: pd.DataFrame, top_n: int = TOP_N_SKUS) -> pd.DataFrame:
    totals = df.groupby("StockCode")["Quantity"].sum().nlargest(top_n)
    keep = set(totals.index)
    df = df[df["StockCode"].isin(keep)]

    daily = (
        df.groupby(["StockCode", "date"])
        .agg(
            units=("Quantity", "sum"),
            revenue=("Quantity", lambda s: 0.0),   # filled below
            avg_price=("Price", "mean"),
            n_invoices=("Invoice", "nunique"),
            n_customers=("Customer ID", "nunique"),
        )
        .reset_index()
    )
    line_revenue = df.assign(rev=df["Quantity"] * df["Price"])
    rev = line_revenue.groupby(["StockCode", "date"])["rev"].sum().rename("revenue")
    daily = daily.drop(columns=["revenue"]).merge(rev, on=["StockCode", "date"], how="left")

    # A SKU with no invoice on a given day sold zero units. Reindexing every
    # SKU onto the full calendar makes those zeros explicit; without this the
    # models would silently learn from a gap-ridden series.
    full_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    frames = []
    for sku, g in daily.groupby("StockCode"):
        g = (
            g.set_index("date")
            .reindex(full_range)
            .assign(StockCode=sku)
            .rename_axis("date")
            .reset_index()
        )
        g["units"] = g["units"].fillna(0.0)
        g["revenue"] = g["revenue"].fillna(0.0)
        g["n_invoices"] = g["n_invoices"].fillna(0.0)
        g["n_customers"] = g["n_customers"].fillna(0.0)
        # No sale means no observed price, not a price of zero: carry the last
        # known price forward, which is what the shelf price actually did.
        g["avg_price"] = g["avg_price"].ffill().bfill()
        frames.append(g)

    out = pd.concat(frames, ignore_index=True)

    descriptions = (
        df.sort_values("InvoiceDate")
        .groupby("StockCode")["Description"]
        .last()
        .str.strip()
    )
    out["description"] = out["StockCode"].map(descriptions)
    cols = ["date", "StockCode", "description", "units", "revenue",
            "avg_price", "n_invoices", "n_customers"]
    return out[cols].sort_values(["StockCode", "date"])


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to ISO weeks ending Sunday, keeping only complete weeks.

    Weekly is the right grain here. This is a wholesale business: a single
    customer ordering 80,000 units on one day makes daily SKU demand mostly
    noise, and no model can time an individual bulk order from history alone.
    Weekly buckets absorb that timing jitter and match how replenishment is
    actually planned.

    The first and last buckets cover only part of a week (the extract starts
    mid-week and ends mid-week), so they are dropped rather than left to look
    like a demand collapse.
    """
    weekly = (
        daily.set_index("date")
        .groupby("StockCode")
        .resample("W-SUN")
        .agg(
            units=("units", "sum"),
            revenue=("revenue", "sum"),
            avg_price=("avg_price", "mean"),
            n_invoices=("n_invoices", "sum"),
            n_customers=("n_customers", "sum"),
        )
        .reset_index()
    )

    covered = daily["date"]
    first_full = covered.min() + pd.Timedelta(days=6)
    last_full = covered.max()
    weekly = weekly[
        (weekly["date"] >= first_full) & (weekly["date"] <= last_full)
    ]
    # A week is complete only if its whole 7-day span sits inside the extract.
    weekly = weekly[weekly["date"] - pd.Timedelta(days=6) >= covered.min()]
    weekly = weekly[weekly["date"] <= covered.max()]

    descriptions = daily.groupby("StockCode")["description"].last()
    weekly["description"] = weekly["StockCode"].map(descriptions)
    cols = ["date", "StockCode", "description", "units", "revenue",
            "avg_price", "n_invoices", "n_customers"]
    return weekly[cols]


def main() -> None:
    raw = load_raw()
    print(f"raw rows                 {len(raw):>10,}")
    cleaned = clean(raw)
    print(f"after cleaning           {len(cleaned):>10,}")
    daily = to_daily_demand(cleaned)
    print(f"daily rows               {len(daily):>10,}")
    print(f"SKUs                     {daily['StockCode'].nunique():>10,}")
    print(f"date range               {daily['date'].min().date()} -> {daily['date'].max().date()}")
    print(f"zero-demand days         {(daily['units'] == 0).mean():>10.1%}")
    daily.to_csv(DAILY_CSV, index=False)

    weekly = to_weekly(daily)
    print(f"\nweekly rows              {len(weekly):>10,}")
    print(f"complete weeks           {weekly['date'].nunique():>10,}")
    print(f"week range               {weekly['date'].min().date()} -> {weekly['date'].max().date()}")
    print(f"zero-demand weeks        {(weekly['units'] == 0).mean():>10.1%}")
    weekly.to_csv(WEEKLY_CSV, index=False)
    print(f"\nwrote {DAILY_CSV.name} and {WEEKLY_CSV.name}")


if __name__ == "__main__":
    main()
