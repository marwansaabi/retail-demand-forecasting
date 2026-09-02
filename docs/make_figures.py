"""Figures for the explainer PDF."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).parent / "assets"
OUT.mkdir(parents=True, exist_ok=True)

ACCENT = "#2563eb"     # winner
MUTED = "#b8c2cf"      # baselines
INK = "#1e293b"
GRID = "#e8ecf1"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": "#64748b",
    "ytick.color": INK,
    "axes.edgecolor": GRID,
})


def fig_results():
    """Single-series magnitude comparison: wMAPE per model, best on top."""
    s = pd.read_csv(Path(__file__).parent.parent / "results/backtest_summary.csv")
    s = s.sort_values("wMAPE", ascending=False)          # bottom-up = worst-first
    colors = [ACCENT if m == "LightGBM" else MUTED for m in s["model"]]

    fig, ax = plt.subplots(figsize=(7.4, 3.2), dpi=200)
    bars = ax.barh(s["model"], s["wMAPE"] * 100, color=colors, height=0.62)
    # Rounded data-ends, value labelled directly - no legend needed for one series.
    for b in bars:
        b.set_joinstyle("round")
    for bar, v in zip(bars, s["wMAPE"] * 100):
        ax.text(v + 1.2, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%",
                va="center", ha="left", fontsize=9, color=INK)

    ax.set_xlabel("wMAPE (%)  ·  menor es mejor", fontsize=9)
    ax.set_xlim(0, 105)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "results.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_rolling_origin():
    """Schematic of the validation scheme - a diagram, not a data chart."""
    fig, ax = plt.subplots(figsize=(7.4, 3.4), dpi=200)
    n_folds, horizon, stride = 8, 4, 4
    total = 104

    for i in range(n_folds):
        y = n_folds - i
        origin = total - horizon - (n_folds - 1 - i) * stride
        ax.barh(y, origin, color=MUTED, height=0.55)
        ax.barh(y, horizon, left=origin, color=ACCENT, height=0.55)
        ax.text(-2, y, f"Fold {i+1}", va="center", ha="right", fontsize=8.5, color=INK)

    ax.text(45, n_folds + 1.15, "Entrenamiento (todo lo anterior al origen)",
            fontsize=9, color="#64748b", ha="center")
    ax.text(99, n_folds + 1.15, "Test\n(4 sem.)", fontsize=9, color=ACCENT,
            ha="center", va="bottom")

    ax.set_xlim(-14, 108)
    ax.set_ylim(0.2, n_folds + 2.1)
    ax.set_xlabel("semanas desde el inicio de los datos", fontsize=9)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "rolling_origin.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_daily_vs_weekly():
    """Why weekly: the same SKU at both grains."""
    daily = pd.read_csv(Path(__file__).parent.parent / "data/daily_demand.csv",
                        parse_dates=["date"])
    weekly = pd.read_csv(Path(__file__).parent.parent / "data/weekly_demand.csv",
                         parse_dates=["date"])
    sku = weekly.groupby("StockCode")["units"].sum().idxmax()
    d = daily[daily["StockCode"] == sku].sort_values("date")
    w = weekly[weekly["StockCode"] == sku].sort_values("date")

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 3.6), dpi=200, sharex=True)
    axes[0].plot(d["date"], d["units"], color=MUTED, linewidth=0.9)
    axes[0].set_title("Demanda diaria: picos de pedidos individuales",
                      fontsize=9.5, loc="left", color=INK)
    axes[1].plot(w["date"], w["units"], color=ACCENT, linewidth=1.6)
    axes[1].set_title("Misma referencia, agregada a semanal",
                      fontsize=9.5, loc="left", color=INK)
    for ax in axes:
        ax.set_ylabel("unidades", fontsize=8.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "daily_vs_weekly.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_results()
    fig_rolling_origin()
    fig_daily_vs_weekly()
    print("figuras generadas:", *[p.name for p in sorted(OUT.glob("*.png"))])
