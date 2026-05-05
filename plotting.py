"""Headless plots: demand forecasts, the inventory simulation, and the LP allocation."""
from __future__ import annotations

import logging
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

from config import Config

LOGGER = logging.getLogger("supply.plot")


def plot_forecasts(panel: pd.DataFrame, fc: Dict[str, Dict], cfg: Config,
                   path: str) -> None:
    """Actual history + hold-out actual vs Holt-Winters / seasonal-naive forecast."""
    skus = list(fc.keys())[:4]
    fig, axes = plt.subplots(2, 2, figsize=(16, 8), dpi=cfg.dpi)
    h = cfg.holdout_days
    for ax, sku in zip(axes.ravel(), skus):
        series = panel[sku]
        train, test = series.iloc[:-h], series.iloc[-h:]
        ax.plot(train.index[-90:], train.iloc[-90:], color="#444", lw=1.0, label="History")
        ax.plot(test.index, test.values, color="#111", lw=1.6, label="Actual (hold-out)")
        ax.plot(test.index, fc[sku]["hw"], color="#d62728", lw=1.5, ls="--",
                label="Holt-Winters")
        ax.plot(test.index, fc[sku]["naive"], color="#1f77b4", lw=1.0, ls=":",
                label="Seasonal-naive")
        ax.set_title(f"SKU {sku}  (HW WAPE {fc[sku]['wape']:.2f})", fontsize=10)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", labelsize=7)
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Demand Forecast vs Actual (hold-out)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved forecast plot -> %s", path)


def plot_inventory(dates, demand, sim: Dict, reorder_r: float, sku: str,
                   frontier: pd.DataFrame, cfg: Config, path: str) -> None:
    """(Q,R) inventory trajectory + the cost vs service-level frontier."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), dpi=cfg.dpi)

    ax = axes[0]
    ax.plot(dates, sim["on_hand"], color="#2ca02c", lw=1.4, label="On-hand inventory")
    ax.axhline(reorder_r, color="#d62728", ls="--", lw=1.0, label="Reorder point R")
    stockouts = np.where((sim["on_hand"] <= 0) & (np.asarray(demand) > 0))[0]
    if len(stockouts):
        ax.scatter(np.asarray(dates)[stockouts], np.zeros(len(stockouts)),
                   color="black", s=14, zorder=5, label="Stockout day")
    ax.set_title(f"(Q,R) inventory simulation - SKU {sku}\n"
                 f"fill rate {sim['fill_rate']*100:.1f}%, cost ${sim['total_cost']:.0f}",
                 fontsize=10)
    ax.set_xlabel("Date"); ax.set_ylabel("Units")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25)
    ax.tick_params(axis="x", labelsize=7)

    ax = axes[1]
    st = frontier["service_target"] * 100
    ax.plot(st, frontier["total_cost"], "o-", color="#9467bd", lw=1.5, label="Total cost")
    ax.set_xlabel("Target service level (%)")
    ax.set_ylabel("Total cost ($)", color="#9467bd")
    ax.tick_params(axis="y", labelcolor="#9467bd")
    ax2 = ax.twinx()
    ax2.plot(st, frontier["fill_rate"] * 100, "s--", color="#2ca02c", lw=1.3,
             label="Achieved fill rate")
    ax2.set_ylabel("Achieved fill rate (%)", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax.set_title("Cost of service: higher target -> more safety stock", fontsize=10)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved inventory plot -> %s", path)


def plot_allocation(skus: List[str], alloc: Dict, baseline: Dict,
                    demand_caps: np.ndarray, cfg: Config, path: str) -> None:
    """LP allocation vs forecast demand and the naive proportional baseline."""
    x = np.arange(len(skus))
    fig, ax = plt.subplots(figsize=(13, 6), dpi=cfg.dpi)
    ax.bar(x - 0.27, demand_caps, width=0.27, color="#cccccc", label="Forecast demand")
    ax.bar(x, alloc["quantities"], width=0.27, color="#2ca02c",
           label=f"LP allocation (profit ${alloc['profit']:.0f})")
    ax.bar(x + 0.27, baseline["quantities"], width=0.27, color="#1f77b4",
           label=f"Proportional (profit ${baseline['profit']:.0f})")
    ax.set_xticks(x)
    ax.set_xticklabels(skus, rotation=45, ha="right", fontsize=8)
    ax.set_title("Procurement allocation under budget + capacity limits",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Units")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved allocation plot -> %s", path)
