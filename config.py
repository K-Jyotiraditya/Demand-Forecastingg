"""Run configuration for the supply-chain forecasting + optimization engine."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    """Hyper-parameters and I/O knobs for one supply-chain run."""

    # --- Data (real: UCI Online Retail transactional data) -----------------
    data_url: str = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
                     "00352/Online%20Retail.xlsx")
    cache_path: str = "online_retail.pkl"
    top_n_skus: int = 8
    holdout_days: int = 28          # final stretch held out to score forecasts

    # --- Forecasting (Holt-Winters additive) -------------------------------
    season_period: int = 7          # weekly seasonality in daily demand

    # --- Inventory policy (Q, R) -------------------------------------------
    lead_time: int = 7              # days from order to receipt
    order_cost: float = 50.0        # fixed cost per replenishment order (S)
    holding_cost_rate: float = 0.25  # annual holding cost as a fraction of unit cost
    stockout_penalty: float = 3.0   # penalty per unit of unmet demand
    service_level: float = 0.95     # target cycle service level
    annual_days: int = 365

    # --- Margins + allocation LP -------------------------------------------
    # Real products carry different margins; we draw a per-SKU gross margin so
    # profit-per-dollar varies and the LP has a genuine trade-off to solve.
    margin_low: float = 0.20
    margin_high: float = 0.55
    budget_frac: float = 0.60       # procurement budget as a fraction of full need
    capacity_frac: float = 0.70     # warehouse capacity as a fraction of full need

    seed: int = 42
    dpi: int = 150
