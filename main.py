"""Driver: real retail data -> forecast -> inventory policy -> allocation LP -> plots.

    python main.py            # downloads + caches Online Retail on first call
    python -m pytest -q       # offline unit tests
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from allocation import allocate, proportional_baseline
from config import Config
from data import build_demand_panel, clean_transactions, load_retail
from forecasting import (fit_holt_winters, forecast_metrics, seasonal_naive)
from inventory import (eoq, reorder_point, safety_stock, service_cost_frontier,
                       simulate_qr)
from plotting import plot_allocation, plot_forecasts, plot_inventory

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
LOGGER = logging.getLogger("supply")
HERE = Path(__file__).resolve().parent


def run(cfg: Config) -> dict:
    panel, prices = build_demand_panel(clean_transactions(load_retail(cfg)), cfg)
    skus = list(panel.columns)
    h, m = cfg.holdout_days, cfg.season_period

    # --- Forecasting: Holt-Winters vs seasonal-naive on the hold-out ---
    fc, hw_metrics, naive_metrics = {}, [], []
    for sku in skus:
        series = panel[sku].to_numpy()
        train, test = series[:-h], series[-h:]
        hw = fit_holt_winters(train, m, h)
        naive = seasonal_naive(train, m, h)
        mh, mn = forecast_metrics(test, hw["forecast"]), forecast_metrics(test, naive)
        fc[sku] = {"hw": hw["forecast"], "naive": naive,
                   "wape": mh["wape"], "rmse": mh["rmse"]}
        hw_metrics.append(mh); naive_metrics.append(mn)

    def _avg(ms, k):
        return float(np.mean([m[k] for m in ms]))
    LOGGER.info("Forecast (avg over %d SKUs) | HW WAPE %.3f vs naive %.3f | "
                "HW MAE %.2f vs naive %.2f", len(skus),
                _avg(hw_metrics, "wape"), _avg(naive_metrics, "wape"),
                _avg(hw_metrics, "mae"), _avg(naive_metrics, "mae"))

    # --- Per-SKU economics: heterogeneous margins (real COGS would replace this) ---
    rng = np.random.default_rng(cfg.seed)
    margin_rate = rng.uniform(cfg.margin_low, cfg.margin_high, len(skus))
    price = np.array([float(prices[s]) for s in skus])
    unit_costs = price * (1 - margin_rate)
    margins = price * margin_rate

    # --- Forecast-driven inventory policy + (Q,R) simulation over the hold-out ---
    # Safety stock buffers FORECAST ERROR (not raw history), which is the proper
    # way to connect the forecaster to the inventory policy.
    print(f"\n{'SKU':<10}{'muFC/day':>9}{'EOQ':>8}{'ROP':>8}{'FillRate':>10}{'Cost$':>10}")
    print("-" * 55)
    inv_results = {}
    for i, sku in enumerate(skus):
        test = panel[sku].to_numpy()[-h:]
        mu_fc = float(fc[sku]["hw"].mean())
        sigma_err = float(fc[sku]["rmse"])
        uc = float(unit_costs[i])
        # EOQ uses the long-run demand rate (from history); safety stock buffers
        # the near-term forecast error. Splitting the two avoids degenerate lot
        # sizes when the forecast level is ~0 (intermittent demand).
        hist_mean = float(panel[sku].to_numpy()[:-h].mean())
        q = eoq(max(hist_mean, 1e-6) * cfg.annual_days, cfg.order_cost,
                uc * cfg.holding_cost_rate)
        ss = safety_stock(sigma_err, cfg.lead_time, cfg.service_level)
        r = reorder_point(mu_fc, cfg.lead_time, ss)
        sim = simulate_qr(test, q, r, cfg, uc)
        inv_results[sku] = {"q": q, "r": r, "sim": sim, "demand": test,
                            "unit_cost": uc, "mu": mu_fc, "sigma": sigma_err}
        print(f"{sku:<10}{mu_fc:>9.1f}{q:>8.0f}{r:>8.0f}"
              f"{sim['fill_rate']*100:>9.1f}%{sim['total_cost']:>10.0f}")

    # --- Allocation LP for the next cycle ---
    horizon_demand = np.array([fc[s]["hw"].sum() for s in skus])
    full_budget = float(unit_costs @ horizon_demand)
    budget = cfg.budget_frac * full_budget
    capacity = cfg.capacity_frac * float(horizon_demand.sum())
    alloc = allocate(margins, unit_costs, horizon_demand, budget, capacity)
    base = proportional_baseline(horizon_demand, unit_costs, margins, budget, capacity)
    LOGGER.info("Allocation | budget $%.0f cap %.0f u | LP profit $%.0f vs "
                "proportional $%.0f (+%.1f%%)", budget, capacity, alloc["profit"],
                base["profit"], 100 * (alloc["profit"] / base["profit"] - 1))

    # --- Plots ---
    plot_forecasts(panel, fc, cfg, str(HERE / "demand_forecast.png"))
    rep = max(skus, key=lambda s: fc[s]["wape"])            # hardest SKU to forecast
    rinfo = inv_results[rep]
    frontier = service_cost_frontier(
        rinfo["demand"], rinfo["mu"], rinfo["sigma"], cfg, rinfo["unit_cost"],
        np.array([0.50, 0.75, 0.90, 0.95, 0.99]))
    plot_inventory(panel.index[-h:], rinfo["demand"], rinfo["sim"], rinfo["r"],
                   rep, frontier, cfg, str(HERE / "inventory_policy.png"))
    plot_allocation(skus, alloc, base, horizon_demand, cfg,
                    str(HERE / "allocation_lp.png"))
    return {"fc": fc, "inventory": inv_results, "allocation": alloc}


def main() -> int:
    try:
        run(Config())
    except RuntimeError as exc:
        LOGGER.error("Aborted: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
