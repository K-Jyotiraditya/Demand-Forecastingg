"""Offline unit tests for forecasting, inventory, and allocation (synthetic inputs)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from allocation import allocate
from config import Config
from data import clean_transactions
from forecasting import (fit_holt_winters, forecast_metrics, holt_winters_additive,
                         seasonal_naive)
from inventory import (eoq, newsvendor_quantity, reorder_point, safety_stock,
                       simulate_qr)


# --------------------------------------------------------------------------- #
# Forecasting
# --------------------------------------------------------------------------- #
def _seasonal_series(n=140, m=7, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 20 + 0.15 * t + 4 * np.sin(2 * np.pi * t / m) + rng.normal(0, 1, n)


def test_holt_winters_tracks_signal():
    y = _seasonal_series()
    res = fit_holt_winters(y, m=7, horizon=14)
    assert all(0.0 <= res[k] <= 1.0 for k in ("alpha", "beta", "gamma"))
    assert res["forecast"].shape == (14,)
    assert np.all(res["forecast"] >= 0.0)
    assert np.corrcoef(res["fitted"], y)[0, 1] > 0.8     # fit follows the series


def test_seasonal_naive_repeats_last_season():
    y = np.arange(14.0)                                   # two seasons of m=7
    fc = seasonal_naive(y, m=7, horizon=7)
    assert np.array_equal(fc, np.arange(7.0, 14.0))


def test_forecast_metrics_zero_on_perfect():
    a = np.array([3.0, 1.0, 4.0, 1.0])
    m = forecast_metrics(a, a)
    assert m["mae"] == pytest.approx(0.0) and m["wape"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
def test_eoq_closed_form():
    assert eoq(1000, 10, 2) == pytest.approx(100.0)      # sqrt(2*1000*10/2)


def test_safety_stock_and_reorder_point():
    ss = safety_stock(10.0, 4, 0.95)
    assert ss == pytest.approx(1.6448536 * 10 * 2, rel=1e-4)
    assert reorder_point(5.0, 4, ss) == pytest.approx(20.0 + ss)


def test_newsvendor_symmetric_costs_gives_mean():
    assert newsvendor_quantity(50.0, 8.0, underage=1.0, overage=1.0) == pytest.approx(50.0)


def test_qr_simulation_bounds_fill_rate():
    demand = np.full(30, 5.0)
    cfg = Config()
    starved = simulate_qr(demand, order_q=0.0, reorder_r=0.0, cfg=cfg, unit_cost=10.0)
    assert starved["fill_rate"] == pytest.approx(0.0, abs=1e-6)
    stocked = simulate_qr(demand, order_q=200.0, reorder_r=100.0, cfg=cfg, unit_cost=10.0)
    assert stocked["fill_rate"] > 0.99


# --------------------------------------------------------------------------- #
# Allocation LP
# --------------------------------------------------------------------------- #
def test_allocation_prefers_high_margin_under_budget():
    # margins 3 vs 1, equal cost, budget caps total spend at 10 units.
    res = allocate(margins=np.array([3.0, 1.0]), unit_costs=np.array([1.0, 1.0]),
                   demand_caps=np.array([10.0, 10.0]), budget=10.0, capacity=100.0)
    assert res["quantities"][0] == pytest.approx(10.0, abs=1e-6)
    assert res["quantities"][1] == pytest.approx(0.0, abs=1e-6)
    assert res["profit"] == pytest.approx(30.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Data cleaning
# --------------------------------------------------------------------------- #
def test_clean_drops_cancellations_and_bad_rows():
    df = pd.DataFrame({
        "InvoiceNo": ["536365", "C536366", "536367", "536368"],
        "StockCode": ["A", "A", "B", None],
        "Quantity": [6, 3, -2, 5],
        "UnitPrice": [2.5, 2.5, 2.5, 0.0],
        "InvoiceDate": pd.to_datetime(["2010-12-01"] * 4),
    })
    cleaned = clean_transactions(df)
    assert len(cleaned) == 1                       # only the first row survives
    assert cleaned.iloc[0]["InvoiceNo"] == "536365"
    assert "date" in cleaned.columns
