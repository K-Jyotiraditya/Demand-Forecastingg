"""Inventory policy: EOQ, safety stock / reorder point, newsvendor, and a (Q,R) sim.

The forecast answers "how much will sell?"; this module answers "how much to hold
and when to reorder?". Closed-form lot-sizing (EOQ) and a service-level safety
stock set the policy; a discrete-event (Q, R) simulation over the *actual* hold-out
demand then measures what that policy really costs and the fill rate it achieves.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import norm


def eoq(annual_demand: float, order_cost: float, holding_cost_per_unit: float) -> float:
    """Economic Order Quantity: sqrt(2 D S / H)."""
    return float(np.sqrt(2.0 * annual_demand * order_cost / max(holding_cost_per_unit, 1e-9)))


def safety_stock(sigma_daily: float, lead_time: int, service_level: float) -> float:
    """z * sigma * sqrt(L): buffer for demand variability over the lead time."""
    return float(norm.ppf(service_level) * sigma_daily * np.sqrt(lead_time))


def reorder_point(mu_daily: float, lead_time: int, ss: float) -> float:
    """Expected lead-time demand plus the safety buffer."""
    return float(mu_daily * lead_time + ss)


def newsvendor_quantity(mu: float, sigma: float, underage: float, overage: float) -> float:
    """Single-period optimum at the critical fractile Cu / (Cu + Co)."""
    fractile = underage / (underage + overage)
    return float(mu + norm.ppf(fractile) * sigma)


def simulate_qr(demand: np.ndarray, order_q: float, reorder_r: float, cfg,
                unit_cost: float) -> Dict:
    """Discrete-event (Q, R) policy simulation over a realised demand path.

    Each day: receive due orders, meet demand (unmet = lost sale), accrue holding
    cost on the closing on-hand, and place a fixed order ``order_q`` whenever the
    inventory position (on-hand + on-order) drops to ``reorder_r``.
    """
    demand = np.asarray(demand, dtype=float)
    n = len(demand)
    on_hand = float(reorder_r)            # start ready to trade
    pending: Dict[int, float] = {}        # arrival_day -> qty on order
    daily_holding = unit_cost * cfg.holding_cost_rate / cfg.annual_days

    on_hand_track = np.empty(n)
    holding = ordering = shortage_units = 0.0
    orders = 0

    for t in range(n):
        on_hand += pending.pop(t, 0.0)            # receive arrivals
        sold = min(on_hand, demand[t])
        shortage_units += demand[t] - sold        # lost sales
        on_hand -= sold
        holding += on_hand * daily_holding
        on_hand_track[t] = on_hand

        position = on_hand + sum(pending.values())
        if position <= reorder_r:
            pending[t + cfg.lead_time] = pending.get(t + cfg.lead_time, 0.0) + order_q
            ordering += cfg.order_cost
            orders += 1

    total_demand = demand.sum()
    stockout_cost = shortage_units * cfg.stockout_penalty
    return {
        "on_hand": on_hand_track,
        "holding_cost": holding,
        "ordering_cost": ordering,
        "stockout_cost": stockout_cost,
        "total_cost": holding + ordering + stockout_cost,
        "fill_rate": float(1.0 - shortage_units / (total_demand + 1e-9)),
        "orders": orders,
        "shortage_units": float(shortage_units),
    }


def service_cost_frontier(demand: np.ndarray, mu: float, sigma: float, cfg,
                          unit_cost: float, levels: np.ndarray) -> pd.DataFrame:
    """Sweep the target service level -> (achieved fill rate, realised cost)."""
    holding_per_unit = unit_cost * cfg.holding_cost_rate
    q = eoq(mu * cfg.annual_days, cfg.order_cost, holding_per_unit)
    rows = []
    for sl in levels:
        ss = safety_stock(sigma, cfg.lead_time, sl)
        r = reorder_point(mu, cfg.lead_time, ss)
        sim = simulate_qr(demand, q, r, cfg, unit_cost)
        rows.append({"service_target": float(sl), "fill_rate": sim["fill_rate"],
                     "total_cost": sim["total_cost"]})
    return pd.DataFrame(rows)
