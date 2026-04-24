"""Multi-product procurement allocation via linear programming.

Given a forecast, a finite budget, and finite warehouse capacity, how many units
of each SKU should we buy for the next cycle to maximise profit? With linear
margins and capped demand this is a linear program:

    maximise  sum_i  margin_i * x_i
    s.t.      sum_i  cost_i  * x_i  <=  budget
              sum_i           x_i  <=  capacity
              0 <= x_i <= forecast_demand_i

Solved with the HiGHS solver behind ``scipy.optimize.linprog``.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import linprog


def allocate(margins: np.ndarray, unit_costs: np.ndarray, demand_caps: np.ndarray,
             budget: float, capacity: float) -> Dict:
    """Profit-maximising order quantities under budget + capacity constraints."""
    margins = np.asarray(margins, float)
    unit_costs = np.asarray(unit_costs, float)
    demand_caps = np.asarray(demand_caps, float)

    res = linprog(
        c=-margins,                                  # linprog minimises
        A_ub=np.vstack([unit_costs, np.ones_like(unit_costs)]),
        b_ub=[budget, capacity],
        bounds=[(0.0, cap) for cap in demand_caps],
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"LP did not solve: {res.message}")

    x = res.x
    return {
        "quantities": x,
        "profit": float(margins @ x),
        "budget_used": float(unit_costs @ x),
        "capacity_used": float(x.sum()),
        "fill_vs_demand": float(x.sum() / (demand_caps.sum() + 1e-9)),
    }


def proportional_baseline(demand_caps: np.ndarray, unit_costs: np.ndarray,
                          margins: np.ndarray, budget: float, capacity: float) -> Dict:
    """Naive baseline: scale every SKU's demand equally to fit the tighter limit."""
    demand_caps = np.asarray(demand_caps, float)
    scale = min(budget / (unit_costs @ demand_caps + 1e-9),
                capacity / (demand_caps.sum() + 1e-9), 1.0)
    x = demand_caps * scale
    return {"quantities": x, "profit": float(np.asarray(margins, float) @ x)}
