"""Demand forecasting: additive Holt-Winters (from scratch) + a seasonal-naive baseline.

Holt-Winters triple exponential smoothing tracks three things a retail demand
series has: a slowly-moving level, a trend, and a repeating weekly pattern. The
smoothing weights are fit by minimising in-sample one-step error. Every forecast
is judged against seasonal-naive ("same weekday last week") so the extra
machinery has to earn its place.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize


def holt_winters_additive(y: np.ndarray, m: int, alpha: float, beta: float,
                          gamma: float, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (in-sample one-step fitted, h-step forecast) for additive HW."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    level = y[:m].mean()
    trend = (y[m:2 * m].mean() - y[:m].mean()) / m if n >= 2 * m else 0.0
    season = list(y[:m] - level)

    fitted = np.empty(n)
    for t in range(n):
        seas = season[t % m]
        fitted[t] = level + trend + seas               # one-step-ahead estimate
        new_level = alpha * (y[t] - seas) + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        season[t % m] = gamma * (y[t] - new_level) + (1 - gamma) * seas
        level, trend = new_level, new_trend

    fc = np.array([level + (h + 1) * trend + season[(n + h) % m]
                   for h in range(horizon)])
    return fitted, np.maximum(fc, 0.0)                  # demand can't be negative


def fit_holt_winters(y: np.ndarray, m: int, horizon: int) -> Dict:
    """Fit (alpha, beta, gamma) by minimising in-sample SSE, then forecast."""
    def sse(params):
        fitted, _ = holt_winters_additive(y, m, *params, horizon=1)
        return float(np.sum((np.asarray(y, float) - fitted) ** 2))

    res = minimize(sse, x0=[0.3, 0.05, 0.3], bounds=[(0, 1)] * 3, method="L-BFGS-B")
    alpha, beta, gamma = res.x
    fitted, forecast = holt_winters_additive(y, m, alpha, beta, gamma, horizon)
    return {"alpha": alpha, "beta": beta, "gamma": gamma,
            "fitted": fitted, "forecast": forecast}


def seasonal_naive(y: np.ndarray, m: int, horizon: int) -> np.ndarray:
    """Forecast = the value one season ago (same weekday last week)."""
    y = np.asarray(y, dtype=float)
    return np.array([y[-m + (h % m)] for h in range(horizon)])


def forecast_metrics(actual: np.ndarray, pred: np.ndarray) -> Dict:
    """MAE, RMSE, WAPE, and bias. WAPE is used because demand has zero days."""
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    err = actual - pred
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "wape": float(np.sum(np.abs(err)) / (np.sum(np.abs(actual)) + 1e-9)),
        "bias": float(np.mean(err)),
    }
