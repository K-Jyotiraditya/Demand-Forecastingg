"""Real retail data: download, cache, clean, and aggregate to a demand panel.

Source: the UCI *Online Retail* dataset -- ~541k real e-commerce transactions
(a UK gift retailer, Dec-2010 .. Dec-2011). We strip cancellations and bad rows,
then aggregate to a daily demand panel for the highest-volume SKUs, which is the
clean signal the forecaster and inventory policy consume.
"""
from __future__ import annotations

import logging
import os
import pickle
from typing import Tuple

import pandas as pd

from config import Config

LOGGER = logging.getLogger("supply.data")


def load_retail(cfg: Config) -> pd.DataFrame:
    """Return the raw transaction frame, cached after the first download."""
    if os.path.exists(cfg.cache_path):
        with open(cfg.cache_path, "rb") as fh:
            return pickle.load(fh)

    try:
        df = pd.read_excel(cfg.data_url)
    except Exception as exc:  # noqa: BLE001 - surface any network/parse error
        raise RuntimeError(
            f"Could not download Online Retail data and no cache at "
            f"'{cfg.cache_path}'. Check the network. Original error: {exc}"
        ) from exc

    with open(cfg.cache_path, "wb") as fh:
        pickle.dump(df, fh)
    LOGGER.info("Downloaded + cached %d transactions", len(df))
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Drop cancellations, returns, and non-positive prices/quantities."""
    out = df.copy()
    out["InvoiceNo"] = out["InvoiceNo"].astype(str)
    out = out[~out["InvoiceNo"].str.startswith("C")]        # 'C...' = cancellation
    out = out[(out["Quantity"] > 0) & (out["UnitPrice"] > 0)]
    out = out.dropna(subset=["StockCode"])
    out["date"] = pd.to_datetime(out["InvoiceDate"]).dt.normalize()
    return out


def build_demand_panel(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.Series]:
    """Daily demand for the top-N SKUs (zero-filled calendar) + mean unit price.

    SKUs are ranked by volume in the *training* window (excluding the hold-out),
    so every modelled SKU has real history and we never pick a cold-start product
    whose sales live entirely in the test period.
    """
    train_cut = df["date"].max() - pd.Timedelta(days=cfg.holdout_days)
    ranking = df[df["date"] <= train_cut].groupby("StockCode")["Quantity"].sum()
    top = ranking.nlargest(cfg.top_n_skus).index
    sub = df[df["StockCode"].isin(top)]

    panel = (sub.groupby(["date", "StockCode"])["Quantity"].sum()
             .unstack(fill_value=0.0))
    full = pd.date_range(panel.index.min(), panel.index.max(), freq="D")
    panel = panel.reindex(full, fill_value=0.0).astype(float)

    prices = sub.groupby("StockCode")["UnitPrice"].mean().reindex(panel.columns)
    LOGGER.info("Demand panel: %d days x %d SKUs", panel.shape[0], panel.shape[1])
    return panel, prices
