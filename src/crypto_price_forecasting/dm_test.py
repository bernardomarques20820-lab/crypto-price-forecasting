from __future__ import annotations

import math

import numpy as np
import pandas as pd


def dm_test_squared_loss(y_true, y_pred_model, y_pred_benchmark, horizon):
    d = (y_true - y_pred_model) ** 2 - (y_true - y_pred_benchmark) ** 2
    t = len(d)
    d_bar = np.mean(d)
    var_hat = np.var(d)
    dm_stat = d_bar / math.sqrt(var_hat / t) if var_hat > 0 else 0
    p_value = math.erfc(abs(dm_stat) / math.sqrt(2.0))
    return dm_stat, p_value


def run_dm_tests_price(tbl_preds, benchmark_model="NAIVE_ZERO"):
    rows = []
    for (asset, horizon), group in tbl_preds.groupby(["Asset", "Horizon"]):
        bench = group[group["Model"] == benchmark_model]
        for model in group["Model"].unique():
            if model == benchmark_model:
                continue
            cand = group[group["Model"] == model]
            merged = cand.merge(bench, on="Date", suffixes=("_m", "_b"))
            if merged.empty:
                continue
            stat, p_value = dm_test_squared_loss(
                merged["y_true_m"].values,
                merged["y_pred_m"].values,
                merged["y_pred_b"].values,
                horizon,
            )
            rows.append(
                {
                    "Asset": asset,
                    "Horizon": horizon,
                    "Model": model,
                    "Benchmark": benchmark_model,
                    "DM_stat": stat,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)
