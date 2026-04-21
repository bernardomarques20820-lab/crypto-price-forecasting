from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import align_forecasts, compute_price_forecast_metrics


def build_naive_rw_baseline(price_map, horizons=(1, 7, 14), train_frac=0.8):
    """Baseline random walk: y_hat_{t+h} = y_t."""
    resolved_horizons = tuple(sorted({int(h) for h in horizons}))
    outs_naive = {}
    frames = []

    for asset, series in price_map.items():
        y = pd.Series(series).astype("float64").sort_index().dropna()
        split = int(np.floor(len(y) * train_frac))
        preds = {h: pd.Series(np.nan, index=y.index, dtype="float64") for h in resolved_horizons}
        rows = []

        for h in resolved_horizons:
            start_eval = max(split, (split - 1) + h)
            if start_eval >= len(y):
                continue

            for step in range(start_eval, len(y)):
                preds[h].iloc[step] = float(y.iloc[step - h])

            y_true_s = y.iloc[start_eval:]
            y_pred_s = preds[h].iloc[start_eval:]
            y_lag_s = y.shift(h).iloc[start_eval:]
            aligned = align_forecasts(y_true_s, y_pred_s, y_lag_s)
            if aligned.empty:
                continue

            metrics = compute_price_forecast_metrics(aligned["y_true"], aligned["y_pred"], aligned["y_lag"])
            rows.append(
                {
                    "Asset": asset,
                    "Horizon": h,
                    "RMSE": metrics["RMSE"],
                    "MAE": metrics["MAE"],
                    "DirectionalAccuracy": metrics["DirectionalAccuracy"],
                    "Model": "NAIVE_ZERO",
                }
            )

        mt = pd.DataFrame(rows)
        outs_naive[asset] = {"preds": preds, "y": y, "split": split, "metrics_table": mt}
        frames.append(mt)

    return outs_naive, pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
