from __future__ import annotations

import numpy as np
import pandas as pd


def align_forecasts(y_true, y_pred, y_lag=None):
    data = {"y_true": y_true, "y_pred": y_pred}
    if y_lag is not None:
        data["y_lag"] = y_lag
    return pd.DataFrame(data).dropna()


def compute_basic_forecast_metrics(y_true, y_pred):
    aligned = align_forecasts(y_true, y_pred)
    if aligned.empty:
        return {
            "RMSE": np.nan,
            "MAE": np.nan,
            "MAPE": np.nan,
            "sMAPE": np.nan,
            "TheilU1": np.nan,
            "TheilU2": np.nan,
            "DirectionalAccuracy": np.nan,
        }

    y = aligned["y_true"].astype(float).values
    yhat = aligned["y_pred"].astype(float).values
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    mae = float(np.mean(np.abs(y - yhat)))
    mask_mape = y != 0
    mape = float(np.mean(np.abs((y[mask_mape] - yhat[mask_mape]) / y[mask_mape])) * 100) if mask_mape.any() else np.nan
    denom = (np.abs(y) + np.abs(yhat)) / 2.0
    mask_smape = denom > 0
    smape = float(np.mean(np.abs(y[mask_smape] - yhat[mask_smape]) / denom[mask_smape]) * 100) if mask_smape.any() else np.nan
    u1_num = np.sqrt(np.mean((yhat - y) ** 2))
    u1_den = np.sqrt(np.mean(y**2)) + np.sqrt(np.mean(yhat**2))
    theil_u1 = float(u1_num / u1_den) if u1_den > 1e-12 else np.nan
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "sMAPE": smape,
        "TheilU1": theil_u1,
        "TheilU2": np.nan,
        "DirectionalAccuracy": np.nan,
    }


def compute_directional_accuracy_price(y_true, y_pred, y_lag):
    aligned = align_forecasts(y_true, y_pred, y_lag)
    if aligned.empty:
        return np.nan
    return float(
        np.mean(
            np.sign(aligned["y_pred"].values - aligned["y_lag"].values)
            == np.sign(aligned["y_true"].values - aligned["y_lag"].values)
        )
    )


def compute_theil_u2(y_true, y_pred, y_lag):
    aligned = align_forecasts(y_true, y_pred, y_lag)
    if aligned.empty:
        return np.nan
    yt = aligned["y_true"].astype(float).values
    yp = aligned["y_pred"].astype(float).values
    yl = aligned["y_lag"].astype(float).values
    mse_model = np.mean((yp - yt) ** 2)
    mse_naive = np.mean((yl - yt) ** 2)
    return float(np.sqrt(mse_model / mse_naive)) if mse_naive > 1e-12 else np.nan


def compute_price_forecast_metrics(y_true, y_pred, y_lag=None):
    metrics = compute_basic_forecast_metrics(y_true, y_pred)
    if y_lag is not None:
        metrics["TheilU2"] = compute_theil_u2(y_true, y_pred, y_lag)
        metrics["DirectionalAccuracy"] = compute_directional_accuracy_price(y_true, y_pred, y_lag)
    return metrics


def empty_price_nn_metrics_row(horizon):
    return {
        "Horizon": int(horizon),
        "RMSE": np.nan,
        "MAE": np.nan,
        "MAPE": np.nan,
        "sMAPE": np.nan,
        "TheilU1": np.nan,
        "TheilU2": np.nan,
        "DirectionalAccuracy": np.nan,
    }


def build_price_nn_table(outs_dict, model_label):
    frames = []
    for asset, out in outs_dict.items():
        metrics_table = out.get("metrics_table")
        if not isinstance(metrics_table, pd.DataFrame) or metrics_table.empty:
            metrics_table = pd.DataFrame(
                [
                    {
                        "Asset": asset,
                        "Horizon": h,
                        "RMSE": np.nan,
                        "MAE": np.nan,
                        "MAPE": np.nan,
                        "sMAPE": np.nan,
                        "TheilU1": np.nan,
                        "TheilU2": np.nan,
                        "DirectionalAccuracy": np.nan,
                    }
                    for h in sorted(out.get("preds", {}).keys())
                ]
            )
        metrics_table = metrics_table.copy()
        if "Asset" not in metrics_table.columns:
            metrics_table["Asset"] = asset
        for col in ["Horizon", "RMSE", "MAE", "MAPE", "sMAPE", "TheilU1", "TheilU2", "DirectionalAccuracy"]:
            if col not in metrics_table.columns:
                metrics_table[col] = np.nan
        frames.append(
            metrics_table[["Asset", "Horizon", "RMSE", "MAE", "MAPE", "sMAPE", "TheilU1", "TheilU2", "DirectionalAccuracy"]]
        )

    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["Asset", "Horizon", "RMSE", "MAE", "MAPE", "sMAPE", "TheilU1", "TheilU2", "DirectionalAccuracy"]
    )
    table["Model"] = model_label
    return table[
        ["Asset", "Horizon", "RMSE", "MAE", "MAPE", "sMAPE", "TheilU1", "TheilU2", "DirectionalAccuracy", "Model"]
    ].sort_values(["Asset", "Horizon"]).reset_index(drop=True)


def tidy_metrics(outs, model_name="MODEL"):
    needed = ["RMSE", "MAE", "MAPE", "sMAPE", "TheilU1", "TheilU2"]
    frames = []
    for asset, out in outs.items():
        dfm = out.get("metrics_table")
        if not isinstance(dfm, pd.DataFrame) or dfm.empty:
            dfm = pd.DataFrame(out.get("metrics", {})).T
        if dfm is None or dfm.empty:
            continue
        for col in needed:
            if col not in dfm.columns:
                dfm[col] = np.nan
        dfm = dfm[needed].copy()
        try:
            horizon = pd.Index(dfm.index).astype(int)
        except Exception:
            horizon = pd.to_numeric(pd.Index(dfm.index.astype(str)).str.extract(r"(\d+)")[0], errors="coerce")
        dfm.insert(0, "Horizon", horizon)
        dfm.insert(0, "Asset", asset)
        dfm["Model"] = model_name
        frames.append(dfm.reset_index(drop=True))
    if not frames:
        return pd.DataFrame(columns=["Asset", "Horizon", "RMSE", "MAE", "MAPE", "sMAPE", "TheilU1", "TheilU2", "Model"])
    res = pd.concat(frames, ignore_index=True)
    for col in ["Horizon", "RMSE", "MAE", "MAPE", "sMAPE", "TheilU1", "TheilU2"]:
        res[col] = pd.to_numeric(res[col], errors="coerce")
    return res[["Asset", "Horizon", "RMSE", "MAE", "MAPE", "sMAPE", "TheilU1", "TheilU2", "Model"]]


def build_directional_tables(outs_dict):
    tables = {}
    for asset, out in outs_dict.items():
        y = out["y"]
        preds = out["preds"]
        split = out["split"]
        by_h = {}
        for h, yhat in preds.items():
            start_eval = max(split, (split - 1) + h)
            df_h = pd.DataFrame({"y_true": y.iloc[start_eval:], "y_pred": yhat.iloc[start_eval:]}).dropna()
            if not df_h.empty:
                by_h[h] = df_h
        if by_h:
            tables[asset] = by_h
    return tables
