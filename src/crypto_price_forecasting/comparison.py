from __future__ import annotations

import ast
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA

from .baselines import build_naive_rw_baseline
from .config import APPROVED_ARIMA_ORDERS, ARIMA_RESIDUAL_ORDERS, MAIN_PRICE_ASSETS, MAIN_PRICE_HORIZONS
from .dm_test import run_dm_tests_price
from .metrics import build_price_nn_table, compute_price_forecast_metrics


def _coerce_arima_order(order_value):
    if isinstance(order_value, str):
        order_value = ast.literal_eval(order_value)
    if isinstance(order_value, np.ndarray):
        order_value = order_value.tolist()
    if isinstance(order_value, (list, tuple)) and len(order_value) == 3:
        return tuple(int(v) for v in order_value)
    raise ValueError(f"Invalid ARIMA order: {order_value!r}")


def _empty_family_metrics(asset, horizons, model):
    return pd.DataFrame(
        [
            {
                "Asset": asset,
                "Horizon": int(h),
                "RMSE": np.nan,
                "MAE": np.nan,
                "MAPE": np.nan,
                "sMAPE": np.nan,
                "TheilU1": np.nan,
                "TheilU2": np.nan,
                "DirectionalAccuracy": np.nan,
                "Model": model,
            }
            for h in horizons
        ]
    )


def build_arima_price_table(
    price_map,
    arima_orders=APPROVED_ARIMA_ORDERS,
    assets=MAIN_PRICE_ASSETS,
    horizons=MAIN_PRICE_HORIZONS,
    train_frac=0.8,
):
    """Walk-forward ARIMA em precos (d=1) com ordens fixadas."""
    resolved_assets = tuple(assets)
    resolved_horizons = tuple(int(h) for h in horizons)

    outs_arima_price = {}
    frames = []

    for asset in resolved_assets:
        series = price_map.get(asset)
        order = arima_orders.get(asset, (2, 1, 2))

        if series is None:
            tbl = _empty_family_metrics(asset, resolved_horizons, "ARIMA")
            outs_arima_price[asset] = {"preds": {}, "y": pd.Series(dtype="float64", name=asset), "split": 0, "metrics_table": tbl.copy()}
            frames.append(tbl)
            continue

        y = pd.Series(series).astype("float64").sort_index().dropna()
        split = int(np.floor(len(y) * train_frac))
        preds = {h: pd.Series(np.nan, index=y.index, dtype="float64") for h in resolved_horizons}
        metrics_rows = []

        if y.empty or split < 30 or split >= len(y):
            tbl = _empty_family_metrics(asset, resolved_horizons, "ARIMA")
            outs_arima_price[asset] = {"preds": preds, "y": y, "split": split, "metrics_table": tbl.copy(), "order": order}
            frames.append(tbl)
            continue

        fitted = None
        try:
            fitted = ARIMA(y.iloc[:split], order=order).fit()
        except Exception as exc:
            print(f"[ARIMA_PRICE][{asset}] fit failure for {order}: {exc}")

        last_seen_idx = split - 1
        for t in range(split, len(y)):
            if fitted is not None:
                for h in resolved_horizons:
                    target_pos = last_seen_idx + h
                    if target_pos < len(y):
                        try:
                            forecast = fitted.forecast(steps=h)
                            preds[h].iloc[target_pos] = float(np.asarray(forecast)[-1])
                        except Exception as exc:
                            print(f"[ARIMA_PRICE][{asset}][h={h}] forecast fail step {t}: {exc}")
            if fitted is not None:
                try:
                    fitted = fitted.append(endog=[float(y.iloc[t])], refit=False)
                except Exception:
                    try:
                        fitted = ARIMA(y.iloc[: t + 1], order=order).fit()
                    except Exception as exc:
                        print(f"[ARIMA_PRICE][{asset}] update fail step {t}: {exc}")
                        fitted = None
            last_seen_idx = t

        for h in resolved_horizons:
            start_eval = max(split, (split - 1) + h)
            y_true_s = y.iloc[start_eval:]
            y_pred_s = preds[h].iloc[start_eval:]
            y_lag_s = y.shift(h).iloc[start_eval:]
            metrics = compute_price_forecast_metrics(y_true_s, y_pred_s, y_lag_s)
            metrics_rows.append({"Asset": asset, "Horizon": int(h), **metrics, "Model": "ARIMA"})

        tbl = pd.DataFrame(metrics_rows)
        outs_arima_price[asset] = {"preds": preds, "y": y, "split": split, "metrics_table": tbl.copy(), "order": order}
        frames.append(tbl)

    tbl_arima_price = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["Asset", "Horizon", "RMSE", "MAE", "MAPE", "sMAPE", "TheilU1", "TheilU2", "DirectionalAccuracy", "Model"])
    )
    for col in ["Horizon", "RMSE", "MAE", "MAPE", "sMAPE", "TheilU1", "TheilU2", "DirectionalAccuracy"]:
        tbl_arima_price[col] = pd.to_numeric(tbl_arima_price[col], errors="coerce")
    return outs_arima_price, tbl_arima_price.sort_values(["Asset", "Horizon"]).reset_index(drop=True)


def build_price_family_comparison_table(arima_table, ets_table, lstm_table, rnn_table, gru_table, assets=MAIN_PRICE_ASSETS, horizons=MAIN_PRICE_HORIZONS):
    resolved_assets = tuple(assets)
    resolved_horizons = tuple(int(h) for h in horizons)
    required_cols = ["Asset", "Horizon", "Model", "RMSE", "MAE", "MAPE", "sMAPE", "TheilU1", "TheilU2", "DirectionalAccuracy"]

    def _prepare(df, fallback_model):
        base = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(columns=required_cols)
        for col in required_cols:
            if col not in base.columns:
                base[col] = np.nan
        if "Model" in base.columns:
            base["Model"] = base["Model"].fillna(fallback_model).astype(str)
            base.loc[base["Model"].eq("nan"), "Model"] = fallback_model
        else:
            base["Model"] = fallback_model
        base["Asset"] = base["Asset"].astype(str)
        for col in ["Horizon", "RMSE", "MAE", "MAPE", "sMAPE", "TheilU1", "TheilU2", "DirectionalAccuracy"]:
            base[col] = pd.to_numeric(base[col], errors="coerce")
        base = base[base["Asset"].isin(resolved_assets) & base["Horizon"].isin(resolved_horizons)]
        return base[required_cols].copy()

    frames = [
        _prepare(arima_table, "ARIMA"),
        _prepare(ets_table, "ETS"),
        _prepare(lstm_table, "LSTM"),
        _prepare(rnn_table, "RNN"),
        _prepare(gru_table, "GRU"),
    ]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=required_cols)
    if combined.empty:
        return combined
    combined["Model"] = pd.Categorical(combined["Model"], categories=["ARIMA", "ETS", "LSTM", "RNN", "GRU"], ordered=True)
    combined = combined.sort_values(["Asset", "Horizon", "RMSE", "Model"], na_position="last").reset_index(drop=True)
    combined["Model"] = combined["Model"].astype(str)
    return combined


def build_price_family_winners_table(tbl):
    rows = []
    for (asset, horizon), group in tbl.groupby(["Asset", "Horizon"], sort=True):
        def _best(col, ascending=True):
            g = group.dropna(subset=[col]).sort_values([col, "Model"], ascending=[ascending, True])
            return (g.iloc[0]["Model"], float(g.iloc[0][col])) if not g.empty else (pd.NA, np.nan)

        brm, brv = _best("RMSE")
        bmm, bmv = _best("MAE")
        bdm, bdv = _best("DirectionalAccuracy", ascending=False)
        rows.append(
            {
                "Asset": asset,
                "Horizon": int(horizon),
                "best_rmse_model": brm,
                "best_rmse": brv,
                "best_mae_model": bmm,
                "best_mae": bmv,
                "best_diracc_model": bdm,
                "best_diracc": bdv,
            }
        )
    return pd.DataFrame(rows).sort_values(["Asset", "Horizon"]).reset_index(drop=True)


def build_price_family_rank_summary(tbl):
    base = tbl.copy()
    base["rmse_rank"] = base.groupby(["Asset", "Horizon"])["RMSE"].rank(method="min", ascending=True)
    base["mae_rank"] = base.groupby(["Asset", "Horizon"])["MAE"].rank(method="min", ascending=True)
    base["diracc_rank"] = base.groupby(["Asset", "Horizon"])["DirectionalAccuracy"].rank(method="min", ascending=False)
    summary = (
        base.groupby("Model", sort=True)
        .agg(
            mean_rmse_rank=("rmse_rank", "mean"),
            mean_mae_rank=("mae_rank", "mean"),
            mean_diracc_rank=("diracc_rank", "mean"),
            rmse_wins=("rmse_rank", lambda s: int((s == 1).sum())),
            mae_wins=("mae_rank", lambda s: int((s == 1).sum())),
            diracc_wins=("diracc_rank", lambda s: int((s == 1).sum())),
        )
        .reset_index()
        .sort_values(["mean_rmse_rank", "mean_mae_rank", "mean_diracc_rank", "Model"])
        .reset_index(drop=True)
    )
    return summary


def merge_family_with_naive_baseline(tbl_family_price_all, tbl_naive_price):
    family_cols = ["Asset", "Horizon", "Model", "RMSE", "MAE", "MAPE", "sMAPE", "TheilU1", "TheilU2", "DirectionalAccuracy"]
    base = tbl_family_price_all.copy()
    for col in family_cols:
        if col not in base.columns:
            base[col] = pd.NA if col in ["Asset", "Model"] else np.nan
    base = base[family_cols].copy()

    if not base.empty and "Model" in base.columns:
        base = base.loc[base["Model"].astype(str) != "NAIVE_ZERO"].copy()

    naive = tbl_naive_price.copy()
    for col in family_cols:
        if col not in naive.columns:
            naive[col] = pd.NA if col in ["Asset", "Model"] else np.nan
    naive = naive[family_cols]

    merged = pd.concat([base, naive], ignore_index=True) if not base.empty else naive.copy()
    for col in ["Horizon", "RMSE", "MAE", "MAPE", "sMAPE", "TheilU1", "TheilU2", "DirectionalAccuracy"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
    return merged.sort_values(["Asset", "Horizon", "RMSE", "Model"], na_position="last").reset_index(drop=True)


def build_price_family_predictions_table(model_outputs, assets=None, horizons=(1, 7, 14)):
    resolved_horizons = tuple(int(h) for h in horizons)
    resolved_assets = set(assets) if assets is not None else None
    rows = []
    for model_name, outs_dict in model_outputs.items():
        for asset, out in outs_dict.items():
            if resolved_assets is not None and asset not in resolved_assets:
                continue
            y = pd.Series(out.get("y")).dropna()
            preds = out.get("preds")
            split = out.get("split", 0)
            for h in resolved_horizons:
                if h not in preds:
                    continue
                start_eval = max(split, (split - 1) + h)
                df_h = pd.DataFrame(
                    {
                        "Asset": asset,
                        "Horizon": h,
                        "Model": model_name,
                        "Date": y.index[start_eval:],
                        "y_true": y.iloc[start_eval:].values,
                        "y_pred": preds[h].iloc[start_eval:].values,
                    }
                ).dropna()
                rows.append(df_h)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["Asset", "Horizon", "Model", "Date", "y_true", "y_pred"])


def rerun_price_networks_with_seeds(price_map, seeds=(7, 13, 29), horizons=(1, 7, 14), train_frac=0.8):
    from .neural import run_gru_price_multiasset, run_lstm_price_multiasset

    rows = []
    for seed in seeds:
        outs_lstm = run_lstm_price_multiasset(price_map, horizons, train_frac, lstm_kwargs={"seed": seed})
        tbl_lstm = build_price_nn_table(outs_lstm, "LSTM")
        tbl_lstm["Seed"] = seed
        rows.append(tbl_lstm)

        outs_gru = run_gru_price_multiasset(price_map, horizons, train_frac, gru_kwargs={"seed": seed})
        tbl_gru = build_price_nn_table(outs_gru, "GRU")
        tbl_gru["Seed"] = seed
        rows.append(tbl_gru)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_network_seed_summary(tbl):
    if tbl is None or tbl.empty:
        return pd.DataFrame(
            columns=[
                "Asset",
                "Horizon",
                "Model",
                "rmse_mean",
                "rmse_std",
                "mae_mean",
                "mae_std",
                "directional_accuracy_mean",
                "directional_accuracy_std",
            ]
        )
    return (
        tbl.groupby(["Asset", "Horizon", "Model"])
        .agg(
            rmse_mean=("RMSE", "mean"),
            rmse_std=("RMSE", "std"),
            mae_mean=("MAE", "mean"),
            mae_std=("MAE", "std"),
            directional_accuracy_mean=("DirectionalAccuracy", "mean"),
            directional_accuracy_std=("DirectionalAccuracy", "std"),
        )
        .reset_index()
    )


def build_directional_accuracy_detailed_table(family_outputs, horizons=(1, 7, 14)):
    rows = []
    for model_name, outs_dict in family_outputs.items():
        for asset, out in outs_dict.items():
            y = pd.Series(out.get("y")).dropna()
            preds = out.get("preds", {})
            split = out.get("split", 0)
            for h in horizons:
                if h not in preds:
                    continue
                start_eval = max(split, (split - 1) + h)
                y_true_s = y.iloc[start_eval:]
                y_pred_s = preds[h].iloc[start_eval:]
                y_lag_s = y.shift(h).iloc[start_eval:]

                metrics = compute_price_forecast_metrics(y_true_s, y_pred_s, y_lag_s)
                aligned = pd.DataFrame({"y_true": y_true_s, "y_pred": y_pred_s, "y_lag": y_lag_s}).dropna()
                if aligned.empty:
                    continue

                real_dir = np.sign(aligned["y_true"].values - aligned["y_lag"].values)
                pred_dir = np.sign(aligned["y_pred"].values - aligned["y_lag"].values)

                mask_up = real_dir == 1
                mask_down = real_dir == -1
                hit_up = float(np.mean(pred_dir[mask_up] == real_dir[mask_up])) if mask_up.any() else np.nan
                hit_down = float(np.mean(pred_dir[mask_down] == real_dir[mask_down])) if mask_down.any() else np.nan

                rows.append(
                    {
                        "Ativo": asset,
                        "Horizonte": h,
                        "Modelo": model_name,
                        "N": int(len(aligned)),
                        "Acuracia": round(float(metrics["DirectionalAccuracy"]), 4),
                        "Hit (Alta)": round(hit_up, 4),
                        "Hit (Baixa)": round(hit_down, 4),
                    }
                )

    return pd.DataFrame(rows).sort_values("Acuracia", ascending=False).reset_index(drop=True)


def build_arima_residual_diagnostics_table(price_map, arima_orders=ARIMA_RESIDUAL_ORDERS):
    rows = []
    for asset, order in arima_orders.items():
        y = pd.Series(price_map[asset]).astype("float64").sort_index().dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ARIMA(y, order=order).fit()
        resid = pd.Series(fit.resid).dropna()
        lb = acorr_ljungbox(resid, lags=[10, 20, 30], return_df=True)
        arch = het_arch(resid, nlags=10)
        rows.append(
            {
                "Ativo": asset,
                "Ordem": f"({order[0]},{order[1]},{order[2]})",
                "AIC": round(float(fit.aic), 2),
                "LB(10) p": round(float(lb.loc[10, "lb_pvalue"]), 4),
                "LB(20) p": round(float(lb.loc[20, "lb_pvalue"]), 4),
                "LB(30) p": round(float(lb.loc[30, "lb_pvalue"]), 4),
                "ARCH-LM p": round(float(arch[1]), 4),
            }
        )
    return pd.DataFrame(rows)
