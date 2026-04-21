from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .metrics import align_forecasts, compute_price_forecast_metrics


def _smape_np(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    denom = (np.abs(y) + np.abs(yhat)) / 2.0
    ok = denom > 0
    out = np.zeros_like(y, float)
    out[ok] = np.abs(y[ok] - yhat[ok]) / denom[ok]
    return float(np.mean(out) * 100)


def _aic_or_inf(fit_res):
    try:
        return float(fit_res.aic)
    except Exception:
        return float("inf")


def _make_candidates(
    trend_opts=("add", "mul", None),
    seasonal_opts=("add", "mul", None),
    seasonal_periods_opts=(None, 7),
    damped_opts=(False, True),
):
    """Gera combinações válidas para ExponentialSmoothing em preços."""
    cands = []
    for tr in trend_opts:
        for se in seasonal_opts:
            for seasonal_periods in seasonal_periods_opts:
                for damp in damped_opts:
                    if se is None and seasonal_periods is not None:
                        continue
                    if se in ("add", "mul") and (seasonal_periods is None or int(seasonal_periods) < 2):
                        continue
                    cands.append(
                        {
                            "trend": tr,
                            "seasonal": se,
                            "seasonal_periods": seasonal_periods,
                            "damped_trend": damp,
                        }
                    )
    return cands


def _fit_once(y, cfg, use_boxcox=None):
    try:
        mod = ExponentialSmoothing(
            y,
            trend=cfg["trend"],
            damped_trend=cfg["damped_trend"],
            seasonal=cfg["seasonal"],
            seasonal_periods=cfg["seasonal_periods"],
            initialization_method="estimated",
        )
        return mod.fit(optimized=True, use_boxcox=use_boxcox, remove_bias=False)
    except Exception:
        return None


def _auto_boxcox_flag(y, cfg):
    needs_mul = (cfg["trend"] == "mul") or (cfg["seasonal"] == "mul")
    if not needs_mul:
        return None
    return "log" if np.nanmin(np.asarray(y, float)) <= 0 else None


def _select_best_ets(y_train, select_by="smape", val_frac=0.2, candidates=None):
    if candidates is None:
        candidates = _make_candidates()
    n = len(y_train)
    val_start = int(np.floor((1 - val_frac) * n)) if select_by == "smape" else None
    y_tr = y_train[:val_start] if val_start else y_train
    y_val = y_train[val_start:] if val_start else None

    best_cfg = None
    best_score = np.inf

    for cfg in candidates:
        use_boxcox = _auto_boxcox_flag(y_train, cfg)
        if select_by == "smape":
            fit = _fit_once(y_tr, cfg, use_boxcox=use_boxcox)
            if fit is None or y_val is None or len(y_val) == 0:
                continue
            try:
                fc = np.asarray(fit.forecast(len(y_val)), float)
                score = _smape_np(y_val.values, fc)
            except Exception:
                score = np.inf
        else:
            fit = _fit_once(y_train, cfg, use_boxcox=use_boxcox)
            if fit is None:
                continue
            score = _aic_or_inf(fit)

        if score < best_score:
            best_score = score
            best_cfg = {**cfg, "use_boxcox": use_boxcox}

    return best_cfg


def ets_walk_forward_price(
    y_series,
    horizons=(1, 7, 14),
    train_frac=0.8,
    select_by="smape",
    val_frac=0.2,
    candidates=None,
    refit_every=5,
):
    y = pd.Series(y_series).astype("float64").sort_index().dropna()
    n = len(y)
    if n < 50:
        raise ValueError("Série muito curta.")

    resolved_horizons = tuple(sorted({int(h) for h in horizons}))
    split = int(np.floor(train_frac * n))
    y_train = y.iloc[:split]

    if candidates is None:
        candidates = _make_candidates()

    best_cfg = _select_best_ets(y_train, select_by=select_by, val_frac=val_frac, candidates=candidates)
    if best_cfg is None:
        raise RuntimeError("Nenhuma configuração ETS pôde ser ajustada.")

    use_boxcox = best_cfg.get("use_boxcox", None)
    fit = _fit_once(y_train, best_cfg, use_boxcox=use_boxcox)
    if fit is None:
        raise RuntimeError("Falha ao ajustar ETS com a melhor configuração no treino.")

    preds = {h: pd.Series(np.nan, index=y.index, dtype="float64") for h in resolved_horizons}
    last_seen_idx = split - 1
    step_count = 0

    for t in range(split, n):
        try:
            max_h = max(resolved_horizons)
            fc_all = np.asarray(fit.forecast(max_h), float)
            for h in resolved_horizons:
                target_pos = last_seen_idx + h
                if target_pos < n:
                    preds[h].iloc[target_pos] = fc_all[h - 1]
        except Exception:
            pass

        step_count += 1
        if step_count % int(refit_every) == 0:
            y_seen = y.iloc[: t + 1]
            fit = _fit_once(y_seen, best_cfg, use_boxcox=use_boxcox) or fit

        last_seen_idx = t

    metrics_rows = []
    for h in resolved_horizons:
        start_eval = max(split, (split - 1) + h)
        y_true_s = y.iloc[start_eval:]
        y_pred_s = preds[h].iloc[start_eval:]
        y_lag_s = y.shift(h).iloc[start_eval:]

        aligned = align_forecasts(y_true_s, y_pred_s, y_lag_s)

        if aligned.empty:
            metrics_rows.append(
                {
                    "Horizon": int(h),
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "MAPE": np.nan,
                    "sMAPE": np.nan,
                    "TheilU1": np.nan,
                    "TheilU2": np.nan,
                    "DirectionalAccuracy": np.nan,
                }
            )
            continue

        row = {
            "Horizon": int(h),
            **compute_price_forecast_metrics(
                aligned["y_true"],
                aligned["y_pred"],
                aligned["y_lag"],
            ),
        }
        metrics_rows.append(row)

    metrics_table = pd.DataFrame(metrics_rows)[
        ["Horizon", "RMSE", "MAE", "MAPE", "sMAPE", "TheilU1", "TheilU2", "DirectionalAccuracy"]
    ]
    return {
        "preds": preds,
        "y_true": y,
        "y": y,
        "split": split,
        "metrics_table": metrics_table,
        "best_cfg": best_cfg,
    }


def evaluate_ets_price_multiasset(price_map, horizons=(1, 7, 14), train_frac=0.8):
    resolved_horizons = tuple(sorted({int(h) for h in horizons}))
    outs_ets_price = {}
    candidates = _make_candidates()

    for asset, series in price_map.items():
        try:
            print(f"[ETS_PRICE][{asset}] iniciando avaliação")
            series_named = pd.Series(series).copy()
            series_named.name = asset
            out = ets_walk_forward_price(
                series_named,
                horizons=resolved_horizons,
                train_frac=train_frac,
                select_by="aic",
                val_frac=0.2,
                candidates=candidates,
                refit_every=5,
            )
            metrics_table = out["metrics_table"].copy()
            metrics_table["Asset"] = asset
            outs_ets_price[asset] = {
                "preds": out["preds"],
                "y_true": out["y_true"],
                "y": out["y"],
                "split": out["split"],
                "metrics_table": metrics_table[
                    ["Asset", "Horizon", "RMSE", "MAE", "MAPE", "sMAPE", "TheilU1", "TheilU2", "DirectionalAccuracy"]
                ],
                "best_cfg": out.get("best_cfg"),
            }
            print(f"[ETS_PRICE][{asset}] OK  cfg={out.get('best_cfg')}")
        except Exception as exc:
            print(f"[ETS_PRICE][{asset}] FALHOU: {exc}")
            y_s = pd.Series(series).sort_index().dropna()
            split = int(np.floor(len(y_s) * train_frac)) if len(y_s) else 0
            empty_metrics = pd.DataFrame(
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
                    for h in resolved_horizons
                ]
            )
            outs_ets_price[asset] = {
                "preds": {h: pd.Series(np.nan, index=y_s.index, dtype="float64") for h in resolved_horizons},
                "y_true": y_s,
                "y": y_s,
                "split": split,
                "metrics_table": empty_metrics,
                "best_cfg": None,
            }

    return outs_ets_price


def build_ets_config_table(outs_ets_price):
    rows = []
    for asset, out in outs_ets_price.items():
        cfg = out.get("best_cfg")
        if cfg is None:
            rows.append({"Ativo": asset, "Trend": "—", "Seasonal": "—", "Damped": "—", "Periods": "—", "AIC": np.nan})
            continue
        y_train = out["y"].iloc[: out["split"]]
        try:
            fit = ExponentialSmoothing(
                y_train,
                trend=cfg["trend"],
                damped_trend=cfg["damped_trend"],
                seasonal=cfg["seasonal"],
                seasonal_periods=cfg["seasonal_periods"],
                initialization_method="estimated",
            ).fit(optimized=True, use_boxcox=cfg.get("use_boxcox"), remove_bias=False)
            aic = float(fit.aic)
        except Exception:
            aic = np.nan
        rows.append(
            {
                "Ativo": asset,
                "Trend": cfg["trend"] if cfg["trend"] else "None",
                "Seasonal": cfg["seasonal"] if cfg["seasonal"] else "None",
                "Damped": "sim" if cfg["damped_trend"] else "não",
                "Periods": cfg["seasonal_periods"] if cfg["seasonal_periods"] else "—",
                "AIC": round(aic, 2),
            }
        )
    return pd.DataFrame(rows)
