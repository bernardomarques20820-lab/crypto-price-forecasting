from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss

from .config import ARIMA_DIAGNOSTIC_ORDERS
from .visualization import (
    compare_price_diff_vs_log_return_thesis,
    plot_acf_pacf_diagnostics_thesis,
    plot_differenced_acf_pacf_thesis,
    plot_series_and_differences_thesis,
    save_thesis_figure,
)


def stationarity_report(series, name="series"):
    s = pd.Series(series.values if hasattr(series, "values") else series).dropna()
    print(f"=== Stationarity Report: {name} ===")
    print(f"Sample size: {len(s)}")

    adf_stat, adf_p, adf_lags, *_ = adfuller(s, autolag="AIC")
    adf_interp = "Stationary (p < 0.05)" if adf_p < 0.05 else "Non-stationary (p >= 0.05)"
    print("ADF Test:")
    print(f"  Statistic : {adf_stat:.4f}")
    print(f"  p-value   : {adf_p:.4f}")
    print(f"  Lags used : {adf_lags}")
    print(f"  Result    : {adf_interp}")

    print("KPSS Test:")
    try:
        kpss_stat, kpss_p, kpss_lags, _ = kpss(s, regression="c", nlags="auto")
        kpss_interp = "Stationary (p > 0.05)" if kpss_p > 0.05 else "Non-stationary (p <= 0.05)"
        print(f"  Statistic : {kpss_stat:.4f}")
        print(f"  p-value   : {kpss_p:.4f}")
        print(f"  Lags used : {kpss_lags}")
        print(f"  Result    : {kpss_interp}")
    except Exception as exc:
        print(f"  KPSS could not be computed: {exc}")


def plot_series_and_differences(series, name="series"):
    fig = plot_series_and_differences_thesis(series, name=name)
    save_thesis_figure(fig, f"price_fig_series_diff_{name.replace('-', '_')}")
    plt.show()
    return fig


def plot_acf_pacf_diagnostics(series, name="series", lags=30):
    s = pd.Series(series.values if hasattr(series, "values") else series).dropna()
    max_lags = min(lags, len(s) // 2 - 1)
    if max_lags < 1:
        print(f"{name}: series too short to plot ACF/PACF (n={len(s)})")
        return None
    try:
        fig = plot_acf_pacf_diagnostics_thesis(s, name=name, lags=lags)
        save_thesis_figure(fig, f"price_fig_acf_pacf_{name.replace('-', '_')}")
        plt.show()
        return fig
    except Exception as exc:
        print(f"{name}: ACF/PACF plot failed - {exc}")
        return None


def compare_integration_orders(series, name="series"):
    s = pd.Series(series.values if hasattr(series, "values") else series).dropna()
    diff1 = s.diff().dropna()

    print("=" * 50)
    print(f"Integration order analysis: {name}")
    print("=" * 50)

    print("-- Level --")
    _, adf_p_l, _, *_ = adfuller(s, autolag="AIC")
    level_stationary = adf_p_l < 0.05
    level_label = "Stationary" if level_stationary else "Non-stationary"
    print(f"  ADF p-value: {adf_p_l:.4f} -> {level_label}")

    print("-- First Difference --")
    _, adf_p_d, _, *_ = adfuller(diff1, autolag="AIC")
    diff_stationary = adf_p_d < 0.05
    diff_label = "Stationary" if diff_stationary else "Non-stationary"
    print(f"  ADF p-value: {adf_p_d:.4f} -> {diff_label}")

    print("-- Conclusion --")
    if not level_stationary and diff_stationary:
        print("  Suggested integration order: d = 1")
    else:
        print("  Suggested integration order requires further review")


def plot_differenced_acf_pacf(series, name="series", lags=30):
    s = pd.Series(series.values if hasattr(series, "values") else series).dropna()
    diff1 = s.diff().dropna()
    max_lags = min(lags, len(diff1) // 2 - 1)
    if max_lags < 1:
        print(f"{name} (diff): series too short to plot ACF/PACF (n={len(diff1)})")
        return None
    try:
        fig = plot_differenced_acf_pacf_thesis(s, name=name, lags=lags)
        save_thesis_figure(fig, f"price_fig_diff_acf_pacf_{name.replace('-', '_')}")
        plt.show()
        return fig
    except Exception as exc:
        print(f"{name} (diff): ACF/PACF plot failed - {exc}")
        return None


def compare_price_diff_vs_log_return(series, name="series", lags=30):
    s = pd.Series(series.values if hasattr(series, "values") else series).dropna()
    diff1 = s.diff().dropna()

    if (s > 0).all():
        log_ret = np.log(s).diff().dropna()
    else:
        print(f"{name}: series contains non-positive values; log-return skipped")
        log_ret = None

    def _run_tests(ts, label):
        print(f"--- {label} (n={len(ts)}) ---")
        adf_stat, adf_p, adf_lags, *_ = adfuller(ts, autolag="AIC")
        adf_res = "Stationary" if adf_p < 0.05 else "Non-stationary"
        print(f"  ADF  : stat={adf_stat:.4f}  p={adf_p:.4f}  lags={adf_lags}  -> {adf_res}")
        try:
            kpss_stat, kpss_p, kpss_lags, _ = kpss(ts, regression="c", nlags="auto")
            kpss_res = "Stationary" if kpss_p > 0.05 else "Non-stationary"
            print(f"  KPSS : stat={kpss_stat:.4f}  p={kpss_p:.4f}  lags={kpss_lags}  -> {kpss_res}")
        except Exception as exc:
            print(f"  KPSS could not be computed: {exc}")

    print(f"=== compare_price_diff_vs_log_return: {name} ===")
    _run_tests(diff1, "First Difference")
    if log_ret is not None:
        _run_tests(log_ret, "Log Return")

    fig = compare_price_diff_vs_log_return_thesis(s, name=name, lags=lags)
    save_thesis_figure(fig, f"price_fig_diff_vs_logret_{name.replace('-', '_')}")
    plt.show()
    return fig


def arima_information_criteria_grid(series, name="series", max_p=3, max_q=3, d=0):
    s = pd.Series(series.values if hasattr(series, "values") else series).dropna()
    records = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = ARIMA(s, order=(p, d, q)).fit()
                records.append({"order": (p, d, q), "aic": res.aic, "bic": res.bic, "nobs": int(res.nobs)})
            except Exception:
                records.append({"order": (p, d, q), "aic": float("nan"), "bic": float("nan"), "nobs": None})
    df_ic = pd.DataFrame(records)
    df_ic_valid = df_ic.dropna(subset=["bic"])
    df_bic = df_ic_valid.sort_values("bic").reset_index(drop=True)
    df_aic = df_ic_valid.sort_values("aic").reset_index(drop=True)
    print(f"=== IC Grid: {name} (d={d}) ===")
    print("Top 10 by BIC:")
    print(df_bic.head(10).to_string(index=False))
    print("Top 10 by AIC:")
    print(df_aic.head(10).to_string(index=False))
    return df_bic


def compare_arima_ic_for_diff_and_logret(series, name="series", max_p=3, max_q=3):
    s = pd.Series(series.values if hasattr(series, "values") else series).dropna()
    diff1 = s.diff().dropna()

    ic_diff1 = arima_information_criteria_grid(diff1, name=name + " (diff1)", max_p=max_p, max_q=max_q, d=0)

    ic_logret = None
    if (s > 0).all():
        log_ret = np.log(s).diff().dropna()
        ic_logret = arima_information_criteria_grid(log_ret, name=name + " (log_ret)", max_p=max_p, max_q=max_q, d=0)
    else:
        print(f"{name}: non-positive values found; log-return IC grid skipped")

    print("=== Conclusion ===")
    best_diff1 = ic_diff1.iloc[0]["order"] if ic_diff1 is not None and len(ic_diff1) > 0 else None
    best_logret = ic_logret.iloc[0]["order"] if ic_logret is not None and len(ic_logret) > 0 else None
    print(f"  Best order by BIC for diff1   : {best_diff1}")
    print(f"  Best order by BIC for log_ret : {best_logret}")
    return ic_diff1, ic_logret


def fit_and_diagnose_arima_candidates(series, name="series", candidate_orders=None, ljung_box_lags=(10, 20, 30)):
    if candidate_orders is None:
        candidate_orders = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (2, 0, 0)]
    s = pd.Series(series.values if hasattr(series, "values") else series).dropna()
    records = []
    for order in candidate_orders:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = ARIMA(s, order=order).fit()
            resid = pd.Series(res.resid).dropna()
            row = {
                "order": order,
                "aic": round(res.aic, 4),
                "bic": round(res.bic, 4),
                "resid_mean": round(float(resid.mean()), 6),
                "resid_std": round(float(resid.std()), 6),
            }
            lb = acorr_ljungbox(resid, lags=list(ljung_box_lags), return_df=True)
            for lag in ljung_box_lags:
                pval = lb.loc[lag, "lb_pvalue"] if lag in lb.index else float("nan")
                row[f"lb_p_lag{lag}"] = round(float(pval), 4)
            try:
                arch_stat, arch_p, _, _ = het_arch(resid.values, nlags=10)
                row["arch_lm_stat"] = round(float(arch_stat), 4)
                row["arch_lm_pvalue"] = round(float(arch_p), 4)
            except Exception:
                row["arch_lm_stat"] = float("nan")
                row["arch_lm_pvalue"] = float("nan")
            records.append(row)
        except Exception as exc:
            print(f"  ARIMA{order} failed: {exc}")

    if not records:
        print(f"{name}: no models fitted successfully")
        return pd.DataFrame()

    df_diag = pd.DataFrame(records).sort_values("bic").reset_index(drop=True)
    print(f"=== ARIMA candidate diagnostics: {name} ===")
    print(df_diag.to_string(index=False))

    lb_cols = [f"lb_p_lag{lag}" for lag in ljung_box_lags]
    print("--- Assessment ---")
    for idx, row in df_diag.iterrows():
        order = row["order"]
        lb_pvals = [row[col] for col in lb_cols if col in row and not pd.isna(row[col])]
        no_autocorr = all(p > 0.05 for p in lb_pvals) if lb_pvals else False
        arch_p_val = row.get("arch_lm_pvalue", float("nan"))
        has_arch = pd.notna(arch_p_val) and arch_p_val < 0.05
        tags = []
        if idx == 0:
            tags.append("most parsimonious by BIC")
        if no_autocorr:
            tags.append("no residual autocorrelation (Ljung-Box p > 0.05)")
        if has_arch:
            tags.append(f"ARCH effects detected (arch_lm_p={arch_p_val:.4f}) — consider GARCH")
        elif pd.notna(arch_p_val):
            tags.append(f"no ARCH effects (arch_lm_p={arch_p_val:.4f})")
        if tags:
            print(f"  ARIMA{order}: " + ", ".join(tags))

    return df_diag


def walk_forward_arima_price_compare(series, name="series", train_frac=0.8, candidate_orders=None):
    """Walk-forward ARIMA diretamente em preços (d=1)."""
    s = pd.Series(series).dropna()
    if not (s > 0).all():
        print(f"{name}: série contém valores não positivos")
        return pd.DataFrame(columns=["order", "rmse", "mae", "mape", "directional_accuracy", "n_test"])

    if candidate_orders is None:
        candidate_orders = [(0, 1, 0), (1, 1, 0), (1, 1, 1), (1, 1, 2), (2, 1, 2)]

    split = int(len(s) * train_frac)
    train = s.iloc[:split]
    test = s.iloc[split:]
    print(f"{name}: n={len(s)}  train={len(train)}  test={len(test)}")

    records = []
    for order in candidate_orders:
        history = list(train)
        y_pred_list = []
        y_true_list = []
        y_lag_list = []
        failed = False

        for step_idx in range(len(test)):
            actual = float(test.iloc[step_idx])
            lag = float(history[-1])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fitted = ARIMA(history, order=order).fit()
                pred = float(np.asarray(fitted.forecast(steps=1))[0])
            except Exception as exc:
                print(f"ARIMA{order} falhou no step {step_idx}: {exc}")
                failed = True
                break

            y_pred_list.append(pred)
            y_true_list.append(actual)
            y_lag_list.append(lag)
            history.append(actual)

        if failed or not y_pred_list:
            continue

        y_pred = np.asarray(y_pred_list, float)
        y_true = np.asarray(y_true_list, float)
        y_lag = np.asarray(y_lag_list, float)
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        mae = float(np.mean(np.abs(y_pred - y_true)))
        mask = y_true != 0
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("nan")
        da = float(np.mean(np.sign(y_pred - y_lag) == np.sign(y_true - y_lag)))

        records.append(
            {
                "order": order,
                "rmse": round(rmse, 6),
                "mae": round(mae, 6),
                "mape": round(mape, 4),
                "directional_accuracy": round(da, 4),
                "n_test": len(y_true),
            }
        )

    cols = ["order", "rmse", "mae", "mape", "directional_accuracy", "n_test"]
    if not records:
        print(f"{name}: nenhum modelo produziu resultados")
        return pd.DataFrame(columns=cols)

    df_wf = pd.DataFrame(records, columns=cols).sort_values("rmse").reset_index(drop=True)
    print(f"=== Walk-forward price (d=1): {name} ===")
    print(df_wf.to_string(index=False))
    return df_wf


def _coerce_order_tuple(order_value):
    if isinstance(order_value, np.ndarray):
        order_value = order_value.tolist()
    if isinstance(order_value, list):
        order_value = tuple(order_value)
    if isinstance(order_value, tuple) and len(order_value) == 3:
        return tuple(int(v) for v in order_value)
    raise ValueError(f"Invalid order value: {order_value!r}")


def _map_diff_order_to_price_order(order_value):
    order = _coerce_order_tuple(order_value)
    p, d, q = order
    if d == 0:
        return (p, 1, q)
    return order


def build_arima_selection_summary(ic_table, diag_table, wf_table, name="series"):
    ic = ic_table.copy() if ic_table is not None else pd.DataFrame()
    diag = diag_table.copy() if diag_table is not None else pd.DataFrame()
    wf = wf_table.copy() if wf_table is not None else pd.DataFrame()

    if "order" not in wf.columns:
        raise ValueError("wf_table must contain the column 'order'")
    if not ic.empty and "order" not in ic.columns:
        raise ValueError("ic_table must contain the column 'order'")
    if not diag.empty and "order" not in diag.columns:
        raise ValueError("diag_table must contain the column 'order'")

    wf = wf.copy()
    wf["price_order"] = wf["order"].map(_coerce_order_tuple)
    wf_cols = ["price_order"] + [c for c in ["rmse", "mae", "mape", "directional_accuracy", "n_test"] if c in wf.columns]
    summary = wf[wf_cols].drop_duplicates(subset=["price_order"]).copy()

    if not ic.empty:
        ic = ic.copy()
        ic["price_order"] = ic["order"].map(_coerce_order_tuple)
        ic_cols = ["price_order"] + [c for c in ["aic", "bic", "nobs"] if c in ic.columns]
        summary = summary.merge(ic[ic_cols].drop_duplicates(subset=["price_order"]), on="price_order", how="left")

    if not diag.empty:
        diag = diag.copy()
        diag["diff_order"] = diag["order"].map(_coerce_order_tuple)
        # The notebook fits residual diagnostics on diff(1), which corresponds to
        # the same p and q on the price model with d=1. We keep both columns
        # explicit instead of merging diff(1) orders directly with price orders.
        diag["price_order"] = diag["diff_order"].map(_map_diff_order_to_price_order)
        diag_cols = ["price_order", "diff_order"] + [c for c in diag.columns if c not in {"order", "diff_order", "price_order", "aic", "bic", "nobs"}]
        summary = summary.merge(diag[diag_cols].drop_duplicates(subset=["price_order"]), on="price_order", how="left")
    else:
        summary["diff_order"] = pd.NA

    for col in ["aic", "bic", "nobs"]:
        if col not in summary.columns:
            summary[col] = np.nan

    summary["rmse_rank"] = summary["rmse"].rank(method="min", ascending=True)
    summary["mae_rank"] = summary["mae"].rank(method="min", ascending=True)
    summary["diracc_rank"] = summary["directional_accuracy"].rank(method="min", ascending=False)
    summary["bic_rank"] = summary["bic"].rank(method="min", ascending=True)

    lb_cols = sorted(c for c in summary.columns if c.startswith("lb_p_lag"))
    if lb_cols:
        summary["residual_ok"] = (summary[lb_cols] > 0.05).sum(axis=1) >= 2
    else:
        summary["residual_ok"] = False

    baseline_mask = summary["price_order"] == (0, 1, 0)
    baseline_rmse = summary.loc[baseline_mask, "rmse"].dropna()
    if len(baseline_rmse) > 0 and float(baseline_rmse.iloc[0]) != 0.0:
        baseline_value = float(baseline_rmse.iloc[0])
        summary["rmse_gain_vs_010"] = 100.0 * (baseline_value - summary["rmse"]) / baseline_value
    else:
        summary["rmse_gain_vs_010"] = np.nan

    rank_cols = ["rmse_rank", "mae_rank", "diracc_rank", "bic_rank"]
    for col in rank_cols:
        summary[col] = summary[col].astype("Int64")

    ordered_cols = [
        "price_order",
        "diff_order",
        "aic",
        "bic",
        "nobs",
        "resid_mean",
        "resid_std",
        *lb_cols,
        "rmse",
        "mae",
        "directional_accuracy",
        "n_test",
        "rmse_rank",
        "mae_rank",
        "diracc_rank",
        "bic_rank",
        "residual_ok",
        "rmse_gain_vs_010",
    ]
    ordered_cols = [c for c in ordered_cols if c in summary.columns]
    extra_cols = [c for c in summary.columns if c not in ordered_cols]
    summary = summary[ordered_cols + extra_cols]
    summary = summary.sort_values(["rmse_rank", "bic_rank"], ascending=[True, True], na_position="last").reset_index(drop=True)

    print(f"=== ARIMA selection summary: {name} ===")
    print(summary.to_string(index=False))

    if summary.empty:
        print("Best order by RMSE: unavailable")
        print("Best order by BIC: unavailable")
        print("Conclusion: no materially relevant out-of-sample gain over the parsimonious benchmark")
        return summary

    best_rmse_row = summary.sort_values(["rmse_rank", "bic_rank"], na_position="last").iloc[0]
    print(f"Best order by RMSE: ARIMA{best_rmse_row['price_order']}")

    bic_valid = summary.dropna(subset=["bic"])
    bic_prefers_010 = False
    if not bic_valid.empty:
        best_bic_row = bic_valid.sort_values(["bic_rank", "rmse_rank"], na_position="last").iloc[0]
        bic_prefers_010 = best_bic_row["price_order"] == (0, 1, 0)
        print(f"Best order by BIC: ARIMA{best_bic_row['price_order']}")
    else:
        print("Best order by BIC: unavailable")

    best_gain = best_rmse_row.get("rmse_gain_vs_010", np.nan)
    if pd.notna(best_gain) and best_gain < 0.5:
        print("Conclusion: no materially relevant out-of-sample gain over the parsimonious benchmark")
        if bic_prefers_010:
            print("Preferred academic conclusion: weak linear predictability in the conditional mean")
    else:
        tech_candidates = summary[(summary["rmse_gain_vs_010"] >= 0.5) & (summary["residual_ok"])].copy()
        if not tech_candidates.empty:
            tech_row = tech_candidates.sort_values(["rmse_rank", "bic_rank"], na_position="last").iloc[0]
            print(f"Preferred technical benchmark: ARIMA{tech_row['price_order']}")

    return summary


def run_multiasset_arima_diagnostics(asset_map, train_frac=0.8, paper_orders=ARIMA_DIAGNOSTIC_ORDERS):
    candidate_orders = [(0, 1, 0), (1, 1, 0), (1, 1, 1), (1, 1, 2), (2, 1, 2)]
    summary_records = []
    details = {}

    def _safe_pvalues(ts, asset, stage):
        ts = pd.Series(ts).dropna()
        stats = {"adf_pvalue": float("nan"), "kpss_pvalue": float("nan")}
        if ts.empty:
            return stats
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stats["adf_pvalue"] = float(adfuller(ts, autolag="AIC")[1])
        except Exception as exc:
            print(f"{asset} | {stage} ADF failed: {exc}")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stats["kpss_pvalue"] = float(kpss(ts, regression="c", nlags="auto")[1])
        except Exception as exc:
            print(f"{asset} | {stage} KPSS failed: {exc}")
        return stats

    for asset, series in asset_map.items():
        print("=" * 80)
        print(f"ARIMA price diagnostics: {asset}")

        try:
            s = pd.Series(series).dropna()
        except Exception as exc:
            print(f"{asset} | preprocessing failed: {exc}")
            continue

        if s.empty or not (s > 0).all():
            print(f"{asset} | série inválida (vazia ou valores <= 0)")
            continue

        diff1 = s.diff().dropna()

        try:
            stationarity_report(s, name=asset)
            plot_series_and_differences(s, name=asset)
            plot_acf_pacf_diagnostics(s, name=asset, lags=30)
            compare_integration_orders(s, name=asset)
            plot_differenced_acf_pacf(s, name=asset, lags=30)
        except Exception as exc:
            print(f"{asset} | diagnostics failed: {exc}")
            continue

        level_stats = _safe_pvalues(s, asset, "level")
        diff1_stats = _safe_pvalues(diff1, asset, "diff1")

        adf_level = level_stats.get("adf_pvalue", float("nan"))
        adf_diff1 = diff1_stats.get("adf_pvalue", float("nan"))
        if not np.isnan(adf_level) and not np.isnan(adf_diff1):
            if adf_level >= 0.05 and adf_diff1 < 0.05:
                print(f"{asset}: nivel ADF p={adf_level:.4f} (I(1) confirmado) -- I(0) confirmado em diff1 (d=1 justificado).")
            else:
                print(f"[AVISO] {asset}: nivel ADF p={adf_level:.4f}, diff1 ADF p={adf_diff1:.4f} -- verificar ordem.")

        try:
            ic_price = arima_information_criteria_grid(s, name=f"{asset} (price)", max_p=3, max_q=3, d=1)
        except Exception as exc:
            print(f"{asset} | ic_grid failed: {exc}")
            continue

        try:
            diag_price = fit_and_diagnose_arima_candidates(
                diff1,
                name=f"{asset} diff1",
                candidate_orders=[(p, 0, q) for (p, _, q) in candidate_orders],
            )
        except Exception as exc:
            print(f"{asset} | candidate_diagnostics failed: {exc}")
            continue

        try:
            wf_price = walk_forward_arima_price_compare(s, name=asset, train_frac=train_frac, candidate_orders=candidate_orders)
        except Exception as exc:
            print(f"{asset} | walk_forward failed: {exc}")
            continue

        if wf_price is None or wf_price.empty:
            print(f"{asset} | walk_forward: nenhum modelo completado")
            continue

        try:
            selection_summary = build_arima_selection_summary(
                ic_price,
                diag_price,
                wf_price.rename(columns={"mape": "mape_wf", "directional_accuracy": "directional_accuracy"}),
                name=f"{asset} price (d=1)",
            )
        except Exception as exc:
            selection_summary = wf_price.copy()
            print(f"{asset} | selection_summary fallback: {exc}")

        paper_order = paper_orders.get(asset, (2, 1, 2))
        best_rmse_row = wf_price.sort_values("rmse").iloc[0] if not wf_price.empty else {}
        best_rmse_order = tuple(best_rmse_row.get("order", (0, 1, 0))) if hasattr(best_rmse_row, "get") else (0, 1, 0)

        paper_row = wf_price[wf_price["order"] == paper_order]
        paper_rmse = float(paper_row["rmse"].iloc[0]) if not paper_row.empty else float("nan")
        benchmark_row = wf_price[wf_price["order"] == (0, 1, 0)]
        benchmark_rmse = float(benchmark_row["rmse"].iloc[0]) if not benchmark_row.empty else float("nan")

        summary_records.append(
            {
                "asset": asset,
                "n_obs": int(len(s)),
                "level_adf_pvalue": level_stats["adf_pvalue"],
                "level_kpss_pvalue": level_stats["kpss_pvalue"],
                "diff1_adf_pvalue": diff1_stats["adf_pvalue"],
                "diff1_kpss_pvalue": diff1_stats["kpss_pvalue"],
                "paper_order": paper_order,
                "paper_rmse": paper_rmse,
                "best_rmse_order_price": best_rmse_order,
                "best_rmse_price": float(best_rmse_row.get("rmse", float("nan"))) if hasattr(best_rmse_row, "get") else float(best_rmse_row["rmse"]),
                "benchmark_rmse_010": benchmark_rmse,
            }
        )

        details[asset] = {
            "series": s,
            "diff1": diff1,
            "level_stats": level_stats,
            "diff1_stats": diff1_stats,
            "ic_price": ic_price,
            "diag_price": diag_price,
            "wf_price": wf_price,
            "selection_summary": selection_summary,
        }

    multiasset_arima_summary = pd.DataFrame(summary_records)
    if multiasset_arima_summary.empty:
        print("Nenhum resultado ARIMA multiativo produzido")
        return multiasset_arima_summary, details

    for col in [
        "level_adf_pvalue",
        "level_kpss_pvalue",
        "diff1_adf_pvalue",
        "diff1_kpss_pvalue",
        "paper_rmse",
        "best_rmse_price",
        "benchmark_rmse_010",
    ]:
        if col in multiasset_arima_summary.columns:
            multiasset_arima_summary[col] = multiasset_arima_summary[col].astype(float).round(6)

    multiasset_arima_summary = multiasset_arima_summary.sort_values("asset").reset_index(drop=True)
    print("=== Multiasset ARIMA price summary ===")
    print(multiasset_arima_summary.to_string(index=False))
    return multiasset_arima_summary, details
