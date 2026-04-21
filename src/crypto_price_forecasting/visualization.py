from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoLocator, MaxNLocator, PercentFormatter

from .config import (
    RESULTS_FIGURES_DIR,
    THESIS_ASSET_ORDER,
    THESIS_METRIC_COLORS,
    THESIS_MODEL_COLORS,
    THESIS_MODEL_ORDER,
)

try:
    import seaborn as sns
except Exception:  # pragma: no cover - opcional
    sns = None


def set_thesis_plot_style():
    palette_cycle = [
        THESIS_MODEL_COLORS["ARIMA"],
        THESIS_MODEL_COLORS["ETS"],
        THESIS_MODEL_COLORS["LSTM"],
        THESIS_MODEL_COLORS["RNN"],
        THESIS_MODEL_COLORS["GRU"],
        THESIS_MODEL_COLORS["NAIVE_ZERO"],
        "#8c8c8c",
        "#b07d2c",
    ]
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 320,
            "figure.figsize": (7.0, 4.4),
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#7f8c8d",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#d9dde3",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "legend.frameon": False,
            "axes.prop_cycle": plt.cycler(color=palette_cycle),
        }
    )
    return THESIS_MODEL_COLORS.copy()


def finalize_thesis_axis(ax, title=None, xlabel=None, ylabel=None, zero_line=False, legend=True):
    if title:
        ax.set_title(title, loc="left", pad=8, fontweight="semibold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if zero_line:
        ax.axhline(0.0, color="#7f8c8d", linestyle="--", linewidth=0.9, alpha=0.85, zorder=0)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.spines["left"].set_color("#7f8c8d")
    ax.spines["bottom"].set_color("#7f8c8d")
    handles, labels = ax.get_legend_handles_labels()
    if legend and handles:
        if len(handles) > 4 or max(len(str(label)) for label in labels) > 12:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1)
        else:
            ax.legend(loc="best")
    elif ax.get_legend() is not None:
        ax.get_legend().remove()
    return ax


def save_thesis_figure(fig, name, folder=RESULTS_FIGURES_DIR, formats=("png",)):
    if fig is None:
        raise ValueError("Figure is None and cannot be saved.")
    output_dir = Path(folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = str(name).strip().replace(" ", "_")
    saved_paths = {}
    for fmt in formats:
        fmt_clean = str(fmt).lower().strip(".")
        output_path = output_dir / f"{safe_name}.{fmt_clean}"
        fig.savefig(output_path, bbox_inches="tight")
        saved_paths[fmt_clean] = str(output_path)
    return saved_paths


def _thesis_model_sequence(models):
    seen = [str(model) for model in models]
    ordered = [model for model in THESIS_MODEL_ORDER if model in seen]
    ordered.extend(sorted(set(seen).difference(ordered)))
    return ordered


def _thesis_model_color(model_name):
    return THESIS_MODEL_COLORS.get(str(model_name), "#5f6b7a")


def _format_display_number(value, decimals=4):
    return "NA" if pd.isna(value) else f"{float(value):.{decimals}f}"


def _format_display_percent(value, decimals=2):
    return "NA" if pd.isna(value) else f"{100.0 * float(value):.{decimals}f}%"


def _get_thesis_correlation_values(series, nlags, method="acf"):
    from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf

    clean_series = pd.Series(series).dropna().astype(float)
    if method == "acf":
        return sm_acf(clean_series, nlags=nlags, fft=True)
    return sm_pacf(clean_series, nlags=nlags, method="ywm")


def _plot_thesis_correlation_function(ax, values, n_obs, title, ylabel):
    coeffs = np.asarray(values[1:], dtype=float)
    lags = np.arange(1, coeffs.shape[0] + 1)
    conf = 1.96 / np.sqrt(max(int(n_obs), 1))
    ax.axhspan(-conf, conf, color="#d9dde3", alpha=0.45, zorder=0)
    if coeffs.size:
        ax.vlines(lags, 0.0, coeffs, color="#1f4e79", linewidth=1.35)
        ax.scatter(lags, coeffs, color="#1f4e79", s=18, zorder=3)
        local_max = float(np.nanmax(np.abs(coeffs)))
    else:
        local_max = 0.0
    ylim = max(0.08, conf * 2.4, local_max * 1.3)
    ylim = min(ylim, 0.65)
    ax.set_xlim(0.5, max(len(lags), 1) + 0.5)
    ax.set_ylim(-ylim, ylim)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=min(8, max(len(lags), 2))))
    ax.yaxis.set_major_locator(AutoLocator())
    finalize_thesis_axis(ax, title=title, xlabel="Lag", ylabel=ylabel, zero_line=True, legend=False)
    return ax


def plot_series_and_differences_thesis(series, name="series"):
    set_thesis_plot_style()
    s = pd.Series(series).dropna().reset_index(drop=True)
    diff1 = s.diff().dropna()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0))
    axes[0].plot(s.index, s.values, color="#1f4e79")
    finalize_thesis_axis(axes[0], title=f"{name}: nível", xlabel="Observação", ylabel="Preço", legend=False)
    if diff1.empty:
        axes[1].text(0.5, 0.5, "Série curta", transform=axes[1].transAxes, ha="center", va="center")
        finalize_thesis_axis(
            axes[1],
            title="Primeira diferença",
            xlabel="Observação",
            ylabel="Delta do preço",
            legend=False,
        )
    else:
        axes[1].plot(diff1.index, diff1.values, color="#9c4f2f")
        finalize_thesis_axis(
            axes[1],
            title="Primeira diferença",
            xlabel="Observação",
            ylabel="Delta do preço",
            zero_line=True,
            legend=False,
        )
    fig.tight_layout()
    return fig


def plot_acf_pacf_diagnostics_thesis(series, name="series", lags=30):
    set_thesis_plot_style()
    s = pd.Series(series).dropna()
    max_lags = min(lags, len(s) // 2 - 1)
    if max_lags < 1:
        raise ValueError(f"{name}: series too short to plot ACF/PACF (n={len(s)})")
    acf_vals = _get_thesis_correlation_values(s, nlags=max_lags, method="acf")
    pacf_vals = _get_thesis_correlation_values(s, nlags=max_lags, method="pacf")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0))
    _plot_thesis_correlation_function(axes[0], acf_vals, len(s), f"ACF: {name}", "Autocorrelação")
    _plot_thesis_correlation_function(axes[1], pacf_vals, len(s), f"PACF: {name}", "Autocorrelação parcial")
    fig.tight_layout()
    return fig


def plot_differenced_acf_pacf_thesis(series, name="series", lags=30):
    set_thesis_plot_style()
    diff1 = pd.Series(series).dropna().diff().dropna()
    max_lags = min(lags, len(diff1) // 2 - 1)
    if max_lags < 1:
        raise ValueError(f"{name}: differenced series too short to plot ACF/PACF (n={len(diff1)})")
    acf_vals = _get_thesis_correlation_values(diff1, nlags=max_lags, method="acf")
    pacf_vals = _get_thesis_correlation_values(diff1, nlags=max_lags, method="pacf")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.0))
    _plot_thesis_correlation_function(axes[0], acf_vals, len(diff1), f"ACF: {name} diff(1)", "Autocorrelação")
    _plot_thesis_correlation_function(
        axes[1],
        pacf_vals,
        len(diff1),
        f"PACF: {name} diff(1)",
        "Autocorrelação parcial",
    )
    fig.tight_layout()
    return fig


def compare_price_diff_vs_log_return_thesis(series, name="series", lags=30):
    set_thesis_plot_style()
    s = pd.Series(series).dropna()
    diff1 = s.diff().dropna()
    log_ret = np.log(s).diff().dropna() if (s > 0).all() else None
    n_rows = 2 if log_ret is not None else 1
    fig, axes = plt.subplots(n_rows, 3, figsize=(14.4, 4.2 * n_rows))
    axes = np.atleast_2d(axes)

    axes[0, 0].plot(diff1.index, diff1.values, color="#9c4f2f")
    finalize_thesis_axis(
        axes[0, 0],
        title=f"{name}: diff(1)",
        xlabel="Observação",
        ylabel="Delta do preço",
        zero_line=True,
        legend=False,
    )
    max_lags_d = min(lags, len(diff1) // 2 - 1)
    if max_lags_d >= 1:
        acf_diff = _get_thesis_correlation_values(diff1, nlags=max_lags_d, method="acf")
        pacf_diff = _get_thesis_correlation_values(diff1, nlags=max_lags_d, method="pacf")
        _plot_thesis_correlation_function(axes[0, 1], acf_diff, len(diff1), f"ACF: {name} diff(1)", "Autocorrelação")
        _plot_thesis_correlation_function(
            axes[0, 2],
            pacf_diff,
            len(diff1),
            f"PACF: {name} diff(1)",
            "Autocorrelação parcial",
        )
    else:
        axes[0, 1].text(0.5, 0.5, "Série curta", transform=axes[0, 1].transAxes, ha="center", va="center")
        axes[0, 2].text(0.5, 0.5, "Série curta", transform=axes[0, 2].transAxes, ha="center", va="center")
        finalize_thesis_axis(axes[0, 1], title="ACF: diff(1)", xlabel="Lag", ylabel="Autocorrelação", zero_line=True, legend=False)
        finalize_thesis_axis(
            axes[0, 2],
            title="PACF: diff(1)",
            xlabel="Lag",
            ylabel="Autocorrelação parcial",
            zero_line=True,
            legend=False,
        )

    if log_ret is not None:
        axes[1, 0].plot(log_ret.index, log_ret.values, color="#2b6f89")
        finalize_thesis_axis(
            axes[1, 0],
            title=f"{name}: log-ret",
            xlabel="Observação",
            ylabel="Log-retorno",
            zero_line=True,
            legend=False,
        )
        max_lags_l = min(lags, len(log_ret) // 2 - 1)
        if max_lags_l >= 1:
            acf_log = _get_thesis_correlation_values(log_ret, nlags=max_lags_l, method="acf")
            pacf_log = _get_thesis_correlation_values(log_ret, nlags=max_lags_l, method="pacf")
            _plot_thesis_correlation_function(axes[1, 1], acf_log, len(log_ret), f"ACF: {name} log-ret", "Autocorrelação")
            _plot_thesis_correlation_function(
                axes[1, 2],
                pacf_log,
                len(log_ret),
                f"PACF: {name} log-ret",
                "Autocorrelação parcial",
            )
        else:
            axes[1, 1].text(0.5, 0.5, "Série curta", transform=axes[1, 1].transAxes, ha="center", va="center")
            axes[1, 2].text(0.5, 0.5, "Série curta", transform=axes[1, 2].transAxes, ha="center", va="center")
            finalize_thesis_axis(axes[1, 1], title="ACF: log-ret", xlabel="Lag", ylabel="Autocorrelação", zero_line=True, legend=False)
            finalize_thesis_axis(
                axes[1, 2],
                title="PACF: log-ret",
                xlabel="Lag",
                ylabel="Autocorrelação parcial",
                zero_line=True,
                legend=False,
            )
    fig.tight_layout()
    return fig


def plot_clean_correlation_heatmap(corr_matrix, title="Correlação estática sincronizada", annot=None):
    set_thesis_plot_style()
    corr = corr_matrix.copy() if isinstance(corr_matrix, pd.DataFrame) else pd.DataFrame(corr_matrix)
    if corr.empty:
        raise ValueError("corr_matrix is empty; heatmap cannot be created.")
    annotate = corr.shape[0] <= 12 if annot is None else bool(annot)
    size = max(6.0, min(12.0, 0.55 * corr.shape[0] + 2.5))
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    if sns is not None:
        cmap = sns.diverging_palette(240, 15, s=80, l=40, as_cmap=True)
        sns.heatmap(
            corr,
            ax=ax,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            annot=annotate,
            fmt=".2f" if annotate else "",
            linewidths=0.4,
            linecolor="#f0f0f0",
            cbar_kws={"label": "Correlação"},
        )
    else:
        image = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("Correlação")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    finalize_thesis_axis(ax, title=title, xlabel="Ativo", ylabel="Ativo", legend=False)
    ax.grid(False)
    fig.tight_layout()
    return fig


def select_informative_rolling_assets(
    rolling_corr_dict,
    window=90,
    preferred_assets=("ETH", "Gold", "S&P500", "DXY", "CBOE"),
    max_assets=5,
):
    if window not in rolling_corr_dict:
        raise ValueError(f"Window {window} not found in rolling_corr_dict.")
    rolling_df = rolling_corr_dict[window]
    if rolling_df.empty or rolling_df.shape[1] == 0:
        raise ValueError(f"No rolling-correlation data available for window {window}.")
    selected = [asset for asset in preferred_assets if asset in rolling_df.columns]
    strength_order = rolling_df.abs().mean().sort_values(ascending=False).index.tolist()
    for asset in strength_order:
        if asset not in selected:
            selected.append(asset)
        if len(selected) >= max_assets:
            break
    return selected[:max_assets]


def plot_clean_rolling_correlations_thesis(
    rolling_corr_dict,
    target_asset="BTC",
    selected_assets=None,
    window=90,
    show=False,
):
    set_thesis_plot_style()
    if window not in rolling_corr_dict:
        raise ValueError(f"Window {window} not found in rolling_corr_dict.")
    rolling_df = rolling_corr_dict[window]
    if rolling_df.empty or rolling_df.shape[1] == 0:
        raise ValueError(f"No comparable assets available to plot for window {window}.")
    if selected_assets is None:
        selected_assets = select_informative_rolling_assets(rolling_corr_dict, window=window)
    plot_assets = [asset for asset in selected_assets if asset in rolling_df.columns]
    if not plot_assets:
        plot_assets = list(rolling_df.columns[: min(5, len(rolling_df.columns))])
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    for asset in plot_assets:
        ax.plot(rolling_df.index, rolling_df[asset], linewidth=1.8, label=asset)
    if isinstance(rolling_df.index, pd.DatetimeIndex):
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        fig.autofmt_xdate()
    finalize_thesis_axis(
        ax,
        title=f"Rolling correlation com {target_asset} ({window} dias)",
        xlabel="Data",
        ylabel="Correlação",
        zero_line=True,
        legend=True,
    )
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def build_metric_facet_figure(source_table, metric_col, title, ylabel, percent=False, reference_line=None):
    set_thesis_plot_style()
    if not isinstance(source_table, pd.DataFrame) or source_table.empty:
        raise ValueError(f"Source table for {metric_col} is unavailable or empty.")
    work = source_table.copy()
    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    work["Horizon"] = pd.to_numeric(work["Horizon"], errors="coerce")
    work = work.dropna(subset=["Asset", "Horizon", "Model", metric_col])
    if work.empty:
        raise ValueError(f"No valid rows available to plot {metric_col}.")
    asset_priority = tuple(asset for asset in THESIS_ASSET_ORDER if asset in set(work["Asset"]))
    ordered_assets = [asset for asset in asset_priority if asset in set(work["Asset"])]
    ordered_assets.extend(sorted(set(work["Asset"]).difference(ordered_assets)))
    model_order = _thesis_model_sequence(work["Model"].astype(str).tolist())
    n_assets = len(ordered_assets)
    ncols = 2 if n_assets > 1 else 1
    nrows = int(np.ceil(n_assets / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.8 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, asset in zip(axes, ordered_assets):
        asset_df = work.loc[work["Asset"] == asset].copy()
        for model in model_order:
            model_df = asset_df.loc[asset_df["Model"].astype(str) == model].sort_values("Horizon")
            if model_df.empty:
                continue
            y_values = model_df[metric_col].to_numpy(dtype=float)
            if percent:
                y_values = y_values * 100.0
            ax.plot(
                model_df["Horizon"].to_numpy(dtype=float),
                y_values,
                marker="o",
                linewidth=1.9,
                color=_thesis_model_color(model),
                label=model,
            )
        if reference_line is not None:
            ax.axhline(reference_line, color="#7f8c8d", linestyle=":", linewidth=1.0, alpha=0.9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if percent:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        finalize_thesis_axis(ax, title=asset, xlabel="Horizonte", ylabel=ylabel, legend=False)
    for ax in axes[n_assets:]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels() if n_assets else ([], [])
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=min(len(labels), 3), frameon=False)
    fig.suptitle(title, x=0.01, ha="left", y=1.06, fontsize=14, fontweight="semibold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def build_model_win_count_figure(tbl_winners, title="Vitórias por modelo"):
    set_thesis_plot_style()
    metric_cols = {"RMSE": "best_rmse_model", "MAE": "best_mae_model", "Direction": "best_diracc_model"}
    rows = []
    for metric_label, metric_col in metric_cols.items():
        if metric_col not in tbl_winners.columns:
            continue
        counts = tbl_winners[metric_col].value_counts(dropna=True)
        for model, wins in counts.items():
            rows.append({"Metric": metric_label, "Model": str(model), "Wins": int(wins)})
    win_counts = pd.DataFrame(rows)
    if win_counts.empty:
        raise ValueError("No model-win information was available to plot.")
    model_order = _thesis_model_sequence(win_counts["Model"].tolist())
    metric_order = [metric for metric in ["RMSE", "MAE", "Direction"] if metric in set(win_counts["Metric"])]
    pivot = (
        win_counts.pivot_table(index="Model", columns="Metric", values="Wins", aggfunc="sum", fill_value=0)
        .reindex(index=model_order, columns=metric_order)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(pivot.index))
    width = 0.22 if len(metric_order) >= 3 else 0.28
    for idx, metric_label in enumerate(metric_order):
        offset = (idx - (len(metric_order) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            pivot[metric_label].to_numpy(dtype=float),
            width=width,
            color=THESIS_METRIC_COLORS.get(metric_label, "#5f6b7a"),
            label=metric_label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    finalize_thesis_axis(ax, title=title, xlabel="Modelo", ylabel="Contagem", legend=True)
    fig.tight_layout()
    return fig


def format_main_result_tables(table_map, asset_priority=THESIS_ASSET_ORDER):
    if not isinstance(table_map, dict):
        raise TypeError("table_map must be a dictionary of named DataFrames.")

    asset_priority = tuple(asset_priority)
    formatted_tables = {}

    def _asset_sort_key(series):
        categories = list(asset_priority) + [
            asset for asset in sorted(set(series.astype(str))) if asset not in asset_priority
        ]
        return pd.Categorical(series.astype(str), categories=categories, ordered=True)

    def _model_sort_key(series):
        model_order = _thesis_model_sequence(series.astype(str).tolist())
        return pd.Categorical(series.astype(str), categories=model_order, ordered=True)

    table_names = [
        "tbl_family_logret_all",
        "tbl_family_logret_winners",
        "tbl_family_logret_rank_summary",
        "tbl_dm_logret",
        "tbl_network_seed_summary",
    ]
    for table_name in table_names:
        table_obj = table_map.get(table_name)
        if not isinstance(table_obj, pd.DataFrame) or table_obj.empty:
            print(f"[{table_name}] is unavailable or empty; formatted copy not created.")
            continue
        work = table_obj.copy()
        if "Asset" in work.columns:
            work = work.assign(_asset_order=_asset_sort_key(work["Asset"]))
        if "Model" in work.columns:
            work = work.assign(_model_order=_model_sort_key(work["Model"]))
        sort_cols = [col for col in ["_asset_order", "Horizon", "_model_order"] if col in work.columns]
        if sort_cols:
            work = work.sort_values(sort_cols, na_position="last").reset_index(drop=True)
        formatted = work.drop(columns=[col for col in ["_asset_order", "_model_order"] if col in work.columns]).copy()
        if table_name == "tbl_family_logret_all":
            for col in ["RMSE", "MAE", "MAPE", "sMAPE", "TheilU1", "TheilU2"]:
                if col in formatted.columns:
                    formatted[col] = formatted[col].map(lambda v: _format_display_number(v, 6))
            if "DirectionalAccuracy" in formatted.columns:
                formatted["DirectionalAccuracy"] = formatted["DirectionalAccuracy"].map(lambda v: _format_display_percent(v, 2))
        elif table_name == "tbl_family_logret_winners":
            for col in ["best_rmse", "best_mae"]:
                if col in formatted.columns:
                    formatted[col] = formatted[col].map(lambda v: _format_display_number(v, 6))
            if "best_diracc" in formatted.columns:
                formatted["best_diracc"] = formatted["best_diracc"].map(lambda v: _format_display_percent(v, 2))
        elif table_name == "tbl_family_logret_rank_summary":
            for col in ["mean_rmse_rank", "mean_mae_rank", "mean_diracc_rank"]:
                if col in formatted.columns:
                    formatted[col] = formatted[col].map(lambda v: _format_display_number(v, 3))
        elif table_name == "tbl_dm_logret":
            if "DM_stat" in formatted.columns:
                formatted["DM_stat"] = formatted["DM_stat"].map(lambda v: _format_display_number(v, 6))
            if "p_value" in formatted.columns:
                formatted["p_value"] = formatted["p_value"].map(lambda v: _format_display_number(v, 6))
        elif table_name == "tbl_network_seed_summary":
            for source_col, target_col in [
                ("rmse_mean", "RMSE Mean"),
                ("rmse_std", "RMSE Std"),
                ("mae_mean", "MAE Mean"),
                ("mae_std", "MAE Std"),
                ("directional_accuracy_mean", "Directional Accuracy Mean (%)"),
                ("directional_accuracy_std", "Directional Accuracy Std (%)"),
            ]:
                if source_col not in formatted.columns:
                    continue
                if "accuracy" in source_col:
                    formatted[source_col] = formatted[source_col].map(lambda v: _format_display_percent(v, 2))
                else:
                    formatted[source_col] = formatted[source_col].map(lambda v: _format_display_number(v, 6))
                formatted = formatted.rename(columns={source_col: target_col})
        formatted_tables[table_name] = formatted
    return formatted_tables


def _plot_arima_walkforward(outs_arima, asset, horizons=(1, 7, 14)):
    """Figura de actual vs. ARIMA walk-forward para cada horizonte."""
    entry = outs_arima.get(asset)
    if entry is None:
        print(f"[walk-forward] sem dados para {asset}")
        return None
    y = entry["y"]
    split = entry["split"]
    test_index = y.index[split:]
    fig, axes = plt.subplots(len(horizons), 1, figsize=(12, 4 * len(horizons)), sharex=False)
    if len(horizons) == 1:
        axes = [axes]
    for ax, h in zip(axes, horizons):
        ax.plot(y.index, y.values, label="Atual", linewidth=0.8)
        yhat = entry["preds"].get(h)
        if yhat is not None:
            ax.plot(yhat.index, yhat.values, label=f"ARIMA (walk-forward) h={h}", linestyle=":", linewidth=1, color="orange")
        if len(test_index):
            ax.axvspan(test_index[0], y.index[-1], alpha=0.08, color="gray")
        ax.set_title(f"{asset} — Atual vs. ARIMA walk-forward (h={h})")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_arima_cones(outs_arima, asset, horizons=(1, 7, 14), conf=0.95, min_points=10):
    """Fan chart com cones de previsão empíricos."""
    entry = outs_arima.get(asset)
    if entry is None:
        print(f"[cones] sem dados para {asset}")
        return None
    y = entry["y"]
    split = entry["split"]
    test_idx = y.index[split:]
    y_test = y.loc[test_idx]
    z = 1.96 if abs(conf - 0.95) < 1e-3 else 1.6449 if abs(conf - 0.90) < 1e-3 else 2.5758
    colors_h = plt.cm.viridis([0.2, 0.55, 0.85])[: len(horizons)]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test.index, y_test.values, label="Observado", color="steelblue", linewidth=1.5)
    for h, col in zip(sorted(horizons, reverse=True), colors_h):
        yhat_h = entry["preds"].get(h)
        if yhat_h is None:
            continue
        yhat_test = yhat_h.loc[test_idx]
        err = (y_test - yhat_test).dropna()
        if len(err) < min_points:
            continue
        rmse_exp = (err**2).expanding(min_periods=min_points).mean().pow(0.5)
        yhat_v = yhat_test.loc[rmse_exp.index]
        ax.fill_between(rmse_exp.index, yhat_v - z * rmse_exp, yhat_v + z * rmse_exp, alpha=0.20, color=col, label=f"h={h}")
        ax.plot(yhat_test.index, yhat_test.values, linestyle=":", linewidth=0.8, color=col)
    ax.set_title(f"{asset} — Cones de previsão — período de teste")
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Valor")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _build_metric_heatmap_figure(tbl, metric, title=None, fmt=".3f", cmap="Greens"):
    if tbl is None or tbl.empty or metric not in tbl.columns:
        print(f"[heatmap] dados insuficientes para {metric}")
        return None
    model_order = [model for model in ["ARIMA", "ETS", "GRU", "LSTM", "RNN"] if model in tbl["Model"].astype(str).unique()]
    dfm = tbl.copy()
    dfm[metric] = pd.to_numeric(dfm[metric], errors="coerce")
    pivot = (
        dfm.pivot_table(values=metric, index="Horizon", columns="Model", aggfunc="mean")
        .reindex(columns=model_order)
        .sort_index()
    )
    if pivot.empty:
        print(f"[heatmap] pivot vazio para {metric}")
        return None
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    fig.colorbar(im, ax=ax, label=f"{metric} (média)")
    ax.set_title(title or f"Heatmap — {metric} por Modelo")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Horizonte")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not pd.isna(val):
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", color="orange", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig
