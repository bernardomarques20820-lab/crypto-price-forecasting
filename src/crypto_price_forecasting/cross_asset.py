from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CROSS_ASSET_CANDIDATES, CROSS_ASSET_END, CROSS_ASSET_START
from .data import _dl_one
from .visualization import plot_clean_correlation_heatmap, plot_clean_rolling_correlations_thesis, save_thesis_figure

try:
    from IPython.display import display
except Exception:  # pragma: no cover - opcional
    display = None


def prepare_single_asset_returns(series, asset_name):
    s = pd.Series(series).dropna()
    s = s.sort_index()
    s = s.pct_change().dropna()
    s.name = asset_name
    return s


def build_synchronized_return_panel(price_map, min_common_obs=200):
    return_series = []
    coverage_rows = []

    for asset, series in price_map.items():
        price_series = pd.Series(series).dropna().sort_index()
        returns_series = prepare_single_asset_returns(price_series, asset)

        coverage_rows.append({"asset": asset, "price_obs": int(price_series.shape[0]), "return_obs": int(returns_series.shape[0])})
        return_series.append(returns_series)

    coverage_table = pd.DataFrame(coverage_rows).sort_values("asset").reset_index(drop=True)

    returns_panel = pd.concat(return_series, axis=1, join="inner")
    returns_panel = returns_panel.sort_index().dropna()

    common_obs = int(returns_panel.shape[0])
    if common_obs < min_common_obs:
        raise ValueError(f"Insufficient synchronized observations after return alignment: {common_obs} < {min_common_obs}.")

    return returns_panel, coverage_table


def summarize_synchronization(returns_panel, coverage_table):
    common_obs = int(returns_panel.shape[0])
    start_date = returns_panel.index.min()
    end_date = returns_panel.index.max()
    n_assets = int(returns_panel.shape[1])

    print(f"Final synchronized return panel shape: {returns_panel.shape}")
    print(f"Common date range: {start_date} -> {end_date}")
    print(f"Number of assets: {n_assets}")
    if display is not None:
        display(coverage_table)

    sync_summary = pd.DataFrame([{"common_obs": common_obs, "start_date": start_date, "end_date": end_date}])
    if display is not None:
        display(sync_summary)
    return sync_summary


def download_corrected_cross_asset_price_map(candidates=CROSS_ASSET_CANDIDATES, start=CROSS_ASSET_START, end=CROSS_ASSET_END):
    price_map = {}
    btc_price = _dl_one("BTC-USD", start, end)
    if btc_price.empty:
        raise ValueError("Failed to download BTC-USD for the corrected cross-asset section.")
    price_map["BTC"] = btc_price.iloc[:, 0].rename("BTC")

    for asset, symbols in candidates.items():
        raw_price = _dl_one(symbols[0], start, end)
        if raw_price.empty:
            raise ValueError(f"Failed to download {asset} ({symbols[0]}) for the corrected cross-asset section.")
        price_map[asset] = raw_price.iloc[:, 0].rename(asset)

    return price_map


def compute_static_correlation_matrix(returns_panel):
    return returns_panel.corr()


def build_static_correlation_heatmap(returns_panel, title="Correlação estática sincronizada"):
    corr_matrix = compute_static_correlation_matrix(returns_panel)
    fig = plot_clean_correlation_heatmap(corr_matrix, title=title)
    return corr_matrix, fig


def compute_clean_rolling_correlations(returns_panel, target_asset="BTC", windows=(30, 90, 180)):
    if target_asset not in returns_panel.columns:
        raise ValueError(f"Target asset {target_asset!r} not found in returns_panel.")

    if not windows:
        raise ValueError("windows must contain at least one positive integer.")

    normalized_windows = []
    for window in windows:
        if not isinstance(window, int) or window <= 0:
            raise ValueError(f"Invalid rolling window: {window!r}. Expected positive integers.")
        normalized_windows.append(window)

    compare_assets = [asset for asset in returns_panel.columns if asset != target_asset]
    if not compare_assets:
        raise ValueError("No comparable assets available after excluding the target asset.")

    panel = returns_panel.sort_index()
    rolling_corr_clean = {}
    for window in normalized_windows:
        rolling_df = pd.DataFrame(index=panel.index)
        for asset in compare_assets:
            rolling_df[asset] = panel[target_asset].rolling(window).corr(panel[asset])
        rolling_corr_clean[window] = rolling_df

    return rolling_corr_clean


def plot_clean_rolling_correlations(rolling_corr_dict, target_asset="BTC", selected_assets=None, window=90, save=True):
    fig = plot_clean_rolling_correlations_thesis(
        rolling_corr_dict,
        target_asset=target_asset,
        selected_assets=selected_assets,
        window=window,
        show=False,
    )
    if save:
        save_thesis_figure(fig, f"price_fig_rolling_corr_{target_asset}_w{window}")
    return fig


def summarize_rolling_correlation_window(rolling_corr_dict, window=90):
    rolling_df = rolling_corr_dict.get(window)
    if rolling_df is None:
        raise ValueError(f"Window {window} not found in rolling_corr_dict.")

    summary_rows = []
    for asset in rolling_df.columns:
        series = rolling_df[asset].dropna()
        summary_rows.append(
            {
                "asset": asset,
                "mean_corr": float(series.mean()) if not series.empty else np.nan,
                "std_corr": float(series.std()) if not series.empty else np.nan,
                "min_corr": float(series.min()) if not series.empty else np.nan,
                "max_corr": float(series.max()) if not series.empty else np.nan,
                "last_corr": float(series.iloc[-1]) if not series.empty else np.nan,
            }
        )

    return pd.DataFrame(summary_rows).sort_values("mean_corr", ascending=False).reset_index(drop=True)
