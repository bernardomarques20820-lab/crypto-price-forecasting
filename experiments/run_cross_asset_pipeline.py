from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_price_forecasting.config import CROSS_ASSET_END, CROSS_ASSET_START, RESULTS_TABLES_DIR
from crypto_price_forecasting.cross_asset import (
    build_static_correlation_heatmap,
    build_synchronized_return_panel,
    compute_clean_rolling_correlations,
    download_corrected_cross_asset_price_map,
    plot_clean_rolling_correlations,
    summarize_rolling_correlation_window,
    summarize_synchronization,
)
from crypto_price_forecasting.visualization import save_thesis_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Executa a analise cross-asset corrigida.")
    parser.add_argument("--start", default=CROSS_ASSET_START)
    parser.add_argument("--end", default=CROSS_ASSET_END)
    parser.add_argument("--rolling-window", type=int, default=90)
    return parser.parse_args()


def main():
    args = parse_args()

    print("[1/4] Baixando painel cross-asset corrigido...")
    price_map = download_corrected_cross_asset_price_map(start=args.start, end=args.end)

    print("[2/4] Sincronizando retornos...")
    returns_panel, coverage_table = build_synchronized_return_panel(price_map)
    sync_summary = summarize_synchronization(returns_panel, coverage_table)

    print("[3/4] Gerando correlacao estatica...")
    corr_matrix, fig_corr = build_static_correlation_heatmap(returns_panel)
    save_thesis_figure(fig_corr, "price_fig_corr_heatmap_static")

    print("[4/4] Gerando correlacoes rolling...")
    rolling_corr = compute_clean_rolling_correlations(returns_panel, target_asset="BTC", windows=(30, 90, 180))
    fig_roll = plot_clean_rolling_correlations(rolling_corr, target_asset="BTC", window=args.rolling_window, save=True)
    rolling_summary = summarize_rolling_correlation_window(rolling_corr, window=args.rolling_window)

    output_dir = ROOT / RESULTS_TABLES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    coverage_table.to_csv(output_dir / "cross_asset_coverage_table.csv", index=False)
    sync_summary.to_csv(output_dir / "cross_asset_sync_summary.csv", index=False)
    corr_matrix.to_csv(output_dir / "cross_asset_corr_matrix.csv")
    rolling_summary.to_csv(output_dir / f"cross_asset_rolling_summary_w{args.rolling_window}.csv", index=False)

    print("Artefatos gerados:")
    print(f"  - {output_dir / 'cross_asset_coverage_table.csv'}")
    print(f"  - {output_dir / 'cross_asset_sync_summary.csv'}")
    print(f"  - {output_dir / 'cross_asset_corr_matrix.csv'}")
    print(f"  - {output_dir / f'cross_asset_rolling_summary_w{args.rolling_window}.csv'}")
    print("Figuras geradas em results/figures/.")

    _ = fig_roll


if __name__ == "__main__":
    raise SystemExit(main())
