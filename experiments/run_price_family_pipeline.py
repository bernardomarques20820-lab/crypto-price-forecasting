from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_price_forecasting.baselines import build_naive_rw_baseline
from crypto_price_forecasting.comparison import (
    build_arima_price_table,
    build_price_family_comparison_table,
    build_price_family_predictions_table,
    build_price_family_rank_summary,
    build_price_family_winners_table,
    merge_family_with_naive_baseline,
)
from crypto_price_forecasting.config import APPROVED_ARIMA_ORDERS, MAIN_PRICE_ASSETS, MAIN_PRICE_HORIZONS, RESULTS_TABLES_DIR
from crypto_price_forecasting.data import build_multiasset_price_map, load_prices
from crypto_price_forecasting.dm_test import run_dm_tests_price
from crypto_price_forecasting.ets import evaluate_ets_price_multiasset
from crypto_price_forecasting.metrics import build_price_nn_table


TABLE_COLS = ["Asset", "Horizon", "RMSE", "MAE", "MAPE", "sMAPE", "TheilU1", "TheilU2", "DirectionalAccuracy", "Model"]


def parse_args():
    parser = argparse.ArgumentParser(description="Executa a comparacao principal entre familias em precos.")
    parser.add_argument("--start", default="2017-11-09")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--horizons", nargs="+", type=int, default=list(MAIN_PRICE_HORIZONS))
    parser.add_argument("--include-neural", action="store_true", help="Executa LSTM, RNN e GRU. Pode ser custoso.")
    return parser.parse_args()


def empty_model_table():
    return pd.DataFrame(columns=TABLE_COLS)


def main():
    args = parse_args()
    horizons = tuple(args.horizons)

    print("[1/6] Baixando e validando precos...")
    raw_prices = load_prices(MAIN_PRICE_ASSETS, start=args.start, end=args.end)
    price_map = build_multiasset_price_map({asset: raw_prices[asset] for asset in MAIN_PRICE_ASSETS})

    print("[2/6] Executando ARIMA aprovado...")
    outs_arima_price, tbl_arima_price = build_arima_price_table(
        price_map,
        arima_orders=APPROVED_ARIMA_ORDERS,
        assets=MAIN_PRICE_ASSETS,
        horizons=horizons,
        train_frac=args.train_frac,
    )

    print("[3/6] Executando ETS...")
    outs_ets_price = evaluate_ets_price_multiasset(price_map, horizons=horizons, train_frac=args.train_frac)
    tbl_ets_price = pd.concat([out["metrics_table"] for out in outs_ets_price.values()], ignore_index=True)
    tbl_ets_price["Model"] = "ETS"
    tbl_ets_price = tbl_ets_price[TABLE_COLS].sort_values(["Asset", "Horizon"]).reset_index(drop=True)

    if args.include_neural:
        from crypto_price_forecasting.neural import (
            run_gru_price_multiasset,
            run_lstm_price_multiasset,
            run_rnn_price_multiasset,
        )

        print("[4/6] Executando redes recorrentes...")
        outs_lstm_price = run_lstm_price_multiasset(price_map, horizons=horizons, train_frac=args.train_frac)
        outs_rnn_price = run_rnn_price_multiasset(price_map, horizons=horizons, train_frac=args.train_frac)
        outs_gru_price = run_gru_price_multiasset(price_map, horizons=horizons, train_frac=args.train_frac)
        tbl_lstm_price = build_price_nn_table(outs_lstm_price, "LSTM")
        tbl_rnn_price = build_price_nn_table(outs_rnn_price, "RNN")
        tbl_gru_price = build_price_nn_table(outs_gru_price, "GRU")
    else:
        print("[4/6] Pulando redes recorrentes (use --include-neural para habilitar).")
        outs_lstm_price = {}
        outs_rnn_price = {}
        outs_gru_price = {}
        tbl_lstm_price = empty_model_table()
        tbl_rnn_price = empty_model_table()
        tbl_gru_price = empty_model_table()

    print("[5/6] Consolidando tabelas de comparacao...")
    tbl_family_price_all = build_price_family_comparison_table(
        tbl_arima_price,
        tbl_ets_price,
        tbl_lstm_price,
        tbl_rnn_price,
        tbl_gru_price,
        assets=MAIN_PRICE_ASSETS,
        horizons=horizons,
    )

    outs_naive_price, tbl_naive_price = build_naive_rw_baseline(price_map, horizons=horizons, train_frac=args.train_frac)
    tbl_family_price_all = merge_family_with_naive_baseline(tbl_family_price_all, tbl_naive_price)
    tbl_family_price_winners = build_price_family_winners_table(tbl_family_price_all)
    tbl_family_price_rank_summary = build_price_family_rank_summary(tbl_family_price_all)

    family_output_registry = {"ARIMA": outs_arima_price, "ETS": outs_ets_price, "NAIVE_ZERO": outs_naive_price}
    if args.include_neural:
        family_output_registry.update({"LSTM": outs_lstm_price, "RNN": outs_rnn_price, "GRU": outs_gru_price})

    tbl_family_predictions = build_price_family_predictions_table(
        family_output_registry,
        assets=MAIN_PRICE_ASSETS,
        horizons=horizons,
    )

    print("[6/6] Executando DM test e persistindo resultados...")
    tbl_dm_price = run_dm_tests_price(tbl_family_predictions, benchmark_model="NAIVE_ZERO")

    output_dir = ROOT / RESULTS_TABLES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    tbl_arima_price.to_csv(output_dir / "tbl_arima_price.csv", index=False)
    tbl_ets_price.to_csv(output_dir / "tbl_ets_price.csv", index=False)
    tbl_naive_price.to_csv(output_dir / "tbl_naive_price.csv", index=False)
    tbl_family_price_all.to_csv(output_dir / "tbl_family_price_all.csv", index=False)
    tbl_family_price_winners.to_csv(output_dir / "tbl_family_price_winners.csv", index=False)
    tbl_family_price_rank_summary.to_csv(output_dir / "tbl_family_price_rank_summary.csv", index=False)
    tbl_family_predictions.to_csv(output_dir / "tbl_family_predictions.csv", index=False)
    tbl_dm_price.to_csv(output_dir / "tbl_dm_price.csv", index=False)

    print("Tabelas geradas em:")
    for path in sorted(output_dir.glob("*.csv")):
        print(f"  - {path}")


if __name__ == "__main__":
    raise SystemExit(main())
