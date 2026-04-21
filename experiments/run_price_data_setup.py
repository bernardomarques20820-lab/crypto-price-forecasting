from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_price_forecasting.config import MAIN_PRICE_ASSETS, RESULTS_TABLES_DIR
from crypto_price_forecasting.data import build_multiasset_price_map, comparison_protocol_summary, load_prices


def parse_args():
    parser = argparse.ArgumentParser(description="Baixa os precos-base e gera a tabela de protocolo da comparacao principal.")
    parser.add_argument("--start", default="2017-11-09")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--train-frac", type=float, default=0.8)
    return parser.parse_args()


def main():
    args = parse_args()

    print("[1/3] Baixando precos principais...")
    raw_prices = load_prices(MAIN_PRICE_ASSETS, start=args.start, end=args.end)

    print("[2/3] Validando e organizando price_map...")
    price_map = build_multiasset_price_map({asset: raw_prices[asset] for asset in MAIN_PRICE_ASSETS})

    print("[3/3] Gerando protocolo da comparacao...")
    protocol = comparison_protocol_summary(price_map, train_frac=args.train_frac)

    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (ROOT / RESULTS_TABLES_DIR).mkdir(parents=True, exist_ok=True)

    raw_prices.to_csv(ROOT / "data" / "raw" / "main_prices.csv")
    protocol.to_csv(ROOT / RESULTS_TABLES_DIR / "comparison_protocol.csv", index=False)

    print("Arquivos gerados:")
    print(f"  - {ROOT / 'data' / 'raw' / 'main_prices.csv'}")
    print(f"  - {ROOT / RESULTS_TABLES_DIR / 'comparison_protocol.csv'}")


if __name__ == "__main__":
    raise SystemExit(main())
