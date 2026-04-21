from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yfinance as yf


def _dl_one(symbol, start, end, retries=3, sleep=1.5):
    """Baixa a série de fechamento de um único ativo no Yahoo Finance."""
    last_err = None
    for _ in range(retries):
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                col = "Close"
                close = df[[col]].copy() if isinstance(df[col], pd.DataFrame) else df[col]

                if isinstance(close, pd.DataFrame):
                    close.columns = [symbol]
                    return close
                return close.rename(symbol).to_frame()
        except Exception as exc:  # pragma: no cover - depende de rede
            last_err = exc
        time.sleep(sleep)

    print(f"[WARN] Falha ao baixar {symbol}: {last_err}")
    return pd.DataFrame(columns=[symbol])


def load_prices(tickers, start, end):
    """Baixa um ou mais ativos e retorna um DataFrame consolidado."""
    if isinstance(tickers, str):
        tickers = [tickers]

    frames = []
    for ticker in tickers:
        close_t = _dl_one(ticker, start, end)
        frames.append(close_t)

    close = pd.concat(frames, axis=1).ffill().dropna(how="all")
    if close.empty:
        raise RuntimeError(
            "Nenhum preço baixado. Tente novamente mais tarde ou reduza a lista de tickers."
        )
    close.index.name = "Date"
    return close


def prepare_price_series(price_series, name="series"):
    """Valida e retorna a série de preços sem transformação."""
    s = pd.Series(price_series, name=name).copy()
    s = s.sort_index().dropna()

    if s.empty:
        raise ValueError(f"{name}: a série ficou vazia após ordenação e remoção de NaN.")

    if (s <= 0).any():
        raise ValueError(f"{name}: todos os valores devem ser positivos (série de preços).")

    s.name = "price"
    return s


def build_multiasset_price_map(asset_map):
    """Constrói mapa {ticker: série de preços validada}."""
    price_map = {}
    for asset, series in asset_map.items():
        price = prepare_price_series(series, name=asset)
        price_map[asset] = price
        print(f"{asset}: n_prices={len(price)}")
    return price_map


def comparison_protocol_summary(price_map, horizons=None, train_frac=0.8):
    if not 0 < train_frac < 1:
        raise ValueError("train_frac deve estar estritamente entre 0 e 1.")

    if horizons is not None:
        resolved_horizons = (
            tuple(horizons)
            if not isinstance(horizons, (int, np.integer))
            else (int(horizons),)
        )
        print(f"Usando horizons informado: {resolved_horizons}")
    else:
        resolved_horizons = (1, 7, 14)
        print("Usando horizons padrão da comparação principal: (1, 7, 14)")

    rows = []
    for asset, price in price_map.items():
        n_obs = len(price)
        train_size = int(np.floor(n_obs * train_frac))
        test_size = n_obs - train_size
        rows.append(
            {
                "asset": asset,
                "n_obs": n_obs,
                "train_size": train_size,
                "test_size": test_size,
                "target": "price (nivel)",
                "candidate_families": "ARIMA(d=1), ETS, LSTM, RNN, GRU",
                "horizons_used": resolved_horizons,
            }
        )

    return pd.DataFrame(rows)
