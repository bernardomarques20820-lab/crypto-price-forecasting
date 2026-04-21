from __future__ import annotations

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from .metrics import (
    compute_basic_forecast_metrics,
    compute_price_forecast_metrics,
    empty_price_nn_metrics_row,
)
from .visualization import save_thesis_figure

try:
    from IPython.display import display
except Exception:  # pragma: no cover - optional
    display = None


def _make_sequences(arr, look_back, h):
    X, y = [], []
    for i in range(look_back, len(arr) - h + 1):
        X.append(arr[i - look_back : i])
        y.append(arr[i + h - 1])
    X = np.array(X).reshape(-1, look_back, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y


def log_step_failure(model_name, asset, step_label, exc):
    asset_label = asset if asset is not None else "N/A"
    print(f"[{model_name}][{asset_label}][{step_label}] failed: {exc}")


class LSTMModel:
    def __init__(self, h=1, look_back=60, units1=64, units2=32, dropout=0.2, lr=1e-3, epochs=20, batch_size=32, seed=42):
        self.h = h
        self.look_back = look_back
        self.units1 = units1
        self.units2 = units2
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.buffer = None
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception as exc:
            log_step_failure("LSTM", None, "seed_setup", exc)

    def _build(self):
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.look_back, 1)),
                keras.layers.LSTM(self.units1, return_sequences=True),
                keras.layers.Dropout(self.dropout),
                keras.layers.LSTM(self.units2),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer=keras.optimizers.Adam(self.lr), loss="mse")
        return model

    def fit(self, y_train: pd.Series):
        y = pd.Series(y_train).astype("float32").dropna().values.reshape(-1, 1)
        y_scaled = self.scaler.fit_transform(y)
        X, t = _make_sequences(y_scaled.ravel(), self.look_back, self.h)
        if X.size == 0:
            raise ValueError(
                f"Serie curta para LSTM: precisa de look_back({self.look_back}) + h({self.h}) pontos no minimo."
            )
        self.model = self._build()
        self.model.fit(X, t, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.buffer = y_scaled[-self.look_back :].reshape(1, self.look_back, 1)

    def predict(self, h: int):
        assert h == self.h, f"Modelo direto foi treinado para h={self.h}"
        pred_scaled = self.model.predict(self.buffer, verbose=0)[0, 0]
        return float(self.scaler.inverse_transform([[pred_scaled]])[0, 0])

    def update(self, new_y):
        new_scaled = self.scaler.transform(np.array([[float(new_y)]]))[0, 0]
        self.buffer = np.roll(self.buffer, -1, axis=1)
        self.buffer[0, -1, 0] = new_scaled


class RNNModel:
    def __init__(self, h=1, look_back=60, units1=64, units2=32, dropout=0.2, lr=1e-3, epochs=20, batch_size=32, seed=42):
        self.h = h
        self.look_back = look_back
        self.units1 = units1
        self.units2 = units2
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.buffer = None
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception as exc:
            log_step_failure("RNN", None, "seed_setup", exc)

    def _build(self):
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.look_back, 1)),
                keras.layers.SimpleRNN(self.units1, return_sequences=True, activation="tanh"),
                keras.layers.Dropout(self.dropout),
                keras.layers.SimpleRNN(self.units2, activation="tanh"),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer=keras.optimizers.Adam(self.lr), loss="mse")
        return model

    def fit(self, y_train: pd.Series):
        y = pd.Series(y_train).astype("float32").dropna().values.reshape(-1, 1)
        y_scaled = self.scaler.fit_transform(y)
        X, t = _make_sequences(y_scaled.ravel(), self.look_back, self.h)
        if X.size == 0:
            raise ValueError(f"Serie curta para RNN: precisa de look_back({self.look_back}) + h({self.h}) pontos.")
        self.model = self._build()
        self.model.fit(X, t, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.buffer = y_scaled[-self.look_back :].reshape(1, self.look_back, 1)

    def predict(self, h: int):
        assert h == self.h, f"Modelo direto treinado para h={self.h}"
        pred_scaled = self.model.predict(self.buffer, verbose=0)[0, 0]
        return float(self.scaler.inverse_transform([[pred_scaled]])[0, 0])

    def update(self, new_y):
        new_scaled = self.scaler.transform(np.array([[float(new_y)]]))[0, 0]
        self.buffer = np.roll(self.buffer, -1, axis=1)
        self.buffer[0, -1, 0] = new_scaled


class GRUModel:
    def __init__(self, h=1, look_back=60, units1=64, units2=32, dropout=0.2, lr=1e-3, epochs=20, batch_size=32, seed=42):
        self.h = h
        self.look_back = look_back
        self.units1 = units1
        self.units2 = units2
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.buffer = None
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception as exc:
            log_step_failure("GRU", None, "seed_setup", exc)

    def _build(self):
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.look_back, 1)),
                keras.layers.GRU(self.units1, return_sequences=True, activation="tanh"),
                keras.layers.Dropout(self.dropout),
                keras.layers.GRU(self.units2, activation="tanh"),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer=keras.optimizers.Adam(self.lr), loss="mse")
        return model

    def fit(self, y_train: pd.Series):
        y = pd.Series(y_train).astype("float32").dropna().values.reshape(-1, 1)
        y_scaled = self.scaler.fit_transform(y)
        X, t = _make_sequences(y_scaled.ravel(), self.look_back, self.h)
        if X.size == 0:
            raise ValueError(f"Serie curta para GRU: precisa de look_back({self.look_back}) + h({self.h}) pontos.")
        self.model = self._build()
        self.model.fit(X, t, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        self.buffer = y_scaled[-self.look_back :].reshape(1, self.look_back, 1)

    def predict(self, h: int):
        assert h == self.h, f"Modelo direto treinado para h={self.h}"
        pred_scaled = self.model.predict(self.buffer, verbose=0)[0, 0]
        return float(self.scaler.inverse_transform([[pred_scaled]])[0, 0])

    def update(self, new_y):
        new_scaled = self.scaler.transform(np.array([[float(new_y)]]))[0, 0]
        self.buffer = np.roll(self.buffer, -1, axis=1)
        self.buffer[0, -1, 0] = new_scaled


def evaluate_lstm_walkforward(df, ticker, horizons=(1, 7, 14), look_back=60, lstm_kwargs=None, verbose=True):
    if lstm_kwargs is None:
        lstm_kwargs = dict(units1=64, units2=32, dropout=0.2, lr=1e-3, epochs=20, batch_size=32, seed=42)
    y = df[ticker].astype("float32").dropna()
    n = len(y)
    if n < (look_back + max(horizons) + 5):
        raise ValueError("Serie curta para look_back e horizontes solicitados.")
    split = int(0.8 * n)
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]
    models = {}
    for h in horizons:
        model = LSTMModel(h=h, look_back=look_back, **lstm_kwargs)
        model.fit(y_train)
        models[h] = model
    preds = {h: pd.Series(index=y.index, dtype="float32") for h in horizons}
    last_seen_idx = split - 1
    predict_failures = {h: 0 for h in horizons}
    invalid_horizons = {}
    nan_metrics = {
        "RMSE": np.nan,
        "MAE": np.nan,
        "MAPE": np.nan,
        "sMAPE": np.nan,
        "TheilU1": np.nan,
        "TheilU2": np.nan,
    }
    for t in range(split, n):
        for h in horizons:
            if h in invalid_horizons:
                continue
            target_pos = last_seen_idx + h
            if target_pos < n:
                try:
                    preds[h].iloc[target_pos] = models[h].predict(h)
                except Exception as exc:
                    predict_failures[h] += 1
                    log_step_failure("LSTM", ticker, f"predict_step_{t}_h{h}", exc)
        for h in horizons:
            if h in invalid_horizons:
                continue
            try:
                models[h].update(y.iloc[t])
            except Exception as exc:
                log_step_failure("LSTM", ticker, f"update_step_{t}_h{h}", exc)
                invalid_horizons[h] = f"update_step_{t}_h{h}"
                print(f"[LSTM][{ticker}][h={h}] state invalidated after update failure; metrics will be skipped for this horizon.")
        last_seen_idx = t
    results = {}
    for h in horizons:
        start_eval = max(split, (split - 1) + h)
        y_true = y.iloc[start_eval:]
        y_pred = preds[h].iloc[start_eval:]
        if h in invalid_horizons:
            print(f"[LSTM][{ticker}][metrics_h{h}] skipped due to prior {invalid_horizons[h]} failure.")
            results[h] = nan_metrics.copy()
            continue
        if y_pred.dropna().empty:
            print(f"[LSTM][{ticker}][metrics_h{h}] no valid predictions available; returning NaN metrics.")
            results[h] = nan_metrics.copy()
            continue
        if predict_failures[h] > 0:
            print(f"[LSTM][{ticker}][metrics_h{h}] computed with {predict_failures[h]} logged predict failure(s).")
        results[h] = compute_basic_forecast_metrics(y_true, y_pred)
    if verbose:
        print(f"Ticker: {ticker}")
        print(f"Tamanho total: {n} | Treino: {len(y_train)} | Teste: {len(y_test)}")
        dfm = pd.DataFrame(results).T[["RMSE", "MAE", "MAPE", "sMAPE", "TheilU1", "TheilU2"]].round(6)
        if display is not None:
            display(dfm)
        else:
            print(dfm)
    test_index = y.index[split:]
    for h in horizons:
        plt.figure(figsize=(10, 4))
        plt.plot(y.index, y.values, label="Atual")
        plt.plot(preds[h].index, preds[h].values, label=f"LSTM (walk-forward) h={h}", linestyle="--", linewidth=2)
        plt.axvspan(test_index[0], y.index[-1], alpha=0.1)
        plt.title(f"{ticker} - Atual vs. LSTM walk-forward (h={h})")
        plt.legend()
        plt.tight_layout()
        save_thesis_figure(plt.gcf(), f"price_fig_lstm_forecast_{ticker}_h{h}")
        plt.show()
    return {"metrics": results, "preds": preds, "y": y, "split": split, "models": models}


def _resolve_price_nn_horizons(horizons):
    resolved = tuple(sorted({int(h) for h in horizons}))
    if not resolved:
        raise ValueError("`horizons` deve conter ao menos um horizonte positivo.")
    if any(h <= 0 for h in resolved):
        raise ValueError("Todos os horizontes devem ser inteiros positivos.")
    return resolved


def _resolve_price_nn_look_back(look_back=None):
    if look_back is not None:
        resolved = int(look_back)
        if resolved <= 0:
            raise ValueError("look_back deve ser positivo.")
        print(f"Usando look_back informado: {resolved}")
        return resolved
    try:
        return int(inspect.signature(evaluate_lstm_walkforward).parameters["look_back"].default)
    except Exception:
        return 60


def _default_price_nn_kwargs():
    return {
        "units1": 64,
        "units2": 32,
        "dropout": 0.2,
        "lr": 1e-3,
        "epochs": 20,
        "batch_size": 32,
        "seed": 42,
    }


def _evaluate_recurrent_price_walkforward(series, model_name, model_cls, horizons=(1, 7, 14), train_frac=0.8, look_back=60, model_kwargs=None):
    y = pd.Series(series).astype("float32").sort_index().dropna()
    series_label = getattr(series, "name", None) or getattr(y, "name", None) or "series"
    y.name = series_label

    if y.empty:
        raise ValueError(f"{series_label}: serie de precos vazia.")
    if not 0 < train_frac < 1:
        raise ValueError("train_frac deve estar estritamente entre 0 e 1.")

    resolved_horizons = _resolve_price_nn_horizons(horizons)
    resolved_look_back = int(look_back)

    n = len(y)
    if n < (resolved_look_back + max(resolved_horizons) + 5):
        raise ValueError(f"{series_label}: serie curta para look_back={resolved_look_back}.")

    split = int(np.floor(train_frac * n))
    if split <= resolved_look_back or split >= n:
        raise ValueError(f"{series_label}: split invalido.")

    params = _default_price_nn_kwargs()
    if model_kwargs is not None:
        params.update(model_kwargs)

    y_train = y.iloc[:split]
    preds = {h: pd.Series(np.nan, index=y.index, dtype="float32") for h in resolved_horizons}
    models = {}
    predict_failures = {h: 0 for h in resolved_horizons}
    invalid_horizons = {}

    for h in resolved_horizons:
        try:
            model = model_cls(h=h, look_back=resolved_look_back, **params)
            model.fit(y_train)
            models[h] = model
        except Exception as exc:
            log_step_failure(model_name, series_label, f"fit_h{h}", exc)
            invalid_horizons[h] = f"fit_h{h}"

    last_seen_idx = split - 1
    for t in range(split, n):
        for h in resolved_horizons:
            if h in invalid_horizons:
                continue
            target_pos = last_seen_idx + h
            if target_pos < n:
                try:
                    preds[h].iloc[target_pos] = models[h].predict(h)
                except Exception as exc:
                    predict_failures[h] += 1
                    log_step_failure(model_name, series_label, f"predict_step_{t}_h{h}", exc)
        for h in resolved_horizons:
            if h in invalid_horizons:
                continue
            try:
                models[h].update(y.iloc[t])
            except Exception as exc:
                log_step_failure(model_name, series_label, f"update_step_{t}_h{h}", exc)
                invalid_horizons[h] = f"update_step_{t}_h{h}"
                preds[h].iloc[:] = np.nan
        last_seen_idx = t

    metrics_rows = []
    for h in resolved_horizons:
        if h in invalid_horizons:
            print(f"[{model_name}][{series_label}][h={h}] skipped: {invalid_horizons[h]}")
            metrics_rows.append(empty_price_nn_metrics_row(h))
            continue

        start_eval = max(split, (split - 1) + h)
        y_true_s = y.iloc[start_eval:]
        y_pred_s = preds[h].iloc[start_eval:]
        y_lag_s = y.shift(h).iloc[start_eval:]

        metrics = compute_price_forecast_metrics(y_true_s, y_pred_s, y_lag_s)
        if pd.isna(metrics["RMSE"]):
            print(f"[{model_name}][{series_label}][h={h}] sem previsoes validas - NaN.")
            metrics_rows.append(empty_price_nn_metrics_row(h))
            continue

        if predict_failures[h] > 0:
            print(f"[{model_name}][{series_label}][h={h}] {predict_failures[h]} falhas de predict.")

        metrics_rows.append({"Horizon": int(h), **metrics})

    metrics_table = pd.DataFrame(metrics_rows)[
        ["Horizon", "RMSE", "MAE", "MAPE", "sMAPE", "TheilU1", "TheilU2", "DirectionalAccuracy"]
    ]
    return {
        "preds": preds,
        "y_true": y,
        "y": y,
        "split": split,
        "metrics_table": metrics_table,
        "models": models,
        "look_back": resolved_look_back,
    }


def _run_recurrent_price_multiasset(price_map, model_name, model_cls, horizons=(1, 7, 14), train_frac=0.8, look_back=None, model_kwargs=None):
    resolved_horizons = _resolve_price_nn_horizons(horizons)
    resolved_look_back = _resolve_price_nn_look_back(look_back)
    outs = {}

    for asset, series in price_map.items():
        try:
            print(f"[{model_name}][{asset}] iniciando avaliacao")
            series_named = pd.Series(series).astype("float32").sort_index().dropna()
            series_named.name = asset
            out = _evaluate_recurrent_price_walkforward(
                series_named,
                model_name=model_name,
                model_cls=model_cls,
                horizons=resolved_horizons,
                train_frac=train_frac,
                look_back=resolved_look_back,
                model_kwargs=model_kwargs,
            )
            metrics_table = out["metrics_table"].copy()
            metrics_table["Asset"] = asset
            outs[asset] = {
                "preds": out["preds"],
                "y_true": out["y_true"],
                "y": out["y"],
                "split": out["split"],
                "metrics_table": metrics_table[
                    ["Asset", "Horizon", "RMSE", "MAE", "MAPE", "sMAPE", "TheilU1", "TheilU2", "DirectionalAccuracy"]
                ],
                "models": out["models"],
                "look_back": out["look_back"],
            }
        except Exception as exc:
            log_step_failure(model_name, asset, "asset_eval", exc)
            y_s = pd.Series(series).astype("float32").sort_index().dropna()
            y_s.name = asset
            split = int(np.floor(len(y_s) * train_frac)) if len(y_s) else 0
            outs[asset] = {
                "preds": {h: pd.Series(np.nan, index=y_s.index, dtype="float32") for h in resolved_horizons},
                "y_true": y_s,
                "y": y_s,
                "split": split,
                "metrics_table": pd.DataFrame(
                    [
                        {"Asset": asset, **empty_price_nn_metrics_row(h)}
                        for h in resolved_horizons
                    ]
                ),
                "models": {},
                "look_back": resolved_look_back,
            }

    return outs


def run_lstm_price_multiasset(price_map, horizons=(1, 7, 14), train_frac=0.8, look_back=None, lstm_kwargs=None):
    return _run_recurrent_price_multiasset(
        price_map,
        model_name="LSTM",
        model_cls=LSTMModel,
        horizons=horizons,
        train_frac=train_frac,
        look_back=look_back,
        model_kwargs=lstm_kwargs,
    )


def run_rnn_price_multiasset(price_map, horizons=(1, 7, 14), train_frac=0.8, look_back=None, rnn_kwargs=None):
    return _run_recurrent_price_multiasset(
        price_map,
        model_name="RNN",
        model_cls=RNNModel,
        horizons=horizons,
        train_frac=train_frac,
        look_back=look_back,
        model_kwargs=rnn_kwargs,
    )


def run_gru_price_multiasset(price_map, horizons=(1, 7, 14), train_frac=0.8, look_back=None, gru_kwargs=None):
    return _run_recurrent_price_multiasset(
        price_map,
        model_name="GRU",
        model_cls=GRUModel,
        horizons=horizons,
        train_frac=train_frac,
        look_back=look_back,
        model_kwargs=gru_kwargs,
    )
