# Comparative Multi-Asset Crypto Price Forecasting

This repository contains the Python code accompanying the study **"Análise Comparativa de Modelos Estatísticos e de Aprendizado de Máquina para Previsão de Preços e Movimentos Direcionais de Criptomoedas"**.

The project compares statistical and deep learning approaches for forecasting cryptocurrency prices and directional movements across four assets: **BTC-USD**, **ETH-USD**, **LTC-USD**, and **XRP-USD**. The study evaluates **ARIMA** and **ETS** as statistical benchmarks, and **LSTM**, **RNN**, and **GRU** as recurrent neural network models.

The codebase was organized from the original research notebook into a minimally reproducible academic repository. Its goal is to preserve the methodological logic of the study while making the data preparation, experiment execution, and result generation easier to inspect and reproduce.

## Study Summary

- **Assets:** Bitcoin, Ethereum, Litecoin, Ripple
- **Sample period:** 2017-11-09 to 2024-12-31
- **Forecast horizons:** 1, 7, and 14 days ahead
- **Tasks:**
  - price forecasting
  - directional movement forecasting
- **Model families:**
  - Statistical: ARIMA, ETS
  - Deep learning: LSTM, RNN, GRU
- **Evaluation metrics:**
  - RMSE
  - MAE
  - MAPE
  - sMAPE
  - Theil's U1
  - Theil's U2
  - directional accuracy
- **Statistical comparison:** Diebold-Mariano test against a naive random walk baseline

## Main Findings

In the study, **ARIMA achieved the best overall price-forecasting performance** in most asset-horizon combinations when evaluated by standard error metrics. However, its advantage was often **statistically equivalent to a naive random walk baseline**, highlighting the structural difficulty of forecasting cryptocurrency prices.

For **directional forecasting**, recurrent neural models—especially **RNN** and **GRU**—showed stronger results in some longer-horizon cases, with particular strength for **ETH-USD**.

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── experiments/
│   ├── run_price_data_setup.py
│   ├── run_price_family_pipeline.py
│   └── run_cross_asset_pipeline.py
├── notebooks/
│   └── TCC_final_original.ipynb
├── results/
│   ├── figures/
│   └── tables/
├── src/
│   └── crypto_price_forecasting/
├── tests/
├── README.md
├── CITATION.cff
├── pyproject.toml
└── requirements.txt