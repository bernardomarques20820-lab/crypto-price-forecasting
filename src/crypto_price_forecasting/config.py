from __future__ import annotations

NOTEBOOK_SOURCE_PATH = "notebooks/TCC_final_original.ipynb"
NOTEBOOK_PATH_FINAL = "TCC_final_price.ipynb"

RESULTS_DIR = "results"
RESULTS_FIGURES_DIR = "results/figures"
RESULTS_TABLES_DIR = "results/tables"

MAIN_PRICE_ASSETS = ("BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD")
MAIN_PRICE_HORIZONS = (1, 7, 14)

# The notebook contains conflicting ARIMA orders across sections.
# Conservative decision:
# - keep the orders used in the main family comparison as the approved default
# - preserve the diagnostic-only orders separately instead of overwriting history
# This avoids inventing new orders and keeps the comparison pipeline aligned with
# the version effectively used in the final cross-family tables.
APPROVED_ARIMA_ORDERS = {
    "BTC-USD": (1, 1, 0),
    "ETH-USD": (2, 1, 2),
    "LTC-USD": (1, 1, 1),
    "XRP-USD": (1, 1, 1),
}

# Historical notebook orders used in the multi-asset diagnostic block.
ARIMA_DIAGNOSTIC_ORDERS = {
    "BTC-USD": (1, 1, 2),
    "ETH-USD": (2, 1, 2),
    "LTC-USD": (2, 1, 2),
    "XRP-USD": (2, 1, 2),
}

# Aliases kept to minimize code churn in the extracted modules.
ARIMA_PAPER_ORDERS = APPROVED_ARIMA_ORDERS
ARIMA_RESIDUAL_ORDERS = APPROVED_ARIMA_ORDERS

THESIS_MODEL_ORDER = ["ARIMA", "ETS", "LSTM", "RNN", "GRU", "NAIVE_ZERO"]
THESIS_ASSET_ORDER = ["BTC-USD", "ETH-USD", "LTC-USD", "XRP-USD"]
THESIS_MODEL_COLORS = {
    "ARIMA": "#1f4e79",
    "ETS": "#3a7d44",
    "LSTM": "#9c4f2f",
    "RNN": "#6b4c9a",
    "GRU": "#2b6f89",
    "NAIVE_ZERO": "#6c757d",
}
THESIS_METRIC_COLORS = {"RMSE": "#1f4e79", "MAE": "#3a7d44", "Direction": "#9c4f2f"}

CROSS_ASSET_START = "2017-11-09"
CROSS_ASSET_END = "2025-01-01"
CROSS_ASSET_CANDIDATES = {
    "ETH": ["ETH-USD"],
    "LTC": ["LTC-USD"],
    "XRP": ["XRP-USD"],
    "DASH": ["DASH-USD"],
    "Gold": ["GC=F"],
    "Silver": ["SI=F"],
    "Copper": ["HG=F"],
    "Oil": ["CL=F"],
    "S&P500": ["^GSPC"],
    "DJI": ["^DJI"],
    "NASDAQ": ["^IXIC"],
    "JP225": ["^N225"],
    "CBOE": ["^VIX"],
    "BOVESPA": ["^BVSP"],
    "DXY": ["DX-Y.NYB", "^DXY", "UUP"],
    "EURUSD": ["EURUSD=X"],
    "GBPUSD": ["GBPUSD=X"],
    "USDJPY": ["JPY=X", "USDJPY=X"],
    "USDCAD": ["CAD=X", "USDCAD=X"],
    "AUDUSD": ["AUDUSD=X"],
    "USDSGD": ["SGD=X", "USDSGD=X"],
    "USDCNY": ["CNY=X", "USDCNY=X"],
    "USDRUB": ["RUB=X", "USDRUB=X"],
}

# Alias preserved for notebook compatibility.
CANDIDATES = CROSS_ASSET_CANDIDATES
