from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crypto_price_forecasting import arima, baselines, comparison, config, cross_asset, data, dm_test, ets, metrics, visualization


class SmokeTests(unittest.TestCase):
    def test_core_modules_import(self):
        self.assertIsNotNone(arima)
        self.assertIsNotNone(baselines)
        self.assertIsNotNone(comparison)
        self.assertIsNotNone(config)
        self.assertIsNotNone(cross_asset)
        self.assertIsNotNone(data)
        self.assertIsNotNone(dm_test)
        self.assertIsNotNone(ets)
        self.assertIsNotNone(metrics)
        self.assertIsNotNone(visualization)

    def test_central_constants_exist(self):
        self.assertEqual(config.MAIN_PRICE_HORIZONS, (1, 7, 14))
        self.assertIn("BTC-USD", config.APPROVED_ARIMA_ORDERS)
        self.assertEqual(config.RESULTS_FIGURES_DIR, "results/figures")
        self.assertEqual(config.RESULTS_TABLES_DIR, "results/tables")

    def test_main_utility_functions_exist(self):
        self.assertTrue(callable(data.load_prices))
        self.assertTrue(callable(arima.walk_forward_arima_price_compare))
        self.assertTrue(callable(ets.evaluate_ets_price_multiasset))
        self.assertTrue(callable(comparison.build_arima_price_table))
        self.assertTrue(callable(baselines.build_naive_rw_baseline))
        self.assertTrue(callable(dm_test.run_dm_tests_price))
        self.assertTrue(callable(metrics.compute_price_forecast_metrics))
        self.assertTrue(callable(visualization.save_thesis_figure))

    def test_seed_summary_returns_flat_columns(self):
        tbl = pd.DataFrame(
            [
                {"Asset": "BTC-USD", "Horizon": 1, "Model": "LSTM", "RMSE": 1.0, "MAE": 0.8, "DirectionalAccuracy": 0.6},
                {"Asset": "BTC-USD", "Horizon": 1, "Model": "LSTM", "RMSE": 1.2, "MAE": 0.9, "DirectionalAccuracy": 0.5},
            ]
        )
        summary = comparison.build_network_seed_summary(tbl)
        self.assertIn("rmse_mean", summary.columns)
        self.assertIn("rmse_std", summary.columns)
        self.assertIn("directional_accuracy_mean", summary.columns)
        self.assertNotIsInstance(summary.columns, pd.MultiIndex)

    def test_arima_selection_summary_keeps_price_and_diff_orders(self):
        ic_table = pd.DataFrame([{"order": (1, 1, 2), "aic": 10.0, "bic": 11.0, "nobs": 100}])
        diag_table = pd.DataFrame(
            [
                {
                    "order": (1, 0, 2),
                    "resid_mean": 0.0,
                    "resid_std": 1.0,
                    "lb_p_lag10": 0.2,
                    "lb_p_lag20": 0.3,
                    "lb_p_lag30": 0.4,
                }
            ]
        )
        wf_table = pd.DataFrame(
            [{"order": (1, 1, 2), "rmse": 1.0, "mae": 0.9, "mape": 5.0, "directional_accuracy": 0.5, "n_test": 20}]
        )

        summary = arima.build_arima_selection_summary(ic_table, diag_table, wf_table, name="smoke")

        self.assertIn("price_order", summary.columns)
        self.assertIn("diff_order", summary.columns)
        self.assertEqual(summary.iloc[0]["price_order"], (1, 1, 2))
        self.assertEqual(summary.iloc[0]["diff_order"], (1, 0, 2))

    def test_experiment_scripts_expose_main(self):
        script_names = [
            "run_price_data_setup.py",
            "run_price_family_pipeline.py",
            "run_cross_asset_pipeline.py",
        ]
        for script_name in script_names:
            script_path = ROOT / "experiments" / script_name
            spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            self.assertTrue(hasattr(module, "main"))
            self.assertTrue(callable(module.main))


if __name__ == "__main__":
    unittest.main()
