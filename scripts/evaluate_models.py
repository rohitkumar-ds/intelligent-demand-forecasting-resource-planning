"""Offline rolling-origin evaluation. Saves auditable model metrics."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_PATH, EVAL_DIR
from src.data_validation import validate_dataset
from src.evaluation import rolling_origin_evaluate

def main():
    parser = argparse.ArgumentParser(description="Evaluate forecasting models using rolling-origin backtesting.")
    parser.add_argument("--target", type=str, default="product_demand", help="Target column to evaluate.")
    parser.add_argument("--folds", type=int, default=3, help="Number of rolling-origin folds.")
    parser.add_argument("--horizon", type=int, default=14, help="Forecast horizon in days per fold.")
    parser.add_argument("--tune", action="store_true", help="Enable hyperparameter fine-tuning during cross-validation.")
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    report = validate_dataset(df)
    if not report.passed:
        raise RuntimeError(f"Dataset validation failed: {report}")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating models on '{args.target}' (folds={args.folds}, horizon={args.horizon}, tune={args.tune})...")

    metrics = rolling_origin_evaluate(
        df=df,
        target=args.target,
        folds=args.folds,
        horizon=args.horizon,
        tune=args.tune,
    )

    output_path = EVAL_DIR / "model_metrics.csv"
    metrics.to_csv(output_path, index=False)
    print(f"\nBenchmark results saved to: {output_path.relative_to(ROOT)}\n")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
