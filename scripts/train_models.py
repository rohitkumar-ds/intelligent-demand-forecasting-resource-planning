from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import DATA_PATH, EVAL_DIR, MODEL_DIR, TARGETS
from src.data_validation import validate_dataset
from src.model_registry import MODEL_REGISTRY
from src.preprocessing import chronological_split

def main():
    parser = argparse.ArgumentParser(description="Train and serialize forecasting models offline.")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter fine-tuning during model training.",
    )
    args = parser.parse_args()

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    report = validate_dataset(df)
    if not report.passed:
        raise RuntimeError(f"Dataset validation failed: {report}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    train, _ = chronological_split(df, test_fraction=0.2)
    manifest: dict[str, dict[str, str]] = {}

    print(f"Starting training pipeline (tune={args.tune})...\n")

    # 1. Train production candidate models for all operational targets
    for target in TARGETS:
        print(f"Training XGBoost for target: '{target}'...")
        xgb_cls = MODEL_REGISTRY["XGBoost"]
        model = xgb_cls()
        model.fit(train, target, tune=args.tune)

        artifact_path = MODEL_DIR / f"xgboost_{target}.joblib"
        model.save(artifact_path)
        manifest[target] = {
            "model": "XGBoost",
            "path": str(artifact_path.relative_to(ROOT)),
        }

    # 2. Train benchmark contenders for the primary demand target
    contenders = [
        "Seasonal Naive",
        "Drift",
        "SARIMA",
        "XGBoost",
        "LSTM",
        "Temporal Attention",
    ]

    for name in contenders:
        if name not in MODEL_REGISTRY:
            continue
        try:
            print(f"Training contender model: '{name}' on 'product_demand'...")
            model_cls = MODEL_REGISTRY[name]
            model = model_cls()

            if "tune" in model.fit.__code__.co_varnames:
                model.fit(train, "product_demand", tune=args.tune)
            else:
                model.fit(train, "product_demand")

            ext = ".pt" if name in {"LSTM", "Temporal Attention"} else ".joblib"
            artifact_path = MODEL_DIR / f"{name.lower().replace(' ', '_')}_demand{ext}"
            model.save(artifact_path)
            manifest[name] = {
                "model": name,
                "path": str(artifact_path.relative_to(ROOT)),
            }
        except Exception as e:
            print(f"Failed training {name}: {e}")
            manifest[name] = {"model": name, "error": str(e)}
    # Save manifest
    manifest_path = MODEL_DIR / "registry.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nTraining complete. Manifest saved to {manifest_path.relative_to(ROOT)}:\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

