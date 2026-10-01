from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .model_registry import MODEL_REGISTRY
from .preprocessing import TimeSeriesPreprocessor

KNOWN_FUTURE_EXOGENOUS = ("price", "promotion", "holiday", "lead_time_days")

def mape(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs((y - p) / np.maximum(np.abs(y), 1e-8))) * 100.0)


def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    e = y - q
    return float(np.mean(np.maximum(alpha * e, (alpha - 1.0) * e)))

def metrics(y: np.ndarray, result) -> dict[str, float]:
    y = np.asarray(y, float)
    p = np.asarray(result.point,float)
    lo = np.asarray(result.lower,float)
    hi = np.asarray(result.upper,float)
    a = (1.0 - result.confidence_level) / 2.0
    coverage = float(np.mean((y >= lo) & (y <= hi)) * 100.0)
    expected_cov = result.confidence_level * 100.0
    errors = y - p

    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAE": float(mean_absolute_error(y, p)),
        "MAPE (%)": mape(y, p),
        "Bias" : float(np.mean(errors)),
        "Pinball Lower": pinball(y, lo, a),
        "Pinball Upper": pinball(y, hi, 1.0 - a),
        "Interval Coverage (%)": coverage,
        "Coverage Error": abs(coverage - expected_cov),
        "Avg Interval Width": float(np.mean(hi - lo)),
    }

def _future_exog_for_evaluation(test: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in KNOWN_FUTURE_EXOGENOUS if c in test.columns]
    return test[["date", *cols]].copy()



def rolling_origin_evaluate(
    df: pd.DataFrame,
    target: str = "product_demand",
    folds: int = 5,
    horizon: int = 14,
    confidence: float = 0.95,
    tune: bool = False,
    model_names: list[str] | None = None,
) -> pd.DataFrame:
    names = model_names or list(MODEL_REGISTRY)
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"])
    x = x.sort_values("date").reset_index(drop=True)
    rows = []
    min_train = len(x) - folds * horizon

    for name in names:
        fold_metrics = []
        for fold in range(folds):
            train_end = min_train + fold * horizon
            train = x.iloc[:train_end].copy()
            test = x.iloc[train_end : train_end + horizon].copy()
            try:
                model = MODEL_REGISTRY[name]()
                # Pass tuning flag if supported by the model fit signature
                if "tune" in model.fit.__code__.co_varnames:
                    model.fit(train, target, tune=tune)
                else:
                    model.fit(train, target)

                future_exog = _future_exog_for_evaluation(test)
                # Pass test exogenous data to forecast
                r = model.forecast(train, horizon, confidence=confidence, future_exog=future_exog)
                fold_metrics.append(metrics(test[target].to_numpy(), r))
            except Exception as e:
                rows.append({"Model": name, "Status": f"FAILED: {e}"})
                fold_metrics = []
                break

        if fold_metrics:
            row = {"Model": name, "Status": "OK", "Folds": len(fold_metrics)}
            for k in fold_metrics[0]:
                row[k] = float(np.mean([m[k] for m in fold_metrics]))
            row["Folds"] = folds
            row["Stability"] = float(np.std([m["RMSE"] for m in fold_metrics]))
            rows.append(row)
        else:
            rows.append({"Model": name, "Status": f"FAILED"})

    out = pd.DataFrame(rows) 
    if out.empty:
        return out

    ok = out["Status"].eq("OK")
    if ok.any():
        # Primary selection criterion is forecast accuracy (MAE). RMSE is the
        # first tie-breaker, then interval coverage error. This is simpler and
        # more interpretable than averaging unrelated percentile scores.
        ok_df = out.loc[ok].copy()
        ok_df = ok_df.sort_values(
            ["MAE", "RMSE", "Coverage Error", "Stability"],
            ascending=[True, True, True, True],
        )
        winner = ok_df.index[0]
        out["Production Candidate"] = False
        out.loc[winner, "Production Candidate"] = True
    else:
        out["Production Candidate"] = False

    return out.sort_values(
        ["Production Candidate", "MAE", "RMSE"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)
