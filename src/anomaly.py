"""Historical anomaly detection using robust rolling median and MAD residuals."""
from __future__ import annotations
import numpy as np
import pandas as pd


def detect_anomalies(
    df: pd.DataFrame,
    target: str = "product_demand",
    window: int = 28,
    z_threshold: float = 3.5,
) -> pd.DataFrame:
    """
    Detect historical time-series anomalies using rolling median & MAD robust z-scores.
    Works dynamically for any selected target metric.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")

    x = df.copy()
    x["date"] = pd.to_datetime(x["date"])
    x = x.sort_values("date").reset_index(drop=True)

    series = x[target].astype(float)
    med = series.rolling(window, center=True, min_periods=max(window // 4, 3)).median()
    resid = series - med

    # Compute rolling MAD (Median Absolute Deviation)
    def calc_mad(v: np.ndarray) -> float:
        m = np.median(v)
        return float(np.median(np.abs(v - m)))

    mad = resid.rolling(window, min_periods=max(window // 4, 3)).apply(calc_mad, raw=True)

    # Scale factor for normal distribution consistency (0.6745)
    eps = max(float(np.nanstd(series)) * 0.01, 1e-6)
    robust_z = 0.6745 * resid / np.maximum(mad, eps)

    x["anomaly_score"] = robust_z.fillna(0.0)
    x["anomaly"] = np.abs(x["anomaly_score"]) >= z_threshold

    return x