from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

@dataclass
class ForecastResult:
    dates: pd.DatetimeIndex
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    model_name: str
    confidence_level: float


class BaseForecastModel(ABC):
    name = "Base"

    def __init__(self):
        self.residuals = np.array([], dtype=float)
        self.calibration_residuals = np.array([], dtype=float)
        self.fitted = False
        self.target: str | None = None

    @abstractmethod
    def fit(self, df: pd.DataFrame, target: str) -> BaseForecastModel:
        """Fit the model to historical time-series data."""
        ...

    @abstractmethod
    def predict(
        self,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Generate multi-step point predictions with optional future exogenous data."""
        ...

    def set_calibration_residuals(self, residuals: np.ndarray | list[float]) -> None:
        """Set out-of-fold or holdout calibration residuals for prediction interval estimation."""
        self.calibration_residuals = np.asarray(residuals, dtype=float)

    def interval(self, point: np.ndarray, confidence: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Compute empirical prediction intervals using calibration or fit residuals."""
        point = np.asarray(point, dtype=float)
        resids = (
            self.calibration_residuals
            if len(self.calibration_residuals) >= 10
            else self.residuals
        )

        if len(resids) < 10:
            sigma = max(float(np.std(point)) * 0.05, 1.0)
            z = 1.96
            return point - z * sigma, point + z * sigma

        alpha = 1.0 - confidence
        ql = np.quantile(resids, alpha / 2.0)
        qu = np.quantile(resids, 1.0 - alpha / 2.0)
        return point + ql, point + qu

    def forecast(
        self,
        history: pd.DataFrame,
        horizon: int,
        confidence: float = 0.95,
        future_exog: pd.DataFrame | None = None,
    ) -> ForecastResult:
        """Generate point forecasts bounded at 0 along with prediction intervals."""
        raw_pred = self.predict(history, horizon, future_exog=future_exog)
        point = np.maximum(np.asarray(raw_pred, dtype=float), 0.0)
        lo, hi = self.interval(point, confidence)

        last_date = pd.to_datetime(history["date"]).max()
        dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

        return ForecastResult(
            dates=dates,
            point=point,
            lower=np.maximum(lo, 0.0),
            upper=np.maximum(hi, point),
            model_name=self.name,
            confidence_level=confidence,
        )

    def save(self, path: str | Path) -> None:
        """Unified persistence method using joblib by default."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> BaseForecastModel:
        """Unified loader restoring model objects from disk."""
        return joblib.load(path)
