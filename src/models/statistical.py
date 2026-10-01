from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base import BaseForecastModel


class SeasonalNaiveModel(BaseForecastModel):
    name = "Seasonal Naive"

    def __init__(self, season: int = 7):
        super().__init__()
        self.season = season

    def fit(self, df: pd.DataFrame, target: str, tune: bool = False) -> SeasonalNaiveModel:
        self.target = target
        y = df.sort_values("date")[target].astype(float).to_numpy()

        if tune:
            best_season = self.season
            best_mse = float("inf")
            candidate_seasons = [7, 14, 28, 30]

            for s in candidate_seasons:
                if len(y) > s * 2:
                    # In-sample one-step seasonal lag error
                    res = y[s:] - y[:-s]
                    mse = float(np.mean(res**2))
                    if mse < best_mse:
                        best_mse = mse
                        best_season = s

            self.season = best_season

        self.residuals = y[self.season:] - y[:-self.season]
        # Store holdout residuals for calibrated intervals
        if len(self.residuals) > self.season:
            self.set_calibration_residuals(self.residuals[-min(len(self.residuals), 180):])

        self.fitted = True
        return self

    def predict(
        self,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> np.ndarray:
        if not self.fitted or self.target is None:
            raise RuntimeError("Model must be fitted before predict() is called.")
        y = history.sort_values("date")[self.target].to_numpy(float)
        return np.resize(y[-self.season:], horizon)


class DriftModel(BaseForecastModel):
    name = "Drift"

    def __init__(self):
        super().__init__()
        self.slope: float = 0.0

    def fit(self, df: pd.DataFrame, target: str, tune: bool = False) -> DriftModel:
        self.target = target
        y = df.sort_values("date")[target].to_numpy(float)
        self.slope = (y[-1] - y[0]) / max(len(y) - 1, 1)

        self.residuals = y[1:] - (y[:-1] + self.slope)
        if len(self.residuals) > 10:
            self.set_calibration_residuals(self.residuals[-min(len(self.residuals), 180):])

        self.fitted = True
        return self

    def predict(
        self,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> np.ndarray:
        if not self.fitted or self.target is None:
            raise RuntimeError("Model must be fitted before predict() is called.")
        last_val = history.sort_values("date")[self.target].iloc[-1]
        return last_val + self.slope * np.arange(1, horizon + 1)


class SARIMAModel(BaseForecastModel):
    name = "SARIMA"

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 7),
    ):
        super().__init__()
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_res_ = None

    def fit(self, df: pd.DataFrame, target: str, tune: bool = False) -> SARIMAModel:
        self.target = target
        y = df.sort_values("date")[target].astype(float).reset_index(drop=True)

        if tune:
            best_aic = float("inf")
            best_order = self.order
            best_seasonal = self.seasonal_order

            candidate_orders = [
                ((1, 1, 1), (1, 0, 1, 7)),
                ((0, 1, 1), (0, 1, 1, 7)),
                ((1, 1, 2), (1, 0, 1, 7)),
                ((2, 1, 1), (0, 1, 1, 7)),
            ]

            for ord_params, s_ord_params in candidate_orders:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m = SARIMAX(
                            y,
                            order=ord_params,
                            seasonal_order=s_ord_params,
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        ).fit(disp=False, maxiter=50)

                    if m.aic < best_aic:
                        best_aic = m.aic
                        best_order = ord_params
                        best_seasonal = s_ord_params
                except Exception:
                    continue

            self.order = best_order
            self.seasonal_order = best_seasonal

        # Fit selected specification
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_res_ = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=100)

        ins = np.asarray(self.model_res_.fittedvalues)
        self.residuals = y.to_numpy()[-len(ins):] - ins
        if len(self.residuals) > 10:
            self.set_calibration_residuals(self.residuals[-min(len(self.residuals), 180):])

        self.fitted = True
        return self

    def predict(
        self,
        history: pd.DataFrame,
        horizon: int,
        future_exog: pd.DataFrame | None = None,
    ) -> np.ndarray:
        if not self.fitted or self.model_res_ is None:
            raise RuntimeError("Model must be fitted before predict() is called.")
        
        preds = self.model_res_.forecast(steps=horizon)
        return np.asarray(preds, dtype=float)