from __future__ import annotations
import numpy as np
import pandas as pd
from .base import BaseForecastModel


class ProphetForecaster(BaseForecastModel):
    name = "Prophet"

    def fit(self, df: pd.DataFrame, target: str, tune: bool = False) -> ProphetForecaster:
        from prophet import Prophet

        self.target = target
        x = df[["date", target]].rename(columns={"date": "ds", target: "y"})

        # If tune is requested, adjust changepoint prior scale
        changepoint_prior = 0.05 if not tune else 0.1

        self.model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=changepoint_prior,
        )
        self.model.fit(x)

        fitted = self.model.predict(x)["yhat"].to_numpy()
        self.residuals = x["y"].to_numpy() - fitted

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
        if not self.fitted or self.model is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        future = self.model.make_future_dataframe(periods=horizon, freq="D", include_history=False)
        forecast_df = self.model.predict(future)
        return forecast_df["yhat"].to_numpy()
