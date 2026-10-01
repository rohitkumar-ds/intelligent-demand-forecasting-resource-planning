from __future__ import annotations 
from typing import Any 
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from .base import BaseForecastModel
from ..preprocessing import TimeSeriesPreprocessor


class XGBoostForecaster(BaseForecastModel) :
    name = "XGBoost"

    def __init__(self , params : dict[str, Any] | None = None):
        super().__init__()
        self.prep = TimeSeriesPreprocessor()
        self.params = params or {
                    "n_estimators": 300,
                    "max_depth": 4,
                    "learning_rate": 0.03,
                    "subsample": 0.85,
                    "colsample_bytree": 0.9,
                    "objective": "reg:squarederror",
                    "random_state": 42,
                    "n_jobs": 2,
                }
        self.model : XGBRegressor | None = None

    def fit(self,df:pd.DataFrame,target:str,tune:bool = False) :
        self.target = target 
        self.prep.fit(df,target)
        X, y,_ = self.prep.transform(df,target)

        if len(X) < 100:
            raise ValueError("At least 100 usable observations are recommended for XGBoost.")
        
        if tune :
            best_score = float("inf")
            best_params = self.params.copy()
            param_grid = [
                {"max_depth" : 3 , "learning_rate" : 0.03 , "n_estimators" : 250},
                {"max_depth" : 4 , "learning_rate" : 0.03 , "n_estimators" : 300},
                {"max_depth": 5, "learning_rate": 0.04, "n_estimators": 250},
                            
            ]

            tscv = TimeSeriesSplit(n_splits=3)
            for candidate in param_grid :
                scores = []
                for train_idx , val_idx in tscv.split(X) :
                    X_tr , X_val = X.iloc[train_idx] , X.iloc[val_idx]
                    y_tr , y_val = y.iloc[train_idx] , y.iloc[val_idx]


                    test_model= XGBRegressor(**{**self.params,**candidate},random_state = 42 , n_jobs = 2)
                    test_model.fit(X_tr , y_tr,verbose=False)
                    preds = test_model.predict(X_val)
                    scores.append(float(np.mean((y_val.to_numpy()-preds)**2)))

                mean_mse = float(np.mean(scores))
                if mean_mse < best_score :
                    best_score = mean_mse
                    best_params = {**self.params, **candidate}
            
            self.params = best_params
            
        self.model = XGBRegressor(**self.params)
        self.model.fit(X, y, verbose=False)
        
        pred = self.model.predict(X)
        self.residuals = y.to_numpy() - pred
        oof_residuals = []
        tscv = TimeSeriesSplit(n_splits=3)
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            m_cv = XGBRegressor(**self.params)
            m_cv.fit(X_tr, y_tr, verbose=False)
            oof_residuals.extend((y_val.to_numpy() - m_cv.predict(X_val)).tolist())

        self.set_calibration_residuals(oof_residuals)
        self.fitted = True
        return self

    def _future_value(
        self,
        future_exog: pd.DataFrame | None,
        step: int,
        col: str,
        work: pd.DataFrame,
        next_date: pd.Timestamp,
    ):
        if future_exog is not None and step < len(future_exog) and col in future_exog.columns:
            return future_exog[col].iloc[step]

        if col == "promotion":
            return 0.0
        if col == "holiday":
            # Conservative fallback: no holiday unless a future schedule is supplied.
            return 0.0
        if col in work.columns:
            return work[col].iloc[-1]
        return np.nan

    def predict(self,history:pd.DataFrame,horizon:int,future_exog :pd.DataFrame | None = None) :

        if not self.fitted or self.model is None :
            raise RuntimeError("Model must be fitted before predict() is called.")

        work = history.copy()
        work["date"] = pd.to_datetime(work["date"])
        work = work.sort_values("date").reset_index(drop=True)

        predictions: list[float] = []
        for step in range(horizon):
            next_date = pd.Timestamp(work["date"].max()) + pd.Timedelta(days=1)
            feat_df = self.prep.make_features(work, self.target, drop_na=False)
            row = feat_df.iloc[[-1]].copy()

            # Build future-known exogenous values explicitly. Never use another
            # operational target (revenue/inventory/staffing/utilization).
            for col in self.prep.FUTURE_EXOGENOUS:
                if col in self.prep.feature_columns:
                    row[col] = self._future_value(future_exog, step, col, work, next_date)

            X = row[self.prep.feature_columns].astype(float)
            raw_prediction = float(self.model.predict(X)[0])
            prediction = max(raw_prediction, 0.0)
            predictions.append(prediction)

            new_row = {c: np.nan for c in work.columns}
            new_row.update({c: work[c].iloc[-1] for c in work.columns})
            new_row["date"] = next_date
            new_row[self.target] = prediction
            for col in self.prep.FUTURE_EXOGENOUS:
                if col in work.columns:
                    new_row[col] = self._future_value(future_exog, step, col, work, next_date)
            work = pd.concat([work, pd.DataFrame([new_row])], ignore_index=True)

        return np.asarray(predictions, dtype=float)
