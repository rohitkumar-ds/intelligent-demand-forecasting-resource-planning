from __future__ import annotations 
from dataclasses import dataclass
import numpy as np 
import pandas as pd

@dataclass
class FeatureConfig :
    lags = (1,7,14,28)
    rolling_window = (7,14,28 )

class TimeSeriesPreprocessor :
    FUTURE_EXOGENOUS = (
            "price",
            "promotion",
            "holiday",
            "temperature",
            "competitor_index",
            "lead_time_days",
        )
    
    CALENDAR_FEATURES = (
            "dow_sin",
            "dow_cos",
            "doy_sin",
            "doy_cos",
            "is_weekend",
        )
    def __init__(self,config : FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self.feature_columns = []
        self.target : str | None = None

    def make_features(self,df : pd.DataFrame , target : str , drop_na :bool = True) :

        if "date" not in df.columns or target not in df.columns:
            raise ValueError(f"DataFrame must contain 'date' and '{target}'.")

        x = df.copy()
        x["date"] = pd.to_datetime(x["date"])
        x = x.sort_values("date").reset_index(drop=True)       

        t = np.arange(len(x))
        dow = x["date"].dt.dayofweek
        doy = x["date"].dt.dayofyear

        x["dow_sin"] = np.sin(2*np.pi*dow/7.0)
        x["dow_cos"] = np.cos(2*np.pi * dow /7.0)
        x["doy_sin"] = np.sin(2*np.pi * doy /365.25)
        x["doy_cos"] = np.cos(2* np.pi * doy / 365.25)

        for d in range(7) :
            x[f"is_dow_{d}"] = (dow==d).astype(float)

        x["is_weekend"] = dow.isin([5,6]).astype(float)

        x["trend_index"] = t

        for lag in self.config.lags :
            x[f"lag_{lag}"] = x[target].shift(lag)

        s = x[target].shift(1)
        for w in self.config.rolling_window :
            
            x[f"rolling_mean_{w}"] = s.rolling(w).mean()
            x[f"rolling_std_{w}"] = s.rolling(w).std()

        # Explicitly whitelist exogenous inputs. Other target/business outputs
        # are intentionally excluded even if they are highly correlated.
        for col in self.FUTURE_EXOGENOUS:
            if col in x.columns:
                x[col] = pd.to_numeric(x[col], errors="coerce")
        x = x.replace([np.inf , -np.inf],np.nan)
        if drop_na :
            return x.dropna().reset_index(drop=True)
        return x

    def fit(self,df:pd.DataFrame,target :str) :
        self.target = target
        f = self.make_features(df,target,drop_na=True)
        self.feature_columns = [ c for c in f.columns
                    if c in set(self.CALENDAR_FEATURES)
                    or c.startswith("lag_")
                    or c.startswith("rolling_")
                    or c in self.FUTURE_EXOGENOUS]
        return f

    def transform(self, df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        f = self.make_features(df, target, drop_na=True)
        missing = [c for c in self.feature_columns if c not in f.columns]
        if missing:
            raise ValueError(f"Missing expected feature columns: {missing}")
        return f[self.feature_columns].astype(float), f[target].astype(float), f["date"]

    
def chronological_split(df:pd.DataFrame , test_fraction =0.2) :
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"])
    x = x.sort_values("date").reset_index(drop=True)
    cut = int(len(x)*(1.0 - test_fraction))
    return x.iloc[:cut].copy() , x.iloc[cut:].copy()