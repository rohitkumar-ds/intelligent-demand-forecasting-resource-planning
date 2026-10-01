from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd 
from .config import REQUIRED_COLUMNS

@dataclass
class ValidationReport :
    rows : int 
    date_start : str 
    date_end : str
    missing_values : int
    duplicate_dates: int
    missing_dates: int
    negative_demand : int
    negative_revenue : int
    invalid_utilization : int
    unrealistic_staffing : int
    datetime_valid : bool
    chronological : bool
    passed : bool

def validate_dataset(df : pd.DataFrame) :
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols :
        raise ValueError(f"Missing required columns : {missing_cols}")

    x = df.copy()
    parsed_dates = pd.to_datetime(x["date"],errors="coerce")
    datetime_valid = bool(parsed_dates.notna().all() and len(parsed_dates) > 0)
    x["date"] = parsed_dates

    chronological = bool(datetime_valid and x["date"].is_monotonic_increasing)
    duplicate_dates = int(x["date"].duplicated().sum())

    missing_dates = 0 
    if datetime_valid and len(x) > 1 :
        expected = pd.date_range(x["date"].min() , x["date"].max() , freq = "D")
        missing_dates = int(len(expected.difference(pd.DatetimeIndex(x["date"]))))

    missing_values = int(x.isna().sum().sum())
    negative_demand = int((x["product_demand"] < 0).sum())
    negative_revenue = int((x["sales_revenue"] < 0).sum())
    invalid_utilization = int(((x["resource_utilization"] < 0) | (x["resource_utilization"] > 100)).sum())
    unrealistic_staffing = int(((x["staffing_requirement"]<0)| (x["staffing_requirement"] > 1000)).sum())

    passed = all([
        missing_values == 0 ,
        duplicate_dates == 0 ,
        missing_dates == 0 ,
        negative_demand == 0 ,
        negative_revenue == 0,
        invalid_utilization == 0,
        unrealistic_staffing == 0,
        datetime_valid,
        chronological,
    ])

    date_start = str(x["date"].min().date()) if datetime_valid else "N/A"
    date_end = str(x["date"].max().date()) if datetime_valid else "N/A"

    return ValidationReport(
        rows = len(x),
        date_start=date_start ,
        date_end=date_end ,
        missing_values= missing_values,
        duplicate_dates = duplicate_dates ,
        missing_dates = missing_dates,
        negative_demand= negative_demand ,
        negative_revenue=negative_revenue,
        invalid_utilization=invalid_utilization,
        unrealistic_staffing= unrealistic_staffing,
        datetime_valid=datetime_valid,
        chronological=chronological,
        passed=passed,
    )

def validate_file(path : str | Path) :
    df = pd.read_csv(path)
    return validate_dataset(df)