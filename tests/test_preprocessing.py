import numpy as np
import pandas as pd
from src.preprocessing import TimeSeriesPreprocessor, chronological_split


def test_features_are_leakage_safe(sample_df):
    p = TimeSeriesPreprocessor()
    f = p.fit(sample_df.iloc[:1500], "product_demand")
    X, y, dates = p.transform(sample_df.iloc[:1500], "product_demand")

    assert "lag_1" in p.feature_columns
    assert "rolling_mean_7" in p.feature_columns
    assert dates.is_monotonic_increasing
    assert len(X) == len(y) == len(dates)


def test_preprocessing_drop_na_toggle(sample_df):
    p = TimeSeriesPreprocessor()
    # When drop_na=False, rows are preserved for single-step recursive inference
    f_all = p.make_features(sample_df.iloc[:50], "product_demand", drop_na=False)
    assert len(f_all) == 50


def test_chronological_split(sample_df):
    train, test = chronological_split(sample_df, test_fraction=0.2)
    assert len(train) + len(test) == len(sample_df)
    assert train["date"].max() < test["date"].min()


def test_downstream_operational_targets_are_not_features(sample_df):
    p = TimeSeriesPreprocessor()
    p.fit(sample_df.iloc[:1500], "product_demand")
    forbidden = {
        "sales_revenue",
        "inventory_requirement",
        "staffing_requirement",
        "resource_utilization",
    }
    assert forbidden.isdisjoint(p.feature_columns)


def test_rolling_features_use_only_previous_observations(sample_df):
    p = TimeSeriesPreprocessor()
    features = p.make_features(sample_df.iloc[:50], "product_demand", drop_na=False)
    idx = 20
    expected = sample_df.iloc[:idx]["product_demand"].tail(7).mean()
    assert np.isclose(features.loc[idx, "rolling_mean_7"], expected)