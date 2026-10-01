import pandas as pd
from src.data_validation import validate_dataset


def test_dataset_passes(sample_df):
    report = validate_dataset(sample_df)
    assert report.passed
    assert report.datetime_valid
    assert report.chronological


def test_duplicate_date_fails(sample_df):
    df_dup = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    report = validate_dataset(df_dup)
    assert not report.passed
    assert report.duplicate_dates > 0


def test_non_chronological_fails(sample_df):
    # Shuffling dates must fail the strict monotonicity check
    df_shuffled = sample_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    report = validate_dataset(df_shuffled)
    assert not report.passed
    assert not report.chronological


def test_invalid_utilization_fails(sample_df):
    df_bad = sample_df.copy()
    df_bad.loc[0, "resource_utilization"] = 150.0  # Outside 0-100%
    report = validate_dataset(df_bad)
    assert not report.passed
    assert report.invalid_utilization > 0
