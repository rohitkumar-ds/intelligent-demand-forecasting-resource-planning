from pathlib import Path
import numpy as np
import pandas as pd
from src.anomaly import detect_anomalies
from src.models.statistical import DriftModel, SeasonalNaiveModel
from src.models.tree_based import XGBoostForecaster


def test_multi_target_anomaly_detection(sample_df):
    for target in ["product_demand", "sales_revenue", "resource_utilization"]:
        res = detect_anomalies(sample_df, target=target)
        assert "anomaly" in res.columns
        assert "anomaly_score" in res.columns
        assert res["anomaly"].dtype == bool


def test_xgboost_fit_predict_and_save(sample_df, tmp_path):
    model = XGBoostForecaster()
    train = sample_df.iloc[:200]
    model.fit(train, "product_demand", tune=False)

    res = model.forecast(train, horizon=7, confidence=0.95)
    assert len(res.point) == 7
    assert np.all(res.lower <= res.point)
    assert np.all(res.point <= res.upper)

    # Test polymorphic serialization
    save_file = tmp_path / "xgb_test.joblib"
    model.save(save_file)
    loaded_model = XGBoostForecaster.load(save_file)

    res_loaded = loaded_model.forecast(train, horizon=7, confidence=0.95)
    np.testing.assert_allclose(res.point, res_loaded.point)