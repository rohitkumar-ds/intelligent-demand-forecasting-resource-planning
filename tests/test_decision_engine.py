import numpy as np
import pandas as pd
from src.decision_engine import DecisionEngine


def test_recommendation_single_point():
    eng = DecisionEngine()
    r = eng.recommend(point=1000.0, lower=900.0, upper=1150.0, lead_time_days=7)
    assert r.safety_stock > 0
    assert r.recommended_inventory >= 1000.0 * 7 + r.safety_stock
    assert r.required_staff > 0
    assert r.risk in {"LOW", "MEDIUM", "HIGH"}


def test_recommendation_cumulative_array():
    eng = DecisionEngine()
    # 7-day forecast array
    forecast_points = np.array([1000.0, 1050.0, 980.0, 1020.0, 1100.0, 1010.0, 990.0])
    lower = forecast_points * 0.9
    upper = forecast_points * 1.1

    r = eng.recommend(point=forecast_points, lower=lower, upper=upper, lead_time_days=7)
    expected_cumulative = float(np.sum(forecast_points))
    assert r.recommended_inventory >= expected_cumulative


def test_what_if_simulation():
    eng = DecisionEngine()
    base = {"demand": 1000.0, "price": 30.0}
    sim = eng.what_if(base, demand_growth=0.10, price_change=-0.05, promotion=True)

    assert "Demand" in sim
    assert "Revenue" in sim
    assert "Staffing Hours" in sim
    assert sim["Demand"] > 1000.0
