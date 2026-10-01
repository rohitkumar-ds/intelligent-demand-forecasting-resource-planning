"""Forecast-to-action decision engine with risk classification, cumulative planning, and model-driven what-if simulation."""
from __future__ import annotations
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class OperationalRecommendation:
    expected_demand: float
    safety_stock: float
    recommended_inventory: float
    required_staff: int
    additional_staff: int
    required_capacity: float
    utilization: float
    risk: str
    action: str


class DecisionEngine:
    def __init__(self, units_per_staff_hour: float = 18.0, shift_hours: float = 8.0):
        self.units_per_staff_hour = units_per_staff_hour
        self.shift_hours = shift_hours

    @staticmethod
    def risk(point: float, lower: float, upper: float) -> tuple[str, float]:
        """Classify operational uncertainty risk based on prediction interval width relative to point estimate."""
        point_val = max(abs(point), 1e-9)
        width = max(upper - lower, 0.0) / point_val * 100.0
        if width < 10.0:
            return "LOW", width
        if width <= 20.0:
            return "MEDIUM", width
        return "HIGH", width

    def recommend(
        self,
        point: float | np.ndarray,
        lower: float | np.ndarray,
        upper: float | np.ndarray,
        lead_time_days: int = 7,
        service_level: float = 0.95,
        labor_buffer: float = 0.10,
        capacity: float = 1000.0,
        current_staff: int = 35,
    ) -> OperationalRecommendation:
        """
        Calculates inventory, staffing headcount, capacity utilization, and action guidance.
        Supports both single-day floats and multi-step forecast arrays.
        """
        # If arrays are passed, calculate cumulative lead-time demand
        if isinstance(point, (np.ndarray, list)):
            horizon_len = min(len(point), max(int(lead_time_days), 1))
            lead_time_demand = float(np.sum(point[:horizon_len]))
            first_day_point = float(point[0])
            first_day_lower = float(lower[0])
            first_day_upper = float(upper[0])
            sigma_day = max((first_day_upper - first_day_lower) / (2.0 * 1.96), 1.0)
        else:
            first_day_point = float(point)
            first_day_lower = float(lower)
            first_day_upper = float(upper)
            lead_time_demand = first_day_point * lead_time_days
            sigma_day = max((first_day_upper - first_day_lower) / (2.0 * 1.96), 1.0)

        # Safety Stock formula: z * sigma_L
        z = NormalDist().inv_cdf(min(max(service_level, 0.50), 0.999))
        safety_stock = max(0.0, float(z * sigma_day * np.sqrt(max(lead_time_days, 1))))
        recommended_inventory = max(lead_time_demand + safety_stock, first_day_point)

        # Labor headcount requirement based on daily shift capacity
        daily_units_per_worker = self.units_per_staff_hour * self.shift_hours
        required_staff = int(
            np.ceil((first_day_point / max(daily_units_per_worker, 1e-6)) * (1.0 + labor_buffer))
        )
        additional_staff = max(required_staff - current_staff, 0)

        # Capacity & Utilization
        required_capacity = first_day_point / 0.85
        utilization = (first_day_point / max(capacity, 1.0)) * 100.0
        risk_level, _ = self.risk(first_day_point, first_day_lower, first_day_upper)

        # Action rule logic
        if utilization > 95.0:
            action = "HIGH CAPACITY RISK: Planned load exceeds 95% threshold. Add capacity or reschedule orders."
        elif utilization > 85.0:
            action = "MEDIUM CAPACITY RISK: Planned load exceeds 85% threshold. Add capacity buffer and monitor upper bounds."
        elif additional_staff > 0:
            action = f"Add {additional_staff} staff-equivalent headcount and maintain inventory buffer."
        else:
            action = "Maintain planned operations with calculated safety stock and labor buffer."

        return OperationalRecommendation(
            expected_demand=first_day_point,
            safety_stock=float(safety_stock),
            recommended_inventory=float(recommended_inventory),
            required_staff=required_staff,
            additional_staff=additional_staff,
            required_capacity=float(required_capacity),
            utilization=float(utilization),
            risk=risk_level,
            action=action,
        )

    def what_if(
        self,
        base: dict[str, Any],
        demand_growth: float = 0.0,
        price_change: float = 0.0,
        promotion: bool = False,
        lead_time_days: int = 7,
        service_level: float = 0.95,
        labor_buffer: float = 0.10,
        capacity: float = 1000.0,
        current_staff: int = 35,
        model: Any = None,
        history_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Executes scenario simulation.
        If a trained model and history DataFrame are provided, it simulates directly through the model's feature space.
        """
        base_price = float(base.get("price", 30.0))
        sim_price = max(base_price * (1.0 + price_change), 1.0)

        if model is not None and history_df is not None and hasattr(model, "predict"):
            # Build future exogenous schedule
            future_exog = pd.DataFrame({
                "price": [sim_price],
                "promotion": [1 if promotion else 0],
            })
            pred = model.predict(history_df, horizon=1, future_exog=future_exog)
            sim_demand = float(pred[0]) * (1.0 + demand_growth)
        else:
            # Fallback estimation incorporating elasticity proxy
            base_demand = float(base.get("demand", 1000.0))
            price_elasticity = -0.65
            elasticity_multiplier = 1.0 + (price_elasticity * price_change)
            promo_multiplier = 1.20 if promotion else 1.0
            sim_demand = max(base_demand * (1.0 + demand_growth) * elasticity_multiplier * promo_multiplier, 0.0)

        sim_revenue = sim_demand * sim_price
        rec = self.recommend(
            point=sim_demand,
            lower=sim_demand * 0.90,
            upper=sim_demand * 1.10,
            lead_time_days=lead_time_days,
            service_level=service_level,
            labor_buffer=labor_buffer,
            capacity=capacity,
            current_staff=current_staff,
        )

        return {
            "Demand": round(sim_demand, 1),
            "Revenue": round(sim_revenue, 2),
            "Inventory": round(rec.recommended_inventory, 0),
            "Staffing Hours": int(rec.required_staff * self.shift_hours),
            "Utilization %": round(rec.utilization, 2),
            "Risk": rec.risk,
            "Additional Staff": rec.additional_staff,
            "Action": rec.action,
        }