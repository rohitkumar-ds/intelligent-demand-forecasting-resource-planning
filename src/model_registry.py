"""Central registry for forecasting model factories."""
from .models.statistical import SeasonalNaiveModel, DriftModel, SARIMAModel
from .models.tree_based import XGBoostForecaster
from .models.deep_learning import LSTMForecaster, AttentionForecaster

MODEL_REGISTRY = {
    "Seasonal Naive": SeasonalNaiveModel,
    "Drift": DriftModel,
    "SARIMA": SARIMAModel,
    "XGBoost": XGBoostForecaster,
    "LSTM": LSTMForecaster,
    "Temporal Attention": AttentionForecaster,
}
try:
    from .models.prophet_model import ProphetForecaster
    MODEL_REGISTRY["Prophet"] = ProphetForecaster
except Exception:
    pass
