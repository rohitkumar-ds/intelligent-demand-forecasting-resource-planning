# Intelligent Demand Forecasting & Resource Planning

A time-series forecasting and decision-support project that predicts product demand and related operational metrics from historical data. It compares statistical, tree-based, and deep-learning forecasting approaches, adds anomaly detection and prediction intervals, and turns forecasts into inventory, staffing, capacity, and risk recommendations through a Streamlit dashboard.

## Why I Built This Project

I wanted to build more than a model that only returns a future number. The main idea was to take a forecasting problem through a more complete ML workflow: validate the data, create time-series features safely, compare different forecasting approaches, save reusable model artifacts, and finally convert predictions into operational decisions.

That led to the progression:

**Forecast demand → understand uncertainty → make an operational recommendation.**

The project uses a synthetic dataset so the complete pipeline can be reproduced without depending on a private business dataset.

## Problem Statement

Demand forecasts are useful, but a forecast by itself does not tell an operations team how much inventory to hold, how much staff may be required, or whether planned capacity is risky.

This project therefore combines forecasting with a small decision engine. It uses the forecast and its prediction interval to estimate safety stock, staffing needs, utilization risk, and an operational action. A what-if interface also lets a user test changes in demand, price, promotions, lead time, capacity, and staffing.

## How the Project Works

```text
Dataset
   ↓
Data Validation
   ↓
Time-Series Preprocessing
   ├── Lag features
   ├── Rolling statistics
   └── Calendar / exogenous features
   ↓
Model Training / Rolling-Origin Evaluation
   ↓
Generated Model Artifacts
   ↓
Forecast + 95% Prediction Interval
   ↓
Decision Engine
   ├── Inventory
   ├── Staffing
   ├── Capacity / Utilization
   └── Risk + Action
   ↓
Streamlit Dashboard
```

### Data and validation

`data/demand_data.csv` contains 2,000 daily observations from 2021-01-01 to 2026-06-23. The generator can recreate the dataset when it is missing and will not overwrite an existing CSV by default.

`src/data_validation.py` checks required columns, valid dates, chronological ordering, duplicate/missing dates, missing values, negative demand/revenue, invalid utilization values, and unrealistic staffing values.

### Preprocessing

`TimeSeriesPreprocessor` creates lag features (1, 7, 14 and 28 days), rolling mean/std features (7, 14 and 28 days), calendar encodings, a trend index, and a whitelist of future-known exogenous variables.

Rolling features are shifted so they use previous observations rather than the current target. Operational outputs such as revenue, inventory, staffing, and utilization are explicitly excluded from the feature set when forecasting another target.

### Forecasting and artifacts

Training is performed by `scripts/train_models.py`, outside the dashboard. Generated model artifacts are stored locally under `artifacts/models/` and are intentionally excluded from the Git repository because trained model binaries can be large and are reproducible from the training pipeline.

The repository keeps the model directory structure using `.gitkeep`, while evaluation results are included under `artifacts/evaluation/`.

The dashboard loads locally generated model artifacts rather than retraining models every time it starts.

## Machine Learning Approach

The primary forecasting target is `product_demand`.

| Model              | Role                                                                          |
| ------------------ | ----------------------------------------------------------------------------- |
| Seasonal Naive     | Simple weekly/seasonal baseline                                               |
| Drift              | Baseline that extrapolates the historical trend                               |
| SARIMA             | Statistical model for autoregressive, differencing, and seasonal behavior     |
| XGBoost            | Main feature-based model using lag, rolling, calendar, and exogenous features |
| LSTM               | Neural sequence model using recent target history                             |
| Temporal Attention | LSTM-based sequence model with multi-head temporal attention                  |
| Prophet            | Optional statistical forecasting model with weekly/yearly seasonality         |

The project compares fundamentally different approaches because a more complex model is not automatically a better forecasting model. The benchmark uses chronological rolling-origin validation rather than random train/test splitting.

XGBoost is also trained separately for five operational targets:

* `product_demand`
* `inventory_requirement`
* `sales_revenue`
* `staffing_requirement`
* `resource_utilization`

## Key Features

* Leakage-aware time-series preprocessing
* Multiple statistical, tree-based, and neural forecasting models
* Chronological train/test splitting
* Rolling-origin backtesting
* RMSE, MAE, MAPE, pinball loss, interval coverage, and interval-width evaluation
* Empirical prediction intervals from model residuals
* Robust rolling median/MAD anomaly detection
* Central model registry and locally generated serialized artifacts
* Forecast-driven inventory, staffing, capacity, and risk recommendations
* Model-driven what-if simulation
* Streamlit dashboard with forecast, benchmark, business-impact, anomaly, and data-quality views
* Automated tests for validation, preprocessing, models, anomaly detection, serialization, and decision logic

## Problems I Faced and How I Solved Them

### Problem: Avoiding leakage in time-series features

Using current or future information when creating rolling features can make a forecasting model look better during evaluation than it really is.

**Solution:** I shifted the target before calculating rolling statistics and used lagged target values. I also kept downstream operational targets out of the feature whitelist. The test suite explicitly checks these behaviors.

**Lesson:** In time-series ML, feature construction is part of the validation strategy, not just preprocessing.

### Problem: Future feature values are not always available

The XGBoost model uses variables such as price, promotion, holiday, temperature, competitor index, and lead time. At real prediction time, some of these values may be unknown.

**Solution:** The recursive prediction code accepts a `future_exog` dataframe. Where a future value is not supplied, it uses explicit fallback behavior instead of silently reading another operational target. During evaluation, known values from the test period are passed separately.

**Lesson:** A forecasting model must distinguish between historical features and variables that will genuinely be available at prediction time.

### Problem: Comparing models without randomizing time

A random train/test split would allow future observations to influence the training set.

**Solution:** The project uses chronological splitting for training and rolling-origin backtesting for model comparison. This preserves the direction of time during evaluation.

**Lesson:** Validation methodology can be as important as the choice of algorithm.

### Problem: Forecasts need uncertainty, not only a point estimate

Operational planning based only on one predicted value can hide forecast uncertainty.

**Solution:** The common base model stores calibration residuals and derives empirical prediction intervals. The dashboard exposes a 95% interval, while evaluation measures interval coverage, pinball loss, and average width.

**Lesson:** Forecast accuracy and forecast uncertainty are separate things that both matter for decisions.

### Problem: Turning forecasts into something actionable

A model prediction does not directly answer questions such as how much inventory to plan or whether capacity is sufficient.

**Solution:** I added a `DecisionEngine` that converts the forecast and interval into safety stock, recommended inventory, staffing requirements, utilization, risk level, and an action message. The same engine powers what-if simulations.

**Lesson:** For applied ML, the useful output is often the decision supported by the prediction rather than the prediction itself.

## Business / Industry Use

The current project demonstrates a forecasting and operational decision-support workflow using synthetic data. A company could adapt the same architecture for:

* Inventory and replenishment planning
* Workforce and shift planning
* Revenue planning
* Capacity and resource allocation
* Identifying unusual demand periods
* Reducing understock/overstock risk

The project is **not presented as a deployed production system**. A production implementation would need real data contracts, reliable future feature feeds, monitoring, access control, auditability, and business-specific optimization rules.

## Tech Stack

| Category            | Technology                             |
| ------------------- | -------------------------------------- |
| Language            | Python                                 |
| Data Processing     | Pandas, NumPy                          |
| Machine Learning    | XGBoost, scikit-learn                  |
| Deep Learning       | PyTorch                                |
| Time Series         | Statsmodels SARIMA, Prophet (optional) |
| Visualization / UI  | Streamlit, Plotly                      |
| Model Serialization | Joblib, PyTorch                        |
| Testing             | Pytest                                 |

## Project Structure

```text
├── app/
│   └── dashboard.py
├── artifacts/
│   ├── evaluation/
│   │   └── model_metrics.csv
│   └── models/
│       └── .gitkeep
├── data/
│   ├── demand_data.csv
│   └── generator.py
├── scripts/
│   ├── train_models.py
│   └── evaluate_models.py
├── src/
│   ├── anomaly.py
│   ├── config.py
│   ├── data_validation.py
│   ├── decision_engine.py
│   ├── evaluation.py
│   ├── model_registry.py
│   ├── preprocessing.py
│   └── models/
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

* `data/` — dataset and reproducible synthetic-data generator.
* `src/` — validation, preprocessing, forecasting models, evaluation, registry, and decision logic.
* `scripts/` — offline training and benchmarking entry points.
* `artifacts/evaluation/` — evaluation results included with the repository.
* `artifacts/models/` — local directory for generated model artifacts; trained binaries are excluded from Git.
* `app/` — Streamlit dashboard.
* `tests/` — automated tests for core components.

## How to Run

Create an environment and install the pinned dependencies:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

The repository already contains `data/demand_data.csv`. If the dataset is missing, the dashboard can generate it automatically; alternatively, run:

```bash
python data/generator.py
```

### Train the models

Generated model artifacts are intentionally excluded from GitHub. After cloning the repository, train the models locally:

```bash
python scripts/train_models.py
```

Optional training-time tuning:

```bash
python scripts/train_models.py --tune
```

### Run model evaluation

```bash
python scripts/evaluate_models.py
```

Optional evaluation tuning:

```bash
python scripts/evaluate_models.py --tune
```

### Launch the dashboard

```bash
streamlit run app/dashboard.py
```

### Run the automated tests

```bash
pytest
```

The expected workflow is:

```text
Install dependencies
       ↓
Generate / load dataset
       ↓
Train models
       ↓
Evaluate models
       ↓
Launch dashboard
       ↓
Run tests
```

## Future Enhancements

1. Replace the synthetic dataset with a real data source and production data contract.
2. Add automated model/data drift monitoring and scheduled retraining.
3. Improve uncertainty estimation with conformal prediction or direct probabilistic/quantile forecasting.
4. Add automated hyperparameter optimization and experiment tracking.
5. Integrate the decision engine with real inventory, workforce, and capacity constraints through an API or database-backed workflow.

## License

License information can be added based on the intended repository distribution.
