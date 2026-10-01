from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "demand_data.csv"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
EVAL_DIR = ARTIFACT_DIR / "evaluation"
TARGETS = [
    "product_demand", "inventory_requirement", "sales_revenue",
    "staffing_requirement", "resource_utilization"
]
REQUIRED_COLUMNS = [
    "date", "product_demand", "inventory_requirement", "sales_revenue",
    "staffing_requirement", "resource_utilization", "price", "promotion",
    "holiday", "temperature", "competitor_index", "lead_time_days"
]
