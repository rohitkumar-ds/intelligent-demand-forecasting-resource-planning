import sys
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_PATH


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, parse_dates=["date"])