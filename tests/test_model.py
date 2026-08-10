import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import dcf_valuation
import pandas as pd

def test_dcf_returns_positive_value():
    forecast = pd.DataFrame({"fcff": [100, 110, 120, 130, 140]}, index=[1,2,3,4,5])
    result = dcf_valuation(forecast, 0.08, 0.025, 500, 100, 10)
    assert result["intrinsic_price"] > 0
