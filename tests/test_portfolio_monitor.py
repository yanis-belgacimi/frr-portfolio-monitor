import pytest
import pandas as pd
from src.portfolio_monitor import PortfolioMonitor

def test_mismatch_columns():
    test_target_weights = {
        'Government_Bonds': 0.35,
        'IG_Bonds': 0.20,
        'Equities': 0.30,
        'High_Yield_Bonds': 0.10,
        'Emerging_Markets': 0.05
    }
    # Columns IG must raise an ValueError
    test_actual_weights = pd.Series(
    {
        'Government_Bonds': 0.35,
        'IG': 0.20,
        'Equities': 0.30,
        'High_Yield_Bonds': 0.10,
        'Emerging_Markets': 0.05
    })
    with pytest.raises(ValueError):
        portmo = PortfolioMonitor(test_target_weights, 0.05).compute_drift(test_actual_weights)
    

