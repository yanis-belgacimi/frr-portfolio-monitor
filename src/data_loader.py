import pandas as pd

REQUIRED_COLUMNS = ["Stocks", "IG Bonds", "High Yield Bonds", "Emergents"]


def load_portfolio_data(filepath : str) -> pd.DataFrame:
    """
    Load portfolio data by converting it into clean 
    DataFrame with required columns.
    """
    # Load CSV with pandas
    df_data = pd.read_csv(filepath, index_col=0, parse_dates = True)

    # Condition on datetime type and columns
    if not isinstance(df_data.index, pd.DatetimeIndex):
        raise ValueError("Type Error : First column must be datetime")
    
    missing = set(REQUIRED_COLUMNS) - set(df_data.columns)
    if missing:
        raise ValueError(f"Missing required columns {missing}")
    
    return df_data