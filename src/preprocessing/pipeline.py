import pandas as pd
from typing import Dict, Any

def preprocess_for_model(sales_df: pd.DataFrame, drug_code: str) -> pd.DataFrame:
    """
    Standard preprocessing for model training.
    Filters by drug code, ensures datetime index, fills missing values.
    
    Args:
        sales_df: The raw merged dataset (sales + external factors).
        drug_code: The specific drug to filter for.
        
    Returns:
        pd.DataFrame: Ready for ARIMA (Date index, no missing values).
    """
    # Filter
    df = sales_df[sales_df['药品编码'] == drug_code].copy()
    
    # Sort
    if '日期' in df.columns:
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期')
        df.set_index('日期', inplace=True)
        
    # Fill Missing Sales
    # Resample to ensure daily frequency if data is sparse
    # (Assuming daily data is expected)
    df = df.asfreq('D')
    
    # Forward fill or 0 fill for sales?
    # Sales should be 0 if missing usually, but inventory carries over.
    # Exog factors (Temp, Flu) should be interpolated or ffilled.
    
    if '当日销量（单位）' in df.columns:
        df['当日销量（单位）'] = df['当日销量（单位）'].fillna(0)
        
    # Fill Exog Factors
    cols_to_fill = ['平均气温', '流感ILI%', '节假日']
    for c in cols_to_fill:
        if c in df.columns:
            df[c] = df[c].interpolate(method='linear').fillna(method='bfill')
            
    return df
