import pandas as pd
import numpy as np
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
    cols_to_fill = ['平均气温', '流感ILI%', '节假日', '平均降水量', '流感发病率']
    for c in cols_to_fill:
        if c in df.columns:
            df[c] = df[c].interpolate(method='linear').fillna(method='bfill')

    # --- Thesis Requirement: Advanced Data Transformations ---
    # 1. Log Transform Rainfall: ln(Rain + 1) to handle skewness
    if '平均降水量' in df.columns:
        df['Rain_Log'] = np.log1p(df['平均降水量'])
    else:
        df['Rain_Log'] = 0.0
        
    # 2. Standardize Temperature: (T - mean) / std to remove scale
    if '平均气温' in df.columns:
        t_mean = df['平均气温'].mean()
        t_std = df['平均气温'].std()
        if t_std != 0:
            df['Temp_Std'] = (df['平均气温'] - t_mean) / t_std
        else:
            df['Temp_Std'] = 0.0
    else:
        df['Temp_Std'] = 0.0
            
    return df
