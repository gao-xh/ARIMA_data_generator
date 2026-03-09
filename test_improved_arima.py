
import pandas as pd
import numpy as np
import sys
import os

# Mock paths to allow imports
sys.path.append(os.getcwd())

from src.models.improved_arima import ImprovedARIMA
from src.core import constants as C

def test_arima():
    print("Testing ImprovedARIMA...")
    
    # Mock Data
    dates = pd.date_range(start='2024-01-01', end='2025-12-31', freq='D')
    n = len(dates)
    
    df = pd.DataFrame({
        C.COL_DATE: dates,
        C.COL_SALES: np.random.poisson(10, n) + np.sin(np.linspace(0, 10, n)) * 5,
        '平均气温': np.random.normal(20, 5, n),
        'Temp_Std': np.random.normal(0, 1, n),
        'Rain_Log': np.random.normal(0, 1, n),
        '流感发病率': np.random.uniform(0, 0.1, n),
        C.EXT_FLU: np.random.uniform(0, 0.1, n)
    })
    
    # Mock Drug Info
    drug_info = pd.Series({
        '波动区间分类': C.FLUC_MED, # Mid Volatility
        '效期（月）': 12,
        'CV': 0.3
    })
    
    # Split
    split_date = pd.Timestamp('2025-09-01')
    train_df = df[df[C.COL_DATE] < split_date].copy()
    test_df = df[df[C.COL_DATE] >= split_date].copy()
    
    print(f"Train Size: {len(train_df)}")
    print(f"Test Size: {len(test_df)}")
    
    try:
        model = ImprovedARIMA(drug_info)
        print("Model initialized.")
        
        print("Training...")
        model.train(train_df)
        print("Training complete.")
        
        print("Predicting...")
        steps = len(test_df)
        preds = model.predict(steps, future_exog_df=test_df)
        
        print(f"Prediction type: {type(preds)}")
        if isinstance(preds, list):
            print(f"Prediction length: {len(preds)}")
            print(f"First 5 preds: {preds[:5]}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_arima()
