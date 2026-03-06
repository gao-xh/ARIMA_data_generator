# Forward Model Demonstration (Training and Validation)
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Add src to path if needed (though running from root should work if __init__ exists)
sys.path.append(str(Path(__file__).parent))

from src.models import ImprovedARIMA
from src.preprocessing import preprocess_for_model
from src.evaluation import calculate_mape, calculate_rmse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")
RESULT_DIR = Path("output")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

def train_and_evaluate(drug_code: str = None, clinic_id: str = "Clinic_A"):
    """
    Simulates the forward model pipeline using existing data.
    """
    # 1. Load Data
    data_path = DATA_DIR / "synthetic_sales.csv"
    if not data_path.exists():
        logger.error(f"Dataset not found at {data_path}. Please run generate_dataset.py first.")
        return

    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Check if data exists
    if df.empty:
        logger.error("Dataset is empty.")
        return
        
    # Select a Target Drug (High Volatility ideally for demonstration)
    if not drug_code:
        # Pick one automatically (e.g., first one)
        drug_code = df['药品编码'].iloc[0]
        
    logger.info(f"Selected Target Drug: {drug_code} (Clinic: {clinic_id})")
    
    # 2. Preprocess
    # Filter by clinic and drug
    target_df = df[(df['药品编码'] == drug_code) & (df['所属诊所'] == clinic_id)].copy()
    
    if target_df.empty:
        logger.error(f"No data found for Drug {drug_code} in {clinic_id}")
        return
        
    # Standard Preprocessing (Sort, Fill, Index)
    # Note: synthetic data is already clean, but we apply standardization anyway
    target_df['Date'] = pd.to_datetime(target_df['Date']) # Ensure Date type
    target_df = target_df.sort_values('Date')
    
    # 3. Train/Test Split (80/20 chronological)
    train_size = int(len(target_df) * 0.8)
    train_df = target_df.iloc[:train_size]
    test_df = target_df.iloc[train_size:]
    
    logger.info(f"Train Size: {len(train_df)} | Test Size: {len(test_df)}")
    
    # 4. Initialize Model
    # Need Drug Info (Simulated metadata or real metadata)
    # For now, we infer or mock metadata since we might not have the master list in memory easily here
    # Assume High Volatility for demonstration if we picked a high vol drug
    # In real app, load from drug_info.xls
    
    mock_drug_info = pd.Series({
        '波动区间分类': '高波动', # Hardcoded for demo to test complex path
        '效期（月）': 12,
        'CV': 0.6
    })
    
    model = ImprovedARIMA(drug_info=mock_drug_info)
    
    # 5. Train
    # Map column names if needed. Detailed implementation relies on constants.
    # Our generated data uses English or Chinese keys? Let's check constants.
    # Generator uses: '所属诊所', '药品编码', '当日销量（单位）'...
    # Model expects: Sales, Date... and Exog (Temp, Flu). 
    # Generator output has: 'Temp', 'Flu'. Model expects: '平均气温', '流感ILI%'?
    # We need to rename for compatibility or update model config.
    # Let's rename here for the model's standard interface.
    
    rename_map = {
        'Date': '日期', # Model internal uses constants.COL_DATE default '日期'
        'Sales': '当日销量（单位）',
        'Temp': '平均气温',
        'Flu': '流感ILI%'
    }
    # Check if columns are English (from generator) or Chinese (from source)
    # Generator.py output keys: C.COL_DATE ('日期'?), C.COL_CLINIC... 
    # Actually generator uses constants which are Chinese keys!
    # Let's verify constants.py content.
    # C.COL_DATE = '日期', C.EXT_TEMP = '平均气温'.
    # So the dataframe columns ARE CHINESE if generator uses constants.class
    # Wait, generator code: record = { C.COL_DATE: date, ... 'Temp': temp, 'Flu': flu }
    # It mixed constants and hardcoded strings ('Temp', 'Flu').
    # We need to align them.
    
    # Align columns for model
    # Generator output (clean synthetic csv) might have specific headers.
    # Let's assume we align them to what ImprovedARIMA expects (C.EXT_TEMP, C.EXT_FLU).
    
    if 'Temp' in train_df.columns:
        train_df = train_df.rename(columns={'Temp': '平均气温', 'Flu': '流感ILI%', 'Sales': '当日销量（单位）'})
        test_df = test_df.rename(columns={'Temp': '平均气温', 'Flu': '流感ILI%', 'Sales': '当日销量（单位）'})
        
    logger.info(f"Training Columns: {train_df.columns.tolist()}")
    
    model.train(train_df)
    
    # 6. Predict
    future_exog = test_df[['日期', '平均气温', '流感ILI%']] # Pass required cols
    forecast = model.predict(steps=len(test_df), future_exog_df=future_exog)
    
    # 7. Evaluate
    actuals = test_df['当日销量（单位）'].values
    mape = calculate_mape(actuals, forecast)
    rmse = calculate_rmse(actuals, forecast)
    
    logger.info(f"Evaluation Results (Drug: {drug_code}):")
    logger.info(f"MAPE: {mape:.2f}%")
    logger.info(f"RMSE: {rmse:.2f}")

if __name__ == "__main__":
    train_and_evaluate()
