import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import ImprovedARIMA
from sklearn.metrics import mean_absolute_percentage_error
from datetime import timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def main():
    # ... (existing code up to model init)
    data_path = Path("data/processed/synthetic_sales.csv")
    if not data_path.exists():
        logger.error("Dataset not found. Run generate_dataset.py first.")
        return

    logger.info("Loading dataset...")
    df = pd.read_csv(data_path)
    
    # 2. Select a Drug
    # Let's find a drug with 'High Volatility' (or just pick one with good volume)
    # Since we don't have explicit volatility column in CSV, we can infer or just pick one.
    # The simulator assigned 'High' to some random drugs.
    # Let's just pick the most sold drug for demonstration.
    top_drug = df.groupby('药品名称')['当日销量（单位）'].sum().idxmax()
    drug_code = df[df['药品名称'] == top_drug]['药品编码'].iloc[0]
    
    logger.info(f"Selected Drug: {top_drug} (Code: {drug_code})")
    
    # Filter for one clinic and one drug
    clinic = "Clinic_A"
    subset = df[(df['所属诊所'] == clinic) & (df['药品编码'] == drug_code)].copy()
    subset['Date'] = pd.to_datetime(subset['日期'])
    subset = subset.rename(columns={'当日销量（单位）': 'Sales'})
    subset = subset.sort_values('Date').reset_index(drop=True)
    
    # Prepare Drug Info Mock (In real app, load from Drug Info Table)
    # We will assume it's High Volatility because it's top selling (just for demo params)
    drug_info_mock = {
        '药品名称': top_drug,
        '波动区间分类': '高波动', # Force High Volatility to test full ARIMAX
        '效期（月）': 24
    }
    
    # 3. Train/Test Split
    # Train: 2024-01-01 to 2025-08-31
    # Test: 2025-09-01 to 2025-12-31
    split_date = pd.to_datetime("2025-09-01")
    train_data = subset[subset['Date'] < split_date]
    test_data = subset[subset['Date'] >= split_date]
    
    logger.info(f"Train Size: {len(train_data)} days | Test Size: {len(test_data)} days")
    
    # 4. Initialize and Train Model
    model = ImprovedARIMA(drug_info_mock)
    model.train(train_data)
    
    # 5. Predict
    # We need future exog (from test_data)
    future_exog = test_data[['Date', '平均气温', '平均降水量', '流感发病率', '流感ILI%', '节假日']]
    
    # Forecast without Decay
    forecast_raw = model.predict(steps=len(test_data), future_exog_df=future_exog)
    
    # Debug: Check values
    logger.info(f"Sample Actual: {test_data['Sales'].values[:5].tolist()}")
    logger.info(f"Sample Forecast: {forecast_raw.iloc[:5].values.tolist()}")
    
    # Forecast with Decay (Simulate stock is FRESH, 24 months = ~720 days)
    # Since > 90 days, factor should be 1.0 (No decay)
    forecast_decay = model.predict(steps=len(test_data), future_exog_df=future_exog, current_stock_validity_days=720)
    
    # 6. Evaluation
    try:
        mape = evaluate_mape(test_data['Sales'], forecast_raw)
        logger.info(f"Test MAPE: {mape:.2f}%")
    except Exception as e:
        logger.warning(f"Could not calculate MAPE: {e}")

    # 7. Plotting
    plt.figure(figsize=(12, 6))
    # Baseline Model (Naive: Predict last value observed)
    last_val = train_data['Sales'].iloc[-1]
    baseline_forecast = np.full(len(test_data), last_val)
    baseline_mape = evaluate_mape(test_data['Sales'], baseline_forecast)
    
    logger.info(f"Test MAPE: {mape:.2f}% | Baseline (Naive) MAPE: {baseline_mape:.2f}%")

    # Plot Results
    plt.figure(figsize=(14, 7))
    plt.plot(train_data['Date'].iloc[-60:], train_data['Sales'].iloc[-60:], label='Train (Last 60 days)', alpha=0.5)
    plt.plot(test_data['Date'], test_data['Sales'], label='Actual (Test)', color='green', linewidth=2)
    plt.plot(test_data['Date'], forecast_raw, label=f'Forecast (Improved ARIMA, MAPE={mape:.1f}%)', color='red', linestyle='--', linewidth=2)
    plt.plot(test_data['Date'], baseline_forecast, label=f'Baseline (Naive, MAPE={baseline_mape:.1f}%)', color='gray', linestyle=':')
    
    plt.title(f"Inventory Forecast: {top_drug} ({clinic})")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_plot = Path("data/processed/forecast_demo.png")
    plt.savefig(output_plot)
    logger.info(f"Plot saved to {output_plot}")
    
    # Save Forecast Data
    results_df = pd.DataFrame({
        'Date': test_data['Date'].values,
        'Actual': test_data['Sales'].values,
        'Forecast': forecast_raw,
        'Baseline': baseline_forecast
    })
    results_csv = Path("data/processed/forecast_results.csv")
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Forecast results saved to {results_csv}")

if __name__ == "__main__":
    main()
