import pandas as pd
import sys
import os
import logging
from typing import List, Dict

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import DRUG_INFO, EXTERNAL_FACTORS_FILE, PROCESSED_DATA_DIR
from src.core.simulation_config import SimulationConfig
from src.core.tools.simulation_tuner import SimulationTuner
from src.core.tools.validator import ThesisValidator
from src.core.thesis_params import ThesisParams

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load metadata and external factors."""
    try:
        # Load Drug Info
        logger.info(f"Loading Drug Info from {DRUG_INFO}")
        try:
            drug_df = pd.read_csv(DRUG_INFO, encoding='utf-8')
        except UnicodeDecodeError:
            drug_df = pd.read_csv(DRUG_INFO, encoding='gb18030')
        
        # Load External Factors
        logger.info(f"Loading External Factors from {EXTERNAL_FACTORS_FILE}")
        ext_df = pd.read_csv(EXTERNAL_FACTORS_FILE)
        
        # Parse Dates in External Factors
        # Assuming '日期' or 'date' column
        date_col = next((c for c in ext_df.columns if 'date' in c.lower() or '日期' in c), None)
        if date_col:
            ext_df['date'] = pd.to_datetime(ext_df[date_col])
        else:
            # Fallback: Generate if missing
            logger.warning("No date column in external factors. Generating default range.")
            dates = pd.date_range(start='2023-01-01', end='2024-12-31')
            ext_df = pd.DataFrame({'date': dates})
            # Add dummy columns if missing
            if '平均气温' not in ext_df.columns: ext_df['平均气温'] = 20.0
            if 'ILI%' not in ext_df.columns: ext_df['ILI%'] = 0.0
            
        return drug_df, ext_df
        
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        sys.exit(1)

def run_thesis_generation():
    """
    Orchestrate the full dataset generation based on Simulation Logic.
    """
    logger.info("Starting Thesis Dataset Generation...")
    
    drug_df, ext_df = load_data()
    
    # Configuration
    # Start: 2023-01-01, End: 2024-12-31 (2 Years)
    config = SimulationConfig(
        start_date=pd.Timestamp('2023-01-01'),
        end_date=pd.Timestamp('2024-12-31'),
        clinics={'Clinic_Total': 1.0} # Aggregate level for Thesis
    )
    
    all_results = []
    total_drugs = len(drug_df)
    
    # Process each drug
    for idx, row in drug_df.iterrows():
        drug_name = row.get('药品名称', 'Unknown')
        logger.info(f"[{idx+1}/{total_drugs}] Processing {drug_name}...")
        
        # Determine Volatility Category (Need to map from CSV)
        vol_raw = row.get('波动区间分类', '中波动')
        if '高' in vol_raw: vol_cat = 'HIGH'
        elif '低' in vol_raw and '中' not in vol_raw: vol_cat = 'LOW'
        else: vol_cat = 'MEDIUM'
        
        # Prepare Drug Info Dict
        validity_months = row.get('效期（月）', 12)
        try:
            validity_days = int(validity_months) * 30
        except:
            validity_days = 365
            
        drug_info = {
            '药品ID': row.get('药品编号', f'DRUG_{idx:03d}'),
            '药品名称': drug_name,
            '药品品类': row.get('药品品类', 'Unknown'),
            '有效期': validity_days,
            '单价': float(row.get('零售价', 35.0)),
            '日均销量': float(row.get('日均销量', 5.0)),
            '波动区间分类': vol_cat # For internal use
        }
        
        # Initialize Tuner (Simple Runner)
        tuner = SimulationTuner(config, drug_info, ext_df)
        
        # Run Simulation
        # Total duration = 730 days (2 years)
        sim_df = tuner.run_simulation_only(total_days=730)
        
        # Append Metadata
        sim_df['drug_name'] = drug_name
        sim_df['category'] = drug_info['药品品类']
        
        all_results.append(sim_df)
        
    # Combine All
    full_dataset = pd.concat(all_results, ignore_index=True)
    
    # --- TRANSFORM TO TEMPLATE FORMAT ---
    logger.info("Formatting dataset to match sales_inventory_template.csv...")
    
    # 1. Calculate Opening Inventory (Yesterday's Closing)
    # Since we don't have yesterday's row easily for the first day, we can reverse engineer:
    # Closing = Opening + Replenishment - Sales - Loss
    # Opening = Closing - Replenishment + Sales + Loss
    # Note: 'replenishment' column was added to MCMC_Transition recently.
    
    if 'replenishment' not in full_dataset.columns:
        # Fallback if MCMC hasn't been updated yet in memory or logic (should be fixed by previous tool call)
        full_dataset['replenishment'] = full_dataset.get('order_qty', 0) # Approximation if missing
        
    full_dataset['opening_inventory'] = full_dataset['inventory'] - full_dataset['replenishment'] + full_dataset['sales'] + full_dataset['loss']
    
    # 2. Rename Columns
    # Template: 序号,药品编码,日期,当日销量（单位）,当日期初库存,当日期末库存,当日补货量,缺货标记,损耗数量,所属诊所
    
    rename_map = {
        'drug_id': '药品编码',
        'date': '日期',
        'sales': '当日销量（单位）',
        'opening_inventory': '当日期初库存',
        'inventory': '当日期末库存',
        'replenishment': '当日补货量',
        'stockout_flag': '缺货标记',
        'loss': '损耗数量'
    }
    
    # Create Final DataFrame
    final_df = full_dataset.rename(columns=rename_map).copy()
    
    # 3. Add Missing Columns
    final_df['序号'] = range(1, len(final_df) + 1)
    final_df['所属诊所'] = '中心库' # Default or from config
    
    # 4. Enforce Data Types (Round to Int)
    int_cols = ['序号', '当日销量（单位）', '当日期初库存', '当日期末库存', '当日补货量', '缺货标记', '损耗数量']
    for col in int_cols:
        final_df[col] = final_df[col].fillna(0).round().astype(int)
        
    # 5. Select & Reorder Columns
    target_columns = ['序号', '药品编码', '日期', '当日销量（单位）', '当日期初库存', '当日期末库存', '当日补货量', '缺货标记', '损耗数量', '所属诊所']
    final_df = final_df[target_columns]
    
    # Save to CSV
    output_path = os.path.join(PROCESSED_DATA_DIR, "thesis_dataset_final.csv")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"Dataset saved to {output_path}")
    
    # --- VALIDATION PHASE ---
    # Validator expects internal names (sales, inventory, etc.)
    # So we pass 'full_dataset' (internal format) to validator
    
    logger.info("Running Statistical Validation...")
    validator = ThesisValidator(config)
    validation_result = validator.validate_dataset(full_dataset)
    
    # Generate Report
    report = validator.generate_markdown_report(validation_result)
    
    print("\n" + "="*50)
    print("THESIS VALIDATION REPORT")
    print("="*50)
    print(report)
    print("="*50 + "\n")
    
    # Save Report
    report_path = os.path.join(project_root, "docs", "THESIS_VALIDATION_REPORT_FINAL.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

if __name__ == "__main__":
    run_thesis_generation()