import pandas as pd
import sys
import os
import logging
from datetime import datetime

# Adjust path to find src if run directly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.generator_v2 import GeneratorV2
from src.core.simulation_config import SimulationConfig
from src.core import constants as C

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load source files."""
    try:
        # Load Drug Info
        drug_path = os.path.join(project_root, "data lib", "drug_info.csv")
        logger.info(f"Loading drug info from {drug_path}")
        drug_df = pd.read_csv(drug_path)
        
        # Load External Factors
        ext_path = os.path.join(project_root, "data lib", "external_factors.csv")
        logger.info(f"Loading external factors from {ext_path}")
        ext_df = pd.read_csv(ext_path)
        
        # Determine Date Column
        date_col = next((col for col in ext_df.columns if '日期' in col or 'Date' in col), None)
        if date_col:
            # Normalize to datetime
            try:
                ext_df[C.COL_DATE] = pd.to_datetime(ext_df[date_col], format='%Y%m%d')
            except:
                ext_df[C.COL_DATE] = pd.to_datetime(ext_df[date_col])
        else:
            logger.error("Could not find date column in external factors.")
            sys.exit(1)

        # 1.3 Backfill 2023 if needed (Thesis requires 2023 for training)
        min_date = ext_df[C.COL_DATE].min()
        if min_date.year == 2024:
            logger.info("External factors start in 2024. Backfilling 2023 data for Thesis training set...")
            df_2023 = ext_df[ext_df[C.COL_DATE].dt.year == 2024].copy()
            df_2023[C.COL_DATE] = df_2023[C.COL_DATE] - pd.DateOffset(years=1)
            # Adjust '日期' or 'Date' original columns if needed, but we rely on C.COL_DATE
            ext_df = pd.concat([df_2023, ext_df], ignore_index=True)
            ext_df = ext_df.sort_values(C.COL_DATE).reset_index(drop=True)
            
            # CRITICAL: Update the source column so GeneratorV2 doesn't revert it
            if date_col:
                if ext_df[date_col].dtype == 'int64' or ext_df[date_col].dtype == 'float64':
                    ext_df[date_col] = ext_df[C.COL_DATE].dt.strftime('%Y%m%d').astype(int)
                else:
                    ext_df[date_col] = ext_df[C.COL_DATE]

            logger.info(f"Data range extended: {ext_df[C.COL_DATE].min().date()} to {ext_df[C.COL_DATE].max().date()}")

        return drug_df, ext_df
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

def main():
    logger.info("Starting Dataset Generation V2 based on Thesis Constraints...")
    
    # 1. Load Data
    drug_df, ext_df = load_data()
    
    # 2. Configure Dataset Parameters (Targeting Thesis "Before Optimization" Stats)
    # Target Stats:
    # - Stockout Rate: ~3.1%
    # - Loss Rate: ~17.2%
    # - Turnover Days: ~44.6
    # - Period: 2023.01.01 - 2024.12.31 (24 Months)
    logger.info("Defining Thesis 'Non-Optimized' Baseline Environment (2023-2024)")
    
    # 7 Clinics abstracted as one entity
    config = SimulationConfig(
        start_date=pd.Timestamp('2023-01-01'),
        end_date=pd.Timestamp('2024-12-31'), 
        clinics={
            'Main_Clinic': 1.0,   
        },
        # Manual Mode Parameters (The constraints that cause the failure)
        replenishment_days=30,      # Thesis: "Monthly replenishment plan"
        safety_stock_factor=1.5,    # Thesis: "Empirical judgement" (conservative)
        validity_days=360,          # Base validity, will be randomized in Generator
        
        # Environmental Sensitivities (Cause of volatility)
        temp_sensitivity=1.5,  
        flu_sensitivity=3.0,   
        random_noise_sigma=0.3
    )
    
    # 3. Initialize Generator
    generator = GeneratorV2(drug_df, ext_df, config)
    
    # 4. Run Simulation
    logger.info("Running simulation...")
    synthetic_data = generator.generate()
    
    # 5. Post-Processing & Validation
    logger.info(f"Generated {len(synthetic_data)} records.")
    
    # Check for Thesis Effects
    stockouts = synthetic_data[synthetic_data[C.COL_STOCKOUT] == 1]
    logger.info(f"Total Stockout Events: {len(stockouts)} ({len(stockouts)/len(synthetic_data):.2%})")
    
    losses = synthetic_data[synthetic_data[C.COL_LOSS] > 0]
    logger.info(f"Total Expiry Loss Events: {len(losses)} ({len(losses)/len(synthetic_data):.2%})")
    
    # 6. Save Output
    output_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "synthetic_sales.csv")
    
    logger.info(f"Saving to {output_path}")
    synthetic_data.to_csv(output_path, index=False, encoding='utf-8-sig') # utf-8-sig for Excel
    
    logger.info("Done.")

if __name__ == "__main__":
    main()
