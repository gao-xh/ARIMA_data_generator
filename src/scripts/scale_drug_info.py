import pandas as pd
import numpy as np
import os

# Paths
OLD_INFO_PATH = r"ref/旧药房药品信息一览表.xls"
DRUG_INFO_PATH = r"data lib/drug_info.csv"
OUTPUT_PATH = r"data lib/drug_info.csv"

def scale_drug_data():
    try:
        # Load Data
        df_old = pd.read_excel(OLD_INFO_PATH)
        df_curr = pd.read_csv(DRUG_INFO_PATH, encoding='utf-8')
        
        print(f"Loaded Old Info: {df_old.shape}")
        print(f"Loaded Current Info: {df_curr.shape}")
        
        # Normalize Key Columns (Drug Name is safer than Code which might vary in format)
        # Check Name overlap
        old_names = set(df_old['药品名称'].astype(str).str.strip())
        curr_names = set(df_curr['药品名称'].astype(str).str.strip())
        overlap = old_names.intersection(curr_names)
        print(f"Name Overlap: {len(overlap)} / {len(curr_names)}")
        
        # Merge on Name
        df_curr['药品名称_clean'] = df_curr['药品名称'].astype(str).str.strip()
        df_old['药品名称_clean'] = df_old['药品名称'].astype(str).str.strip()
        
        # We want '数量' (Quantity) from Old Info
        merged = pd.merge(df_curr, df_old[['药品名称_clean', '数量']], on='药品名称_clean', how='left')
        
        # Fill missing Quantities with Median (to avoid NaN)
        median_qty = merged['数量'].median()
        if pd.isna(median_qty): median_qty = 100 # Fallback
        merged['数量'] = merged['数量'].fillna(median_qty)
        
        # --- Scaling Logic (Thesis Constraints) ---
        # User: "Initial Qty ~ 1/4 of Total" -> Scale by 4x
        SCALE_FACTOR = 4.0
        
        # 1. Initial Stock (Scaling)
        merged['初始库存'] = (merged['数量'] * SCALE_FACTOR).astype(int)
        
        # 2. Daily Demand (Proportional Adjustment to Hit Target ~2000 Visits)
        # User Constraint: Total Daily Visits ~ 2000
        # Assumption: 1 Visit = 1 Drug Unit (Simplified, or calculate average items per visit)
        # If Avg Items/Visit = 1.5 -> Total Units = 3000
        # If Avg Items/Visit = 2.0 -> Total Units = 4000
        # Let's target Total Daily Units = 2500 (conservative 1.25 items/visit)
        
        # Current Total from Old quantities
        # Old Quantity Sum
        total_old_qty = merged['数量'].sum()
        print(f"Total Old Quantity: {total_old_qty}")
        
        # If we use Quantity/45 logic, what's total daily?
        # daily_sum = (total_old_qty * 4.0) / 45.0
        # Let's calculate the required scale factor
        
        TARGET_DAILY_TOTAL = 2500.0
        
        # Check current distribution of quantities to maintain relative weights
        weights = merged['数量'] / total_old_qty
        
        # Assign Daily Demand based on weights
        merged['日均销量'] = (weights * TARGET_DAILY_TOTAL).round(2)
        
        # Ensure min sales > 0.1
        merged['日均销量'] = merged['日均销量'].apply(lambda x: max(0.1, x))
        
        # 3. Initial Stock (consistent with Turnover Target 45 days)
        TURNOVER_TARGET_DAYS = 45.0
        merged['初始库存'] = (merged['日均销量'] * TURNOVER_TARGET_DAYS).astype(int)
        
        # Drop temp columns
        cols_to_keep = [c for c in df_curr.columns if c != '药品名称_clean'] + ['初始库存', '日均销量']
        final_df = merged[cols_to_keep].copy()
        
        # Save
        final_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
        print(f"Updated {OUTPUT_PATH} with '初始库存' and '日均销量'.")
        print(final_df[['药品名称', '初始库存', '日均销量']].head())
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    scale_drug_data()