from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import random
from pathlib import Path
import math

from src.core import constants as C

# Configure logger
logger = logging.getLogger(__name__)

class DataGenerator:
    """
    Core data generation logic.
    Simulates:
    1. Daily sales based on external factors (Flu, Temperature, Seasonality).
    2. Inventory replenishment based on naive/empirical methods.
    
    Attributes:
        drugs (pd.DataFrame): Drug master data (117 SKUs).
        external_factors (pd.DataFrame): Time backbone (Temperature, ILI%).
    """

    def __init__(self, drug_df: pd.DataFrame, external_factors_df: pd.DataFrame):
        self.drugs = drug_df.copy()
        self.external_factors = external_factors_df.copy()
        
        # Normalize index first
        self._normalize_external_factors()
        
        if C.COL_DATE in self.external_factors.columns:
            self.start_date = pd.to_datetime(self.external_factors[C.COL_DATE]).min()
            self.end_date = pd.to_datetime(self.external_factors[C.COL_DATE]).max()
        else:
            # Fallback if date is index
             self.start_date = self.external_factors.index.min()
             self.end_date = self.external_factors.index.max()
        
        # Ensure external factors are indexed by date
        if C.COL_DATE in self.external_factors.columns:
            self.external_factors[C.COL_DATE] = pd.to_datetime(self.external_factors[C.COL_DATE])
            self.external_factors.set_index(C.COL_DATE, inplace=True)
            self.external_factors.sort_index(inplace=True)
        elif self.external_factors.index.name == C.COL_DATE:
             self.external_factors.index = pd.to_datetime(self.external_factors.index)
             self.external_factors.sort_index(inplace=True)
        else:
            # Maybe index is already datetime but unnamed?
            if isinstance(self.external_factors.index, pd.DatetimeIndex):
                self.external_factors.index.name = C.COL_DATE
            else:
                raise ValueError(f"Missing date column '{C.COL_DATE}' in external factors. Columns found: {self.external_factors.columns}")
        
        # Categorize drugs by volatility if not already
        if '波动区间分类' not in self.drugs.columns:
            self._categorize_drugs()

    def _normalize_external_factors(self):
        """Standardize column names for external factors."""
        mapping = {
            '日期(UTC)': C.COL_DATE,
            'Date': C.COL_DATE,
            '平均气温(C)': C.EXT_TEMP,
            '流感发病率': C.EXT_FLU,
            '流感ILI%': C.EXT_FLU,
            '节假日': C.EXT_HOLIDAY
        }
        self.external_factors.rename(columns=mapping, inplace=True)


    def _categorize_drugs(self):
        """Helper to calculate CV and categorize drugs if missing."""
        logger.info("Categorizing drugs based on simulated CV...")
        # Since we simulate, we can assign classification randomly or by ID logic first.
        # Here we assume the input file HAS this column, based on user context.
        # Fallback: Assign randomly heavily weighted to High/Mid for demonstration.
        np.random.seed(42)
        choices = [C.FLUC_LOW, C.FLUC_MED, C.FLUC_HIGH]
        self.drugs['波动区间分类'] = np.random.choice(choices, size=len(self.drugs), p=[0.3, 0.5, 0.2])

    def generate_full_dataset(self, clinics: List[str] = None) -> pd.DataFrame:
        """
        Main entry point to generate the full dataset for all clinics.
        """
        if not clinics:
            # Default 7 clinics from context
            clinics = [f"Clinic_{i}" for i in range(1, 8)]
            
        all_records = []
        total_steps = len(clinics) * len(self.drugs)
        current_step = 0
        
        logger.info(f"Starting simulation for {len(clinics)} clinics over {len(self.external_factors)} days.")

        for clinic in clinics:
            for _, drug in self.drugs.iterrows():
                drug_records = self._simulate_single_drug(clinic, drug)
                all_records.extend(drug_records)
                current_step += 1
                if current_step % 100 == 0:
                    logger.info(f"Progress: {current_step}/{total_steps} drug-clinic pairs simulated.")
                    
        dataset = pd.DataFrame(all_records)
        return dataset

    def _simulate_single_drug(self, clinic: str, drug: pd.Series) -> List[Dict[str, Any]]:
        """
        Simulates daily sales and inventory for one drug at one clinic.
        """
        records = []
        
        # 1. Base Parameters
        category = drug.get('波动区间分类', C.FLUC_MED)
        drug_code = drug.get('药品编码')
        base_demand = float(drug.get('日均销量', 5))  # Fallback if provided, else random
        if pd.isna(base_demand): base_demand = random.randint(2, 10)
        
        # 2. Simulation State
        current_inventory = float(base_demand * 14) # Start with 2 weeks stock
        replenishment_cycle = 14  # Bi-weekly review
        pending_reorder = 0
        lead_time = random.randint(2, 5) # 2-5 days delivery
        reorder_day_counter = 0

        # Pre-fetch external series for speed
        temp_series = self.external_factors.get(C.EXT_TEMP, pd.Series(dtype=float))
        flu_series = self.external_factors.get(C.EXT_FLU, pd.Series(dtype=float))
        
        # Iterate through timeline
        for date, row in self.external_factors.iterrows():
            # --- 2.1 Sales Simulation (H1 Logic) ---
            # Factor 1: Seasonality/Weather
            temp = row.get(C.EXT_TEMP, 20)
            flu_rate = row.get(C.EXT_FLU, 0)
            
            # Base sales with random noise
            daily_sales = max(0, np.random.poisson(base_demand))
            
            # Apply External Shocks for Sensitive Drugs
            if category == C.FLUC_HIGH or category == C.FLUC_MED:
                # Flu Impact: If ILI% > threshold (e.g. 5%), boost sales
                if flu_rate > 5.0:
                    # Exponential boost based on severity
                    boost = 1 + (flu_rate - 5.0) * 0.2  # e.g., rate=10 -> 1 + 1 = 2x sales
                    daily_sales *= boost
                
                # Temperature Impact: Cold shocks boost demand
                if temp < 5:
                    daily_sales *= 1.3
            
            # Add randomness (White Noise)
            daily_sales = int(daily_sales * random.uniform(0.8, 1.2))
            
            # --- 2.2 Inventory Logic (Problem Simulation) ---
            
            # Receiving Stock (Check if previous order arrived)
            received_qty = 0
            if reorder_day_counter > 0:
                reorder_day_counter -= 1
                if reorder_day_counter == 0:
                    received_qty = pending_reorder
                    pending_reorder = 0
            
            # Update Current Stock (Before Sales)
            start_inventory = current_inventory + received_qty
            
            # Fulfill Demand
            realized_sales = min(start_inventory, daily_sales)
            stockout_flag = 1 if daily_sales > start_inventory else 0
            
            # Update End Stock
            end_inventory = start_inventory - realized_sales
            
            # --- 2.3 Replenishment Logic (Naive Periodic Review) ---
            # Strategy: Every 14 days, order up to "Target Level"
            # Target Level = Avg Daily Sales (last 30 days observed) * 20 days coverage
            replenish_qty = 0
            day_of_year = date.dayofyear
            
            if day_of_year % replenishment_cycle == 0 and pending_reorder == 0:
                # Naive Forecast: Simple average of recent history (simulating "experience")
                # Problem: Reaction is slow. If flu outbreak just started, average is low -> Stockout.
                # If flu just ended, average is high -> Overstock.
                
                # Calculate simple moving average (mocking "experience")
                # For simplicity in loop, use base_demand * random error to simulate human estimation error
                estimated_demand = base_demand * random.uniform(0.9, 1.1) 
                
                # If currently in high demand season (but unseen by naive method), estimation is low
                # We want to force the failure, so keep estimation grounded in "past history" (base_demand)
                target_stock = estimated_demand * 21 # Target 3 weeks coverage
                
                if end_inventory < target_stock:
                    replenish_qty = int(target_stock - end_inventory)
                    pending_reorder = replenish_qty
                    reorder_day_counter = lead_time # Order lead time
            
            # --- 2.4 Loss/Expiry Logic ---
            # Simulate occasional expiry if stock is too high for too long
            loss_qty = 0
            if end_inventory > base_demand * 60: # If stock > 2 months supply
                # Random chance of expiry
                if random.random() < 0.05:
                    loss_qty = int(end_inventory * 0.1) # 10% expires
                    end_inventory -= loss_qty

            # Record Data
            record = {
                C.COL_DATE: date,
                C.COL_CLINIC: clinic,
                C.COL_DRUG_CODE: drug_code,
                C.COL_SALES: int(daily_sales), # Actual Demand
                C.COL_INV_START: int(start_inventory),
                C.COL_INV_END: int(end_inventory),
                C.COL_REPLENISH: int(replenish_qty), # Ordered today
                C.COL_STOCKOUT: stockout_flag,
                C.COL_LOSS: int(loss_qty),
                'Temp': temp,
                'Flu': flu_rate
            }
            records.append(record)
            
            # Carry over
            current_inventory = end_inventory
            
        return records
