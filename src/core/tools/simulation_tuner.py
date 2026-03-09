import pandas as pd
import numpy as np
import random
from typing import Dict, Any, List, Optional, Tuple
import copy
from src.core.algorithms.mcmc_transition import MCMC_Transition
from src.core.simulation_config import SimulationConfig
from src.core.thesis_params import ThesisParams
from src.core import constants as C

class SimulationTuner:
    """
    Adaptive MCMC Controller (Gradient Descent over Simulation Parameters).
    Implements the "Thesis Alignment via Stochastic Optimization" strategy.
    
    OPTIMIZATION UPDATE:
    Now supports running 'Dual Simulation' (Baseline vs Optimized) in parallel
    to generate comparative datasets for Thesis Scenario Analysis.
    """
    
    def __init__(self, config: SimulationConfig, drug_info: Dict[str, Any], external_data: pd.DataFrame, 
                 progress_callback=None):
        """
        progress_callback: Optional callable(dict) to report detailed status to UI.
        """
        self.base_config = config
        
        # Ensure compatibility with drug_info csv columns
        # Map input keys to internal keys if needed
        # FIX: Do not strip other fields! We need '初始库存' and '日均销量' for GeneratorV2.
        self.drug_info = copy.deepcopy(drug_info)
        
        # Ensure critical keys exist (defaults)
        defaults = {
            '药品ID': 'UNKNOWN',
            '药品名称': 'Unknown',
            '药品品类': 'Unknown',
            '波动区间分类': None,
            '有效期': 365,
            '单价': 35.0
        }
        for k, v in defaults.items():
            if k not in self.drug_info:
                self.drug_info[k] = v
        
        # Initialize Tuner validity from Drug Info
        config.validity_days = int(self.drug_info.get('有效期', 365))

        
        self.external_data = external_data
        self.progress_callback = progress_callback

    def _report_progress(self, event_type: str, data: Dict[str, Any]):
        if self.progress_callback:
            payload = {
                'event': event_type, 
                'timestamp': pd.Timestamp.now().isoformat(),
                **data
            }
            self.progress_callback(payload)

    def run_simulation_only(self, total_days=730, evolution_mode=False, split_date=None, seed_value=None) -> pd.DataFrame:
        """
        Run side-by-side simulation: Baseline (A) vs Optimized (B).
        Returns a Combined Wide-Format DataFrame for direct comparison.
        
        Args:
            total_days: Duration of simulation
            evolution_mode: If True, simulate 'Hybrid' (Baseline -> AI) instead of 'Parallel' (AI Only).
            split_date: Date string (YYYY-MM-DD) for switching from Baseline to AI in Evolution Mode.
        """
        self._report_progress('start', {'total_days': total_days, 'drug_id': self.drug_info.get('药品ID')})
        
        # dynamic Seed for Variety but maintain Reproducibility across scenarios
        # Ensures that Demand (Sales) is identical for fair comparison between A and B
        if seed_value is None:
            seed_value = random.randint(0, 10000)
            
        print(f"Simulation Random Seed: {seed_value}")
        
        # --- Scenario A: Baseline (Empirical/Counterfactual) ---
        # In both modes, Scenario A is "What if we NEVER optimize?" (Pure Baseline)
        config_a = copy.deepcopy(self.base_config)
        
        # Reset Random State for Baseline
        np.random.seed(seed_value)
        random.seed(seed_value)
        
        # Force "Baseline" behavior settings
        ThesisParams.SAMPLE_INFO['test_split_date'] = '2099-12-31' 
        
        sim_a = MCMC_Transition(config_a, self.drug_info, self.external_data)
        df_a = sim_a.run_simulation(duration_days=total_days)
        
        # --- Scenario B: Optimized Target (AI Models) ---
        # Mode 1 (Default): Parallel Universe. Assume AI was used from Day 1.
        # Mode 2 (Evolution): Hybrid. Assume Baseline until split_date, then AI.
        
        if evolution_mode and split_date:
            # Evolution Mode: Switch at specific date
            ThesisParams.SAMPLE_INFO['test_split_date'] = split_date
        else:
            # Benchmark Mode: AI active from start
            ThesisParams.SAMPLE_INFO['test_split_date'] = '2000-01-01'
        
        config_b = copy.deepcopy(self.base_config)
        
        # Reset Random State for Optimized
        np.random.seed(seed_value)
        random.seed(seed_value)
        
        sim_b = MCMC_Transition(config_b, self.drug_info, self.external_data)
        df_b = sim_b.run_simulation(duration_days=total_days)
        
        # Restore Global State to Default
        ThesisParams.SAMPLE_INFO['test_split_date'] = '2025-09-01' 

        # --- Pivot & Merge for Visualization ---
        # 1. Standardize Dates
        if 'date' in df_a.columns:
             df_a = df_a.rename(columns={'date': C.COL_DATE})
        if 'date' in df_b.columns:
             df_b = df_b.rename(columns={'date': C.COL_DATE})
             
        # 2. Select and Rename Metrics
        cols_to_keep = [C.COL_DATE, 'inventory', 'loss', 'stockout_flag', 'sales', 'money_tied_up', 'forecast', 'demand']

        # Helper to process DF
        def process_df(df, prefix):
            # Check for missing columns (e.g. if simulation failed/empty)
            for c in cols_to_keep:
                if c not in df.columns and c != 'money_tied_up':
                    df[c] = 0
                
                # Calculate Funds Occupied if not present
                if c == 'money_tied_up' and c not in df.columns:
                     # Inventory * Price
                     # Assuming 'inventory' col exists and price is constant
                     unit_price = float(self.drug_info.get('单价', 35.0))
                     if 'inventory' in df.columns:
                         df[c] = df['inventory'] * unit_price
                     else:
                         df[c] = 0.0

            sub = df[cols_to_keep].copy()
            rename_map = {
                'inventory': f'{prefix}_Inventory',
                'loss': f'{prefix}_Loss',
                'stockout_flag': f'{prefix}_Stockout_Flag',
                'sales': f'{prefix}_Sales',
                'money_tied_up': f'{prefix}_Fund',
                'forecast': f'{prefix}_Forecast',
                'demand': f'{prefix}_Demand'
            }
            return sub.rename(columns=rename_map)

        df_a_clean = process_df(df_a, 'Baseline')
        df_b_clean = process_df(df_b, 'Optimized')
        
        # 3. Merge
        full_df = pd.merge(df_a_clean, df_b_clean, on=C.COL_DATE, how='outer')
        full_df = full_df.fillna(0) # Fill missings with 0
        full_df = full_df.sort_values(by=C.COL_DATE)
        
        self._report_progress('complete', {'rows': len(full_df)})
        return full_df