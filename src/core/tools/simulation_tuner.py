import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import copy
from src.core.algorithms.mcmc_transition import MCMC_Transition
from src.core.simulation_config import SimulationConfig
from src.core.thesis_params import ThesisParams

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
        self.drug_info = {
            '药品ID': drug_info.get('药品ID', 'UNKNOWN'),
            '药品名称': drug_info.get('药品名称', 'Unknown'),
            # Validity in CSV is MONTHS usually, converted to DAYS in UI loading
            '有效期': drug_info.get('有效期', 365),
            '单价': drug_info.get('单价', 35.0),
            # Flu Sensitivity embedded in Drug Info? Or Config?
            # Config has global sensitivity.
        }
        
        # Initialize Tuner validity from Drug Info
        config.validity_days = int(self.drug_info['有效期'])
        
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

    def run_simulation_only(self, total_days=730) -> pd.DataFrame:
        """
        Run side-by-side simulation: Baseline (A) vs Optimized (B).
        Returns a combined DataFrame with a 'scenario' column.
        """
        self._report_progress('start', {'total_days': total_days, 'drug_id': self.drug_info.get('药品ID')})
        
        # --- Scenario A: Baseline (Empirical) ---
        # Logic: Fixed R=30, Manual Safety Stock (High Inventory, Low Service Level if Demand Spikes)
        config_a = copy.deepcopy(self.base_config)
        # Force "Baseline" behavior
        # In current MCMC_Transition, mode is date-dependent. 
        # We need to FORCE the mode.
        # Let's modify MCMC_Transition or use a subclass?
        # Simpler: Pass a flag in Config or DrugInfo to MCMC?
        # Or just run it and let MCMC decide based on date?
        # The user wants "Optimization", so let's stick to the Thesis Split:
        # 2023-2024 (Sep) = Baseline, 2024 (Sep-Dec) = Optimized.
        # BUT for "What-If" analysis, we want to see what Optimized WOULD have done in 2023.
        
        # Strategy: Run TWO independent timelines.
        # 1. Baseline Run: Mode always 'BASELINE'
        sim_a = MCMC_Transition(config_a, self.drug_info, self.external_data)
        # We need to hack/force the mode in MCMC. 
        # For now, let's assume MCMC switches at 'test_split_date'.
        # To force Baseline, we set 'test_split_date' to FAR FUTURE.
        ThesisParams.SAMPLE_INFO['test_split_date'] = '2099-12-31' 
        
        df_a = sim_a.run_simulation(duration_days=total_days)
        df_a['scenario'] = 'Baseline'
        
        # 2. Optimized Run: Mode always 'OPTIMIZED'
        # To force Optimized, we set 'test_split_date' to FAR PAST.
        ThesisParams.SAMPLE_INFO['test_split_date'] = '2000-01-01'
        
        # Crucial: Optimized run should use the UI parameters (L, R, S)
        # The UI config (self.base_config) has the user's chosen values.
        # We need to ensure MCMC uses them in 'OPTIMIZED' mode calculations.
        config_b = copy.deepcopy(self.base_config)
        
        sim_b = MCMC_Transition(config_b, self.drug_info, self.external_data)
        df_b = sim_b.run_simulation(duration_days=total_days)
        df_b['scenario'] = 'Optimized'
        
        # Combine
        full_df = pd.concat([df_a, df_b], ignore_index=True)
        
        # Restore Global State (Clean up)
        # (Ideally shouldn't modify global state, but for this demo script it's effective)
        ThesisParams.SAMPLE_INFO['test_split_date'] = '2024-09-01' 

        self._report_progress('complete', {'rows': len(full_df)})
        return full_df