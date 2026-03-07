import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import copy
from src.core.algorithms.mcmc_transition import MCMC_Transition
from src.core.simulation_config import SimulationConfig
from src.core.thesis_params import ThesisParams
from src.core.tools.validator import ThesisValidator

class SimulationTuner:
    """
    Adaptive MCMC Controller (Gradient Descent over Simulation Parameters).
    Implements the "Thesis Alignment via Stochastic Optimization" strategy.
    """
    
    TOLERANCE_REL = 0.15  # Acceptable relative error (15%)
    MAX_RETRIES = 5       # Max backtracks per segment
    LEARNING_RATE = 0.2   # How much to adjust parameter per step
    segment_days = 30     # Monthly checkpoints

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
            '单价': drug_info.get('单价', 35.0)
        }
        
        # Initialize Tuner validity from Drug Info
        config.validity_days = int(self.drug_info['有效期'])
        
        self.external_data = external_data
        self.validator = ThesisValidator(config)
        self.progress_callback = progress_callback
        
        # Current Parameters (Start from Base Config)
        self.current_params = {
            'safety_factor': config.safety_stock_factor,
            'validity_days': config.validity_days,
            'flu_sens': config.flu_sensitivity
        }

    def _report_progress(self, event_type: str, data: Dict[str, Any]):
        if self.progress_callback:
            payload = {
                'event': event_type, 
                'timestamp': pd.Timestamp.now().isoformat(),
                **data
            }
            self.progress_callback(payload)

    def run_simulation_only(self, total_days=365) -> pd.DataFrame:
        """
        Run simulation without adaptive tuning constraints.
        Direct execution of MCMC logic.
        """
        self._report_progress('start', {'total_days': total_days, 'drug_id': self.drug_info.get('药品ID')})
        
        # Initialize Simulation
        sim = MCMC_Transition(self.base_config, self.drug_info, self.external_data)
        
        # Run Full Duration
        full_records_df = sim.run_simulation(duration_days=total_days)

        self._report_progress('complete', {'rows': len(full_records_df)})
        return full_records_df






