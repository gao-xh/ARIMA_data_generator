import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
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
        
        # State History
        self.history = []     # List of partial DataFrames
        self.snapshots = {}   # Map: day -> snapshot dict
        self.params_log = []  # Log of parameter changes
        
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

    def run_adaptive_simulation(self, total_days=365) -> pd.DataFrame:
        """
        Main execution loop for segmented generation with backtracking.
        """
        self._report_progress('start', {'total_days': total_days, 'drug_id': self.drug_info.get('药品ID')})
        
        # Initialize Simulation
        sim = MCMC_Transition(self.base_config, self.drug_info, self.external_data)
        
        # Apply initial tuned params
        self._apply_params(sim)
        
        current_day = 0
        full_records = []
        
        while current_day < total_days:
            # 1. Define Segment
            segment_end = min(current_day + self.segment_days, total_days)
            segment_len = segment_end - current_day
            
            self._report_progress('segment_start', {
                'start_day': current_day, 
                'end_day': segment_end,
                'current_params': self.current_params
            })
            
            # 2. Save Checkpoint (Snapshot)
            self.snapshots[current_day] = sim.get_snapshot()
            
            # 3. Try Generation (Inner Optimization Loop)
            best_segment_df = None
            min_segment_error = float('inf')
            best_params = None
            
            for attempt in range(self.MAX_RETRIES + 1):
                # Run Segment
                segment_df = sim.run_simulation(duration_days=segment_len)
                
                # Evaluate Segment Performance
                segment_error, feedback = self._evaluate_segment(segment_df, current_day)
                
                self._report_progress('iteration', {
                    'attempt': attempt,
                    'error': segment_error, 
                    'params': self.current_params,
                    'feedback': feedback
                })

                # Keep track of best result
                if segment_error < min_segment_error:
                    min_segment_error = segment_error
                    best_segment_df = segment_df.copy()
                    best_params = copy.deepcopy(self.current_params)

                # Tolerance Check
                if segment_error < self.TOLERANCE_REL:
                    break # Good enough, proceed with current state
                
                # If constraint failed and we have retries left:
                if attempt < self.MAX_RETRIES:
                    # Rolling back state to start of segment
                    sim.load_snapshot(self.snapshots[current_day])
                    
                    # Compute Gradient Update
                    self._adjust_params(feedback)
                    
                    # Apply New Params
                    self._apply_params(sim)
            
            # 4. Commit Segment Logic
            # If the last run wasn't the best one (or if we just want to be sure), 
            # we need to ensure the simulation state is consistent with the chosen output.
            # Easiest way: Re-run the best configuration from the start snapshot one last time.
            
            # Restore start of segment
            sim.load_snapshot(self.snapshots[current_day])
            # Apply Winning Params
            self.current_params = best_params
            self._apply_params(sim)
            # Final Deterministic Run to advance state
            final_segment_df = sim.run_simulation(duration_days=segment_len)
            
            full_records.append(final_segment_df)
            self.params_log.append({
                'day': current_day,
                'params': copy.deepcopy(self.current_params),
                'error': min_segment_error
            })
            
            self._report_progress('segment_committed', {
                'day_range': (current_day, segment_end),
                'final_error': min_segment_error
            })
            
            current_day = segment_end
            
        full_df = pd.concat(full_records, ignore_index=True)
        self._report_progress('complete', {'total_records': len(full_df)})
        return full_df

    def _evaluate_segment(self, df: pd.DataFrame, start_day: int) -> Tuple[float, Dict[str, str]]:
        """
        Check if the generated segment trends towards the targets.
        Returns: (Error Score, Directional Feedback)
        """
        if df.empty:
            return 0.0, {}
            
        # Determine Regime (Baseline vs Optimized)
        split_date = pd.Timestamp(ThesisParams.SAMPLE_INFO['test_split_date'])
        # Check median date of segment
        avg_date = df['date'].iloc[len(df)//2]
        is_baseline = avg_date < split_date
        
        targets = ThesisParams.BASELINE_TARGETS if is_baseline else ThesisParams.OPTIMIZATION_TARGETS
        
        # Calculate Local Metrics (Aligned with ThesisValidator)
        total_loss = df['loss'].sum()
        total_sales = df['sales'].sum()
        avg_inventory = df['inventory'].mean()
        total_days = len(df)
        
        # 1. Loss Rate
        throughput = total_sales + total_loss
        loss_rate = total_loss / throughput if throughput > 0 else 0
        
        # 2. Stockout Rate
        stockout_days = df['stockout_flag'].sum()
        stockout_rate = stockout_days / total_days if total_days > 0 else 0
        
        # 3. Turnover Days (Annualized)
        sales_per_day = total_sales / total_days if total_days > 0 else 0
        turnover_days = avg_inventory / sales_per_day if sales_per_day > 0 else 0
        
        # 4. Backlog Rate (Overstock Rate)
        # Using 90 days threshold as per Validator
        overstock_threshold = 90 * sales_per_day
        if sales_per_day == 0:
             overstock_days = len(df[df['inventory'] > 0])
        else:
             overstock_days = len(df[df['inventory'] > overstock_threshold])
        backlog_rate = overstock_days / total_days if total_days > 0 else 0
        
        # Calculate Error and Feedback
        # Weights: Loss=3, Stockout=5 (Critical), Backlog=1, Turnover=1
        
        errors = {}
        feedback = {}
        total_error = 0.0
        
        # Define tolerances and check each
        checks = {
            'loss_rate': (loss_rate, 0.05, 3.0),
            'stockout_rate': (stockout_rate, 0.01, 5.0),
            'backlog_rate': (backlog_rate, 0.05, 1.0),
            'turnover_days': (turnover_days, 5.0, 1.0)
        }
        
        for key, (actual, tolerance, weight) in checks.items():
            if key not in targets: continue
            
            target = targets[key]
            diff = actual - target
            
            # Add to total error (normalized by target magnitude for scale)
            # Avoid div by zero
            denom = target if target != 0 else 1.0
            norm_diff = abs(diff) / denom
            
            # Special case for Days (absolute diff is better than relative sometimes)
            if key == 'turnover_days':
                 norm_diff = abs(diff) / 10.0 # scale by 10 days
                 
            total_error += norm_diff * weight
            
            # Directional Feedback
            if diff < -tolerance:
                feedback[key] = 'too_low'
            elif diff > tolerance:
                feedback[key] = 'too_high'
                
        return total_error, feedback

    def _adjust_params(self, feedback: Dict[str, str]):
        """
        Stochastic Gradient Descent on Simulation Parameters.
        Adjusts params based on directional feedback from multifaceted evaluation.
        """
        # 1. Loss Rate Control
        # Loss Too Low -> Increase Inventory (Safety Factor) OR Decrease Validity (Simulate worse management)
        # Loss Too High -> Decrease Inventory OR Increase Validity
        if feedback.get('loss_rate') == 'too_low':
            self.current_params['safety_factor'] *= (1 + self.LEARNING_RATE * 0.5)
            # Decrease validity to simulate older stock or bad rotation
            self.current_params['validity_days'] = int(self.current_params['validity_days'] * (1 - self.LEARNING_RATE))
        elif feedback.get('loss_rate') == 'too_high':
            self.current_params['safety_factor'] *= (1 - self.LEARNING_RATE * 0.5)
            self.current_params['validity_days'] = int(self.current_params['validity_days'] * (1 + self.LEARNING_RATE))
            
        # 2. Stockout Rate Control (Priority)
        # Stockout Too Low -> Decrease Safety Factor OR Increase Demand Volatility (Flu)
        # Stockout Too High -> Increase Safety Factor OR Decrease Demand Volatility
        if feedback.get('stockout_rate') == 'too_low':
             # Need more volatility to cause stockouts
             self.current_params['flu_sens'] *= (1 + self.LEARNING_RATE)
             # And maybe LESS inventory
             self.current_params['safety_factor'] *= (1 - self.LEARNING_RATE * 0.5)
        elif feedback.get('stockout_rate') == 'too_high':
             self.current_params['flu_sens'] *= (1 - self.LEARNING_RATE)
             self.current_params['safety_factor'] *= (1 + self.LEARNING_RATE * 0.8)

        # 3. Efficiency Metrics (Turnover & Backlog)
        # Both suggest LOWER inventory if "too_high"
        if feedback.get('turnover_days') == 'too_high' or feedback.get('backlog_rate') == 'too_high':
             # Inventory is too bloated
             self.current_params['safety_factor'] *= (1 - self.LEARNING_RATE * 0.5)
        elif feedback.get('turnover_days') == 'too_low':
             # Inventory is too lean
             self.current_params['safety_factor'] *= (1 + self.LEARNING_RATE * 0.3)

        # Bounds Check
        self.current_params['safety_factor'] = max(0.5, min(4.0, self.current_params['safety_factor']))
        self.current_params['validity_days'] = max(30, min(720, self.current_params['validity_days']))
        self.current_params['flu_sens'] = max(0.0, min(3.0, self.current_params['flu_sens']))

    def _apply_params(self, sim: MCMC_Transition):
        """
        Inject tuned parameters into the live simulation object.
        """
        # Inject into Config (which the Sim uses)
        # Note: SimulationConfig is a dataclass, so we can modify it.
        # But MCMC might extract values on init?
        # Checked code: MCMC uses self.config, and InventoryControl uses self.config.
        # So modifying sim.config should work dynamically!
        
        sim.config.safety_stock_factor = self.current_params['safety_factor']
        sim.config.validity_days = self.current_params['validity_days']
        sim.config.flu_sensitivity = self.current_params['flu_sens']
        
        # Also need to update sub-components if they cached values
        # InventoryControl reads from config in methods?
        # Let's check InventoryControl._calculate_baseline_order -> uses self.safety_factor?
        # Wait, InventoryControl reads params on init.
        # We need to force update InventoryControl.
        
        # Hack: Update InventoryControl attributes directly if they exist
        if hasattr(sim.inventory_control, 'safety_factor'):
            # Re-init policy params
            # sim.inventory_control.safety_factor is derived from Volatility in _init_policy_params
            # We want to OVERRIDE it with our Tuned Factor.
            # Let's set it directly.
            pass # TODO: MCMC Transition needs to support dynamic param update logic more cleanly.
            
            # For now, let's assume we update the specific attribute used in calculation
            # InventoryControl currently uses `self.safety_factor` in `_calculate_baseline_order`.
            # We overwrite it.
            sim.inventory_control.safety_factor = self.current_params['safety_factor']
            
            # Validity Days? InventoryControl uses `self.shelf_life`.
            # But batches already exist with fixed expiry.
            # Changing `shelf_life` only affects NEW batches.
            # This is correct behavior (we can't change physics of *existing* milk).
            sim.inventory_control.shelf_life = self.current_params['validity_days']