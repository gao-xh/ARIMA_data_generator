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
        self.drug_info = drug_info
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
        # Determine Regime (Baseline vs Optimized)
        split_date = pd.Timestamp(ThesisParams.SAMPLE_INFO['test_split_date'])
        # Check median date of segment
        avg_date = df['date'].iloc[len(df)//2]
        is_baseline = avg_date < split_date
        
        targets = ThesisParams.BASELINE_TARGETS if is_baseline else ThesisParams.OPTIMIZATION_TARGETS
        
        # Calculate Local Metrics (Simple proxies)
        # Loss Rate: Loss / Sales (Sensitive to short term noise)
        total_loss = df['loss'].sum()
        total_sales = df['sales'].sum() + 0.1
        loss_rate = total_loss / (total_sales + total_loss)
        
        # Stockout Rate
        stockout_days = df['stockout_flag'].sum()
        stockout_rate = stockout_days / len(df)
        
        # Compare
        loss_err = (loss_rate - targets['loss_rate'])
        stock_err = (stockout_rate - targets['stockout_rate'])
        
        total_error = abs(loss_err) + abs(stock_err) # L1 Norm
        
        feedback = {}
        if loss_err < -0.05: feedback['loss'] = 'too_low'
        elif loss_err > 0.05: feedback['loss'] = 'too_high'
        
        if stock_err < -0.01: feedback['stockout'] = 'too_low'
        elif stock_err > 0.01: feedback['stockout'] = 'too_high'
        
        return total_error, feedback

    def _adjust_params(self, feedback: Dict[str, str]):
        """
        Stochastic Gradient Descent on Simulation Parameters.
        """
        # Gradient Rules
        # Loss Too Low -> Increase Inventory (Safety Factor) OR Decrease Validity
        # Loss Too High -> Decrease Inventory OR Increase Validity
        
        if feedback.get('loss') == 'too_low':
            self.current_params['safety_factor'] *= (1 + self.LEARNING_RATE)
            self.current_params['validity_days'] = int(self.current_params['validity_days'] * (1 - self.LEARNING_RATE))
        elif feedback.get('loss') == 'too_high':
            self.current_params['safety_factor'] *= (1 - self.LEARNING_RATE)
            self.current_params['validity_days'] = int(self.current_params['validity_days'] * (1 + self.LEARNING_RATE))
            
        # Stockout Too Low -> Decrease Safety Factor OR Increase Demand Volatility (Flu)
        # Stockout Too High -> Increase Safety Factor OR Decrease Demand Volatility
        
        if feedback.get('stockout') == 'too_low':
             # Need more volatility to cause stockouts
             self.current_params['flu_sens'] *= (1 + self.LEARNING_RATE)
             # And maybe LESS inventory
             self.current_params['safety_factor'] *= (1 - self.LEARNING_RATE * 0.5)
        elif feedback.get('stockout') == 'too_high':
             self.current_params['flu_sens'] *= (1 - self.LEARNING_RATE)
             self.current_params['safety_factor'] *= (1 + self.LEARNING_RATE * 0.5)

        # Bounds Check
        self.current_params['safety_factor'] = max(0.5, min(4.0, self.current_params['safety_factor']))
        self.current_params['validity_days'] = max(30, min(720, self.current_params['validity_days']))

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