import pandas as pd
import numpy as np
import random
from typing import Dict, Any, List, Optional, Tuple
import copy
from src.core.algorithms.mcmc_transition import MCMC_Transition
from src.core.simulation_config import SimulationConfig
from src.core.thesis_params import ThesisParams
from src.core import constants as C
from src.models.improved_arima import ImprovedARIMA

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
        
        # Capture Model Metrics (AIC, R2, Order) from Optimized Run
        model_metrics = {}
        if hasattr(sim_b, 'get_model_metrics'):
            model_metrics = sim_b.get_model_metrics()
        
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
        
        # Attach metrics to DataFrame
        full_df.attrs['model_metrics'] = model_metrics

        # --- 4. Validation: Pure ARIMAX Fit (Out-of-Loop) ---
        # User Requirement: "Positive calculation of ARIMAX model data to reflect fitting degree"
        # We run a separate validation pass on the "Real Data" (Optimized_Demand from df_b)
        # to calculate R2 and AIC independent of the simulation loop's constraints.
        
        try:
            # Prepare Training Data (Simulated History)
            # Use data up to the test split (e.g. Sep 2025) for training
            # Use data after test split for testing/validation
            
            # The 'demand' column in df_b is the "Real" daily demand generated by the environment.
            # We treat this as ground truth for model validation.
            
            validation_df = df_b.copy()
            validation_df = validation_df.rename(columns={'demand': C.COL_SALES})
            
            # Ensure proper index
            validation_df = validation_df.reset_index(drop=True)
            
            # Ensure External Factors are present
            if self.external_data is not None:
                # Merge external data on Date
                # Check format first
                ext_df = self.external_data.copy()
                if C.COL_DATE not in ext_df.columns and isinstance(ext_df.index, pd.DatetimeIndex):
                    ext_df = ext_df.reset_index()
                elif C.COL_DATE in ext_df.columns and isinstance(ext_df.index, pd.DatetimeIndex):
                    # Sometimes index is Date and column is also there
                    ext_df = ext_df.reset_index(drop=True)
                
                if C.COL_DATE in ext_df.columns:
                     # Filter duplicates in merge just in case
                     ext_df = ext_df.drop_duplicates(subset=[C.COL_DATE])
                     validation_df = pd.merge(validation_df, ext_df, on=C.COL_DATE, how='left')
            
            # Fill NaN
            validation_df = validation_df.fillna(0)
            
            # Split Train/Test
            split_date = pd.Timestamp(ThesisParams.SAMPLE_INFO.get('test_split_date', '2025-09-01'))
            train_mask = validation_df[C.COL_DATE] < split_date
            test_mask = validation_df[C.COL_DATE] >= split_date
            
            train_data = validation_df.loc[train_mask]
            test_data = validation_df.loc[test_mask]
            
            if len(train_data) > 60 and len(test_data) > 0:
                 # Instantiate Model
                 # Need drug_info Series
                 drug_series = pd.Series(self.drug_info)
                 validator_model = ImprovedARIMA(drug_series)
                 
                 # Train on Historical Data
                 validator_model.train(train_data)
                 
                 # Get In-Sample Metrics (Training R2/AIC)
                 train_metrics = validator_model.get_metrics()
                 
                 # Predict Test Period (Out-of-Sample)
                 # We predict step-by-step or one-shot? 
                 # Standard validation: usually one-shot forecast for short term, or rolling.
                 # Let's do a one-shot forecast for the test duration (e.g. 120 days)
                 # This shows the model's pure predictive power without updates.
                 
                 steps = len(test_data)
                 exog_future = test_data  # Prepare_data inside predict handles the column selection
                 
                 forecast_values = validator_model.predict(steps, exog_future)
                 
                 # Calculate Out-of-Sample Metrics
                 y_true = test_data[C.COL_SALES].values
                 y_pred = np.array(forecast_values)
                 
                 # Handle length mismatch (rare)
                 min_len = min(len(y_true), len(y_pred))
                 y_true = y_true[:min_len]
                 y_pred = y_pred[:min_len]
                 
                 from sklearn.metrics import r2_score, mean_squared_error
                 
                 # Optimization: Resample to Weekly Average to smooth out daily noise
                 # This aligns with thesis requirement to capture "Trend" rather than "Daily Jitter"
                 eval_df = pd.DataFrame({
                     'y_true': y_true,
                     'y_pred': y_pred
                 }, index=test_data[C.COL_DATE][:min_len])
                 
                 # Use Weekly Mean (W) - preserves scale of "Avg Daily Sales"
                 eval_weekly = eval_df.resample('W').mean()
                 
                 # Drop NaN (from partial weeks if any)
                 eval_weekly = eval_weekly.dropna()
                 
                 if len(eval_weekly) > 2:
                     r2_out = r2_score(eval_weekly['y_true'], eval_weekly['y_pred'])
                     rmse_out = np.sqrt(mean_squared_error(eval_weekly['y_true'], eval_weekly['y_pred']))
                 else:
                     # Fallback to daily if not enough weeks
                     r2_out = r2_score(y_true, y_pred)
                     rmse_out = np.sqrt(mean_squared_error(y_true, y_pred))
                 
                 # Update model_metrics with this "Pure" validation
                 # Overwrite or Add? Add.
                 model_metrics['r2_train'] = train_metrics.get('r2', 0)
                 model_metrics['aic'] = train_metrics.get('aic', 0)
                 model_metrics['order'] = train_metrics.get('order', None)
                 
                 # Store Test Metrics (used by UI for 'Forecast R2')
                 # We store them in a way UI can pick them up?
                 # UI looks at `getattr(df, 'attrs', {})`
                 # Maybe we add a dedicated key 'validation_metrics'
                 model_metrics['r2_test'] = r2_out
                 model_metrics['rmse_test'] = rmse_out
                 
                 full_df.attrs['model_metrics'] = model_metrics
                 
                 # Optional: Attach the Pure Forecast to the DataFrame for plotting?
                 # Create a Series indexed by date
                 forecast_series = pd.Series(y_pred, index=test_data[C.COL_DATE][:min_len])
                 # Map back to full_df
                 full_df['Pure_ARIMAX_Forecast'] = full_df[C.COL_DATE].map(forecast_series).fillna(0) # Keep 0 for pre-forecast period

                 # Also attach the Fitted Values (Training Period)
                 # validator_model.model_fit.fittedvalues is a Series with DatetimeIndex (if train_data was indexed)
                 # But train_data index was reset?
                 # ImprovedARIMA.train() calls prepare_data which sets index. 
                 # So fittedvalues index should be correct DATES.
                 try:
                     fitted_vals = validator_model.model_fit.fittedvalues
                     # Reindex to full_df dates
                     full_df['Pure_ARIMAX_Fitted'] = full_df[C.COL_DATE].map(fitted_vals)
                 except Exception as e:
                     print(f"Error attaching fitted values: {e}")
                 
        except Exception as e:
            print(f"Pure ARIMAX Validation Failed: {e}")
        
        self._report_progress('complete', {'rows': len(full_df)})
        return full_df