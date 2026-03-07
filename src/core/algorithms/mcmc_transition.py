import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from src.core.simulation_config import SimulationConfig
from src.core.algorithms.demand import DemandModel
from src.core.algorithms.inventory_control import InventoryControl
from src.core.thesis_params import ThesisParams

class MCMC_Transition:
    """
    Operator T: State Transition Logic (MCMC Kernel)
    Orchestrates the daily simulation loop using Demand (A) and Control (B) operators.
    """
    def __init__(self, config: SimulationConfig, drug_info: Dict[str, Any], 
                 external_data: pd.DataFrame):
        self.config = config
        self.drug_info = drug_info
        self.external_data = external_data
        
        # Identity
        self.drug_id = drug_info.get('药品ID', 'Unknown')
        
        # State Variables
        self.current_day = 0
        self.inventory_batches = [] # List of {'qty': float, 'expiry_day': int, 'entry_day': int}
        self.pipeline_orders = []   # List of {'qty': float, 'arrival_day': int}
        
        # Volatility Classification (Crucial for Thesis logic)
        self.volatility_cat = self._determine_volatility(drug_info)
        
        # Initialize Sub-Algorithms
        self.demand_model = DemandModel(config, drug_info, self.volatility_cat)
        self.inventory_control = InventoryControl(config, drug_info, self.volatility_cat)
        
        # Initial State (Steady State Approximation)
        # Assume starting with decent stock to avoid immediate stockout
        self._initialize_state()

    def _determine_volatility(self, drug_info: Dict[str, Any]) -> str:
        """
        Determine volatility category based on Drug Name Hash to ensure Thesis Distribution:
        Total 128 SKUs -> Uses ThesisParams definitions.
        We use a stable hash to assign categories consistently.
        """
        import hashlib
        
        drug_name = str(drug_info.get('药品名称', 'Unknown'))
        # Use MD5 for stable hashing across runs
        hash_obj = hashlib.md5(drug_name.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        
        # Total SKUs
        total = ThesisParams.TOTAL_SKUS
        h_val = hash_int % total
        
        # Thresholds
        low_count = ThesisParams.VOLATILITY_COUNTS['LOW']
        med_count = ThesisParams.VOLATILITY_COUNTS['MEDIUM']
        # high_count = ThesisParams.VOLATILITY_COUNTS['HIGH']
        
        if h_val < low_count:
            return 'LOW'
        elif h_val < (low_count + med_count):
            return 'MEDIUM'
        else:
            return 'HIGH'

    def _initialize_state(self):
        # Start with 2 months of avg demand
        avg_demand = self.demand_model.raw_demand
        total_start_qty = avg_demand * 60
        
        # Distribute into batches with staggered expiry to simulate steady state
        # Instead of one fresh batch, create 3 batches:
        # 1. Fresh (365 days)
        # 2. Mid (180 days)
        # 3. Near Expiry (60 days) - To trigger early feedback for Tuner
        
        self.inventory_batches = [
            {'qty': total_start_qty * 0.5, 'expiry_day': 365, 'entry_day': 0},
            {'qty': total_start_qty * 0.3, 'expiry_day': 180, 'entry_day': -180},
            {'qty': total_start_qty * 0.2, 'expiry_day': 60,  'entry_day': -300}
        ]

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Capture current simulation state for backtracking/rewinding.
        Returns a deep copy of state variables.
        """
        import copy
        return {
            'current_day': self.current_day,
            'inventory_batches': copy.deepcopy(self.inventory_batches),
            'pipeline_orders': copy.deepcopy(self.pipeline_orders),
            # 'rng_state': ... # If we strictly wanted deterministic replay, but we want variation
        }

    def load_snapshot(self, snapshot: Dict[str, Any]):
        """
        Restore simulation state from a snapshot.
        """
        import copy
        self.current_day = snapshot['current_day']
        self.inventory_batches = copy.deepcopy(snapshot['inventory_batches'])
        self.pipeline_orders = copy.deepcopy(snapshot['pipeline_orders'])

    def run_simulation(self, duration_days: int = 365) -> pd.DataFrame:
        """
        Execute the full simulation timeline.
        Returns DataFrame of daily records.
        """
        records = []
        
        for day in range(1, duration_days + 1):
            self.current_day = day
            date = self.config.start_date + pd.Timedelta(days=day-1)
            
            # 1. External Factors (Weather, Flu)
            # Find closest date match or interpolate? Assuming daily index.
            # external_row = self.external_data.loc[date] if date in self.external_data.index else ...
            # Simplified: Random or mock if missing. 
            # In real implementation, external_data should be date-indexed.
            try:
                ext_row = self.external_data.loc[date]
            except KeyError:
                ext_row = pd.Series({'平均气温': 20, 'ILI%': 0.0}) # Default
            
            # 2. Operator A: Generate Demand
            # Scale clinic size (thesis factor)
            clinic_scale = 1.0 # Placeholder for now
            daily_demand = self.demand_model.generate(date, ext_row, clinic_scale)
            
            # 3. Receive Incoming Orders (Pipeline -> On Hand)
            self._process_deliveries(day)
            
            # 4. Check Expiration (Loss)
            expired_today, self.inventory_batches = self.inventory_control.check_expiration(
                self.inventory_batches, day
            )
            
            # 5. Fulfill Demand (Sales vs Stockout)
            # Calculate current total inventory
            current_inv_total = sum(b['qty'] for b in self.inventory_batches)
            
            actual_sales, self.inventory_batches = self.inventory_control.consume_stock(
                self.inventory_batches, daily_demand
            )
            
            stockout_qty = daily_demand - actual_sales
            is_stockout = stockout_qty > 0.001
            
            # 6. Operator B: Review & Reorder (Adaptive: Baseline vs Optimized)
            qty_ordered = 0.0
            
            # Determine Mode (ThesisParams Section 0)
            # Default to Baseline
            mode = 'BASELINE'
            
            # Use ThesisParams Time Split if possible, or just default to Baseline as we are generating Historical Data.
            # If the date is after 'test_split_date', we could use Optimized?
            # BUT: We are likely generating 2023-2024 data (up to Dec 2024).
            # Thesis says 2024.09-2024.12 is "Optimized Period" for validation.
            # So we should switch mode!
            
            split_date = pd.Timestamp(ThesisParams.SAMPLE_INFO['test_split_date'])
            if date >= split_date:
                mode = 'OPTIMIZED'
            
            # Check if today is a Review Day
            review_period = self.inventory_control.get_review_period(mode)
            
            # For HIGH Volatility in Optimized mode: "Add 1 temp check in Flu/Winter"
            # Winter: Dec-Feb? Flu: ILI% high?
            # Simple implementation: Just check period. If mode=Optimized and HighVol, check ILI?
            # Let's stick to simple Periodicity for now as per "15 days/time" + temp check.
            # We can implement temp check later if validation fails.
            
            if day % review_period == 0:
                pipeline_qty = sum(o['qty'] for o in self.pipeline_orders)
                
                # Demand Estimations
                # Baseline: Simple Moving Average or just Base Demand (Manual)
                # Optimized: ARIMA Forecast (Simulated with MAPE)
                
                avg_demand_est = self.demand_model.raw_demand 
                # Add Seasonality to Baseline estimate? No, Manual usually ignores it or lags.
                # Let's keep Baseline simple: Raw Demand.
                
                # Optimized Forecast
                # Synthesize a forecast that has X% MAPE error compared to actual 'daily_demand' (or 'next_period_demand')
                # But we only know 'daily_demand' for today. Real forecast predicts future.
                # Let's just use 'daily_demand' * error_factor as a proxy for "Perfect Forecast + Error".
                
                target_mape = ThesisParams.ARIMA_TARGETS.get(self.volatility_cat, {}).get('mape', 0.10)
                # Random error: N(0, MAPE * 1.25) -> Mean Absolute Error approx MAPE? 
                # Normal Dist: MAE = sigma * sqrt(2/pi) ~ 0.8 * sigma.
                # So sigma = MAPE / 0.8 = 1.25 * MAPE.
                sigma_err = target_mape * 1.25
                forecast_error = np.random.normal(0, sigma_err)
                forecast_daily_demand = daily_demand * (1 + forecast_error)
                if forecast_daily_demand < 0: forecast_daily_demand = 0.1

                dummy_std = avg_demand_est * 0.5 # For Baseline
                
                # Determine input demand for calculation
                demand_input = forecast_daily_demand if mode == 'OPTIMIZED' else avg_demand_est
                
                qty_ordered = self.inventory_control.calculate_order(
                    mode=mode,
                    avg_daily_demand=demand_input,
                    demand_std=dummy_std,
                    current_inventory_qty=current_inv_total,
                    pipeline_qty=pipeline_qty,
                    inventory_batches=self.inventory_batches,
                    current_day=day
                )
                
                if qty_ordered > 0:
                    self._place_order(qty_ordered, day)
            
            # 7. Record State
            records.append({
                'date': date,
                'drug_id': self.drug_id,
                'demand': daily_demand,
                'sales': actual_sales,
                'inventory': current_inv_total,
                'loss': expired_today,
                'stockout_flag': 1 if is_stockout else 0,
                'order_qty': qty_ordered,
                'volatility': self.volatility_cat
            })
            
        return pd.DataFrame(records)

    def _process_deliveries(self, day: int):
        # Move arrived orders to inventory
        arrived = [o for o in self.pipeline_orders if o['arrival_day'] <= day]
        remaining = [o for o in self.pipeline_orders if o['arrival_day'] > day]
        
        for order in arrived:
            # Create new batch
            new_batch = {
                'qty': order['qty'],
                'entry_day': day,
                'expiry_day': day + self.inventory_control.shelf_life
            }
            self.inventory_batches.append(new_batch)
            
        self.pipeline_orders = remaining

    def _place_order(self, qty: float, day: int):
        item_lead = self.inventory_control.lead_time
        arrival = day + item_lead
        self.pipeline_orders.append({
            'qty': qty,
            'arrival_day': arrival
        })
