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
        start_qty = avg_demand * 60
        self.inventory_batches = [{
            'qty': start_qty,
            'expiry_day': 365, # Assumed fresh
            'entry_day': 0
        }]

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
            
            # 6. Operator B: Review & Reorder
            # Only review periodically (e.g. every 30 days) or if continuous (every day)
            # Thesis uses Periodic Review (R).
            # Let's say we check every R days.
            qty_ordered = 0.0
            if day % self.config.replenishment_days == 0:
                # Calculate pipeline inventory
                pipeline_qty = sum(o['qty'] for o in self.pipeline_orders)
                
                # Dynamic Demand Stats (Rolling Window? Or Static?)
                # Thesis implies adaptive. Let's use static base + noise for now, 
                # or a simple moving average of *past* demand if we tracked it.
                # Using known base demand for simplicity of the "Ideal" policy.
                avg_demand_est = self.demand_model.raw_demand # + seasonality adjustment?
                std_demand_est = avg_demand_est * 0.5 # Rough estimate of volatility
                
                target_level = self.inventory_control.calculate_order_target(
                    avg_demand_est, std_demand_est
                )
                
                qty_ordered = self.inventory_control.calculate_order_quantity(
                    current_inv_total, pipeline_qty, target_level
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
