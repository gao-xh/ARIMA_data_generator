from typing import Dict, Any, Tuple, List
import math
import numpy as np
from src.core.simulation_config import SimulationConfig
from src.core.thesis_params import ThesisParams

class InventoryControl:
    """
    Operator B: Inventory Control Policy (Recursive Algorithm)
    Thesis Logic: Periodic review (R, adjusted nQ) adapted for hospital drug management.
    Key Constraints to hit:
    - Stockout Rate: 3.1% (Requires careful safety stock)
    - Loss Rate: 17.2% (Likely high due to expiration from overstock)
    - Turnover: 44.6 days
    """
    def __init__(self, config: SimulationConfig, drug_info: Dict[str, Any], volatility_cat: str):
        self.config = config
        self.volatility_cat = volatility_cat
        
        # Drug Properties
        try:
            val = float(drug_info.get('有效期', 0))
            if val > 0:
                self.shelf_life = val
            else:
                # Thesis Logic: Validity depends on Volatility Category if unknown
                # Referencing ThesisParams
                params = ThesisParams.VOLATILITY_BEHAVIOR.get(self.volatility_cat, ThesisParams.VOLATILITY_BEHAVIOR['MEDIUM'])
                self.shelf_life = float(params['validity_days'])
        except (ValueError, TypeError):
            self.shelf_life = 365.0
            
        try:
            self.lead_time = int(drug_info.get('补货提前期', 3))      # Days
        except (ValueError, TypeError):
            self.lead_time = 3

        # Policy Parameters (To be tuned by Self-Check)
        self._init_policy_params()
        
    def _init_policy_params(self):
        """
        Initialize R, S based on volatility category to target specific KPIs.
        """
        # Periodic Review Logic from Thesis
        self.review_period = self.config.replenishment_days # Default 30 days
        
        # Get behavior parameters from ThesisParams
        params = ThesisParams.VOLATILITY_BEHAVIOR.get(self.volatility_cat, ThesisParams.VOLATILITY_BEHAVIOR['MEDIUM'])
        
        # Use z_score as the safety factor component
        self.safety_factor = float(params.get('z_score', params.get('safety_factor', 1.65)))
        self.service_level = float(params['service_level'])

    def calculate_order_target(self, avg_daily_demand: float, demand_std: float) -> float:
        """
        Calculate Target Inventory Level (S) aka Order-Up-To Level.
        S = (AvgDemand * (R + L)) + SafetyStock
        SafetyStock = Z * sigma_L * sqrt(L + R) usually 
        """
        review_horizon = self.review_period + self.lead_time
        
        # Simplified Statistical Logic:
        # Expected Demand during Review + Lead Time
        expected_demand = avg_daily_demand * review_horizon
        
        # Safety Stock Buffer
        # Ideally: Z * std_dev_demand * sqrt(Review + Lead)
        safety_stock = self.safety_factor * demand_std * np.sqrt(review_horizon)
        
        target_level = expected_demand + safety_stock
        return max(0.0, target_level)
    
    def calculate_order_quantity(self, current_inventory: float, pipeline_inventory: float, 
                               target_level: float) -> float:
        """
        Determine actual order quantity based on current position vs target.
        Order Q = Target - (On Hand + On Order)
        """
        inventory_position = current_inventory + pipeline_inventory
        order_qty = max(0.0, target_level - inventory_position)
        return order_qty

    def check_expiration(self, inventory_batches: List[Dict[str, Any]], current_day: int) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Check for expired items and remove them.
        Returns: (Loss Quantity, Updated Batches)
        """
        expired_qty = 0.0
        active_batches = []
        
        for batch in inventory_batches:
            # Batch: {'qty': 100, 'entry_day': 50, 'expiry_day': 415}
            if batch['expiry_day'] <= current_day:
                expired_qty += batch['qty']
            else:
                active_batches.append(batch)
                
        return expired_qty, active_batches

    def consume_stock(self, inventory_batches: List[Dict[str, Any]], demand_qty: float) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Consume stock for daily demand (FIFO).
        Returns: (Actual Sales, Updated Batches)
        """
        remaining_demand = demand_qty
        updated_batches = []
        actual_sales = 0.0
        
        # Sort batches by expiry date (FEFO/FIFO) - usually expiry matches entry order roughly
        sorted_batches = sorted(inventory_batches, key=lambda x: x['expiry_day'])
        
        for batch in sorted_batches:
            if remaining_demand <= 0:
                updated_batches.append(batch)
                continue
                
            if batch['qty'] > remaining_demand:
                # Partial consumption
                batch['qty'] -= remaining_demand
                actual_sales += remaining_demand
                remaining_demand = 0
                updated_batches.append(batch)
            else:
                # Full batch consumption
                actual_sales += batch['qty']
                remaining_demand -= batch['qty']
                # Batch is empty, do not re-add
        
        return actual_sales, updated_batches
