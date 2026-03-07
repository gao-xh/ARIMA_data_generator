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
        self.safety_factor = list(params.values())[5] if 'safety_factor' not in params else params['safety_factor'] 
        # Fallback if TypedDict behavior differs at runtime

        
    def get_review_period(self, mode: str) -> int:
        if mode == 'OPTIMIZED':
            strat = ThesisParams.REPLENISHMENT_STRATEGY.get(self.volatility_cat, ThesisParams.REPLENISHMENT_STRATEGY['MEDIUM'])
            return strat['review_period_days']
        else:
            return self.config.replenishment_days

    def calculate_order(self, 
                       mode: str,
                       avg_daily_demand: float,
                       demand_std: float,
                       current_inventory_qty: float,
                       pipeline_qty: float,
                       inventory_batches: List[Dict[str, Any]] = None,
                       current_day: int = 0) -> float:
        """
        Unified Order Calculation Router.
        Mode: 'BASELINE' (Empirical) or 'OPTIMIZED' (Thesis Section 3.3.1)
        """
        if mode == 'OPTIMIZED':
            return self._calculate_optimized_order(avg_daily_demand, current_inventory_qty, pipeline_qty, inventory_batches, current_day)
        else:
            return self._calculate_baseline_order(avg_daily_demand, demand_std, current_inventory_qty, pipeline_qty)

    def _calculate_baseline_order(self, avg_daily_demand: float, demand_std: float, 
                                current_inventory: float, pipeline_inventory: float) -> float:
        """
        Baseline Strategy (Empirical Mode):
        - Manual Periodic Review (R=30 usually)
        - Safety Stock = Manual Factor * AvgDemand (High buffer)
        - Target = Avg * (R+L) + SS
        """
        # Baseline uses simplistic R=30 for most
        # But for generating 17.2% loss, we need overstocking.
        # So we use High Safety Factors defined in ThesisParams for 'BASELINE_TARGETS' implicit logic
        
        # Consistent with get_review_period
        review_days = self.get_review_period('BASELINE')
        review_horizon = review_days + self.lead_time
        
        # Empirical Safety Stock Logic (Often just Weeks of Supply)
        # Low Vol: ~2 weeks safety? High Vol: ~4 weeks?
        # Thesis mentions "Manual safety stock is 14 days supply approx"
        # Let's use the Volatility behavior safety factor
        
        ss_qty = self.safety_factor * demand_std * np.sqrt(review_horizon)
        
        target_level = (avg_daily_demand * review_horizon) + ss_qty
        inventory_position = current_inventory + pipeline_inventory
        
        return max(0.0, target_level - inventory_position)

    def _calculate_optimized_order(self, 
                                      forecast_daily_demand: float, 
                                      current_inventory_qty: float, 
                                      pipeline_qty: float,
                                      inventory_batches: List[Dict[str, Any]],
                                      current_day: int) -> float:
        """
        Operator B_new: Thesis Formula (Section 3.3.1)
        OR = SS + Y_hat * T - I - LSL
        """
        
        # 1. Safety Stock (SS)
        # Use optimized parameters from ThesisParams.REPLENISHMENT_STRATEGY
        strat = ThesisParams.REPLENISHMENT_STRATEGY.get(self.volatility_cat, ThesisParams.REPLENISHMENT_STRATEGY['MEDIUM'])
        
        # Target SS from strategy (e.g. 8, 19, 39 units)
        # But this is "Average SS". The formula says SS = Z*sigma*L.
        # Let's use the formula dynamically to allow for demand shifts.
        
        # Heuristic CV for sigma estimation
        if self.volatility_cat == 'LOW': cv = 0.15
        elif self.volatility_cat == 'MEDIUM': cv = 0.35
        else: cv = 0.6
        
        demand_std = forecast_daily_demand * cv
        
        # Z from Strategy
        z = ThesisParams.REPLENISHMENT_STRATEGY['common_params']['z_score']
        # L from Strategy or Drug Info? Strategy has fixed L=4, but drug might have specific.
        # Thesis says "L=4 (replenishment lead time)". Let's use 4 as per optimization design.
        L = 4 
        
        ss_qty = z * demand_std * np.sqrt(L)
        
        # 2. Cycle Stock (Y_hat * T)
        # T from Strategy (30 or 15)
        T = strat['review_period_days']
        
        cycle_stock = forecast_daily_demand * T
        
        # 3. Loss Estimate (LSL)
        lsl_qty = 0.0
        if inventory_batches:
            for batch in inventory_batches:
                days_left = batch['expiry_day'] - current_day
                coeff = 1.0
                if days_left <= 30: coeff = 0.5
                elif days_left <= 90: coeff = 0.8
                lsl_qty += batch['qty'] * (1.0 - coeff)
            
        # 4. Inventory Position (I)
        total_inventory = current_inventory_qty + pipeline_qty
        
        # 5. Calculate Order (OR)
        # OR = SS + (Y * T) - (I - LSL)
        effective_inventory = total_inventory - lsl_qty
        target_level = ss_qty + cycle_stock
        
        order_qty = max(0.0, target_level - effective_inventory)
        return order_qty

    def check_expiration(self, inventory_batches: List[Dict[str, Any]], current_day: int) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Check for expired batches in inventory.
        Returns (expired_qty, updated_batches)
        """
        expired_qty = 0.0
        updated_batches = []
        
        for batch in inventory_batches:
            if batch['expiry_day'] <= current_day:
                expired_qty += batch['qty']
            else:
                updated_batches.append(batch)
                
        return expired_qty, updated_batches

    def consume_stock(self, inventory_batches: List[Dict[str, Any]], demand_qty: float) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Consume stock (FIFO/FEFO) to satisfy demand.
        Returns (satisfied_qty, updated_batches)
        """
        # Sort by expiry (FEFO) - First Expiry First Out
        inventory_batches.sort(key=lambda x: x['expiry_day'])
        
        satisfied_qty = 0.0
        remaining_demand = demand_qty
        updated_batches = []
        
        for batch in inventory_batches:
            if remaining_demand <= 0:
                updated_batches.append(batch)
                continue
                
            if batch['qty'] > remaining_demand:
                # Partial batch consumption
                satisfied_qty += remaining_demand
                batch['qty'] -= remaining_demand
                updated_batches.append(batch)
                remaining_demand = 0
            else:
                # Full batch consumption
                satisfied_qty += batch['qty']
                remaining_demand -= batch['qty']
                # Batch removed (not appended to updated)
        
        return satisfied_qty, updated_batches
