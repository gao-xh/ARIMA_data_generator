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
                params = ThesisParams.VOLATILITY_BEHAVIOR.get(self.volatility_cat, ThesisParams.VOLATILITY_BEHAVIOR['MEDIUM'])
                self.shelf_life = float(params['validity_days'])
        except (ValueError, TypeError):
            self.shelf_life = 365.0
            
        try:
            self.lead_time = int(drug_info.get('补货提前期', 3))      # Days
        except (ValueError, TypeError):
            self.lead_time = 3

        # Manual Replenishment: Use Initial Stock from CSV as the Target Ceiling?
        # If Initial Stock is provided, we should align our Replenishment Target to it.
        # This solves: "Inventory average should hover around initial stock"
        # However, Inventory sawtooths between Target and (Target - Usage).
        # So Target needs to be higher than Initial Stock if Initial Stock is the average.
        # But usually users mean "Fill up to Initial Stock".
        self.initial_stock_ref = drug_info.get('初始库存', None)
        try:
             if self.initial_stock_ref is not None:
                 self.initial_stock_ref = float(self.initial_stock_ref)
        except:
             self.initial_stock_ref = None

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
        Mode: 'BASELINE', 'OPTIMIZED', 'EMERGENCY'
        """
        if mode == 'EMERGENCY':
             # Emergency Logic: Check if critical low, verify pipeline, order up to safe level.
             # Threshold: Lead Time based (ensure explicit coverage during L).
             # Baseline Panic: 3 days. Optimized Safety Net: Lead Time (approx 3-6 days).
             
             # Use dynamic Lead Time or default 3
             lt_days = max(1.0, float(self.lead_time))
             
             # Metric 1: Immediate Danger Threshold (Stock < Lead Time demand)
             threshold = avg_daily_demand * lt_days
             
             if current_inventory_qty < threshold:
                 # Check Pipeline: If help on the way is enough, do not panic.
                 position = current_inventory_qty + pipeline_qty
                 
                 # Safety Target: Covers L + Review Period (approx 7-14 days buffer)
                 safety_target_days = lt_days + 14.0 
                 safety_target = avg_daily_demand * safety_target_days
                 
                 # Logic Update: If Position < Critical Level (Lead Time + small buffer), Panic Order.
                 # Only trigger if the pipeline + on_hand is dangerously low.
                 critical_level = avg_daily_demand * (lt_days + 2.0)
                 
                 if position < critical_level:
                     # Order up to Safety Target
                     needed = safety_target - position
                     return max(0.0, needed)
             return 0.0

        if mode == 'OPTIMIZED':
            return self._calculate_optimized_order(avg_daily_demand, current_inventory_qty, pipeline_qty, inventory_batches, current_day)
        else:
            return self._calculate_baseline_order(avg_daily_demand, demand_std, current_inventory_qty, pipeline_qty)

    def _calculate_baseline_order(self, avg_daily_demand: float, demand_std: float, 
                                current_inventory: float, pipeline_inventory: float) -> float:
        """
        Baseline Strategy (Empirical Mode):
        - Manual Periodic Review (R=30 usually)
        - Target Level Strategy
        """
        review_days = self.get_review_period('BASELINE')
        review_horizon = review_days + self.lead_time
        
        # 2. Set Target Level (The "Up-to" Level)
        # User Constraint: "Inventory amount should fluctuate around Initial Stock"
        # If Initial Stock is provided, we treat it as the "Target Ceiling" (S).
        # OR as the "Average Inventory".
        # Let's assume Target S = Max Stock.
        # Average Inventory will be approx S - (Demand * R / 2).
        # To make Average = Initial Stock, then S = Initial + (Demand * R / 2).
        
        if self.initial_stock_ref is not None and self.initial_stock_ref > 0:
            # Shift Target UP so that Average Inventory matches Initial Stock
            # Avg Inv = S - (Demand * R / 2) => S = Avg Inv + (Demand * R / 2)
            half_cycle_demand = (avg_daily_demand * review_days) / 2.0
            
            # However, if R is large (30 days), this shift is significant.
            # If user sees "Initial Stock" as the "Full Shelf", then we should use just Initial Stock.
            # But user said "Mean around Initial".
            target_level = self.initial_stock_ref + half_cycle_demand
        else:
            # Fallback to calculated safety stock
             ss_qty = self.safety_factor * demand_std * np.sqrt(review_horizon)
             target_level = (avg_daily_demand * review_horizon) + ss_qty
        
        inventory_position = current_inventory + pipeline_inventory
        
        # 3. Emergency Logic (Panic Ordering)
        # If Physical Inventory < 3 days coverage, Order immediately regardless of Review Period?
        # Typically handled outside of R-check loop, or R becomes 1.
        # But this function is called ONLY when Review happens (in MCMC_Transition).
        # Wait, MCMC calls this only "if day % review_period == 0".
        # So we can't implement Emergency Order HERE if it's not called daily.
        
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
        
        # Cycle Stock (Y_hat * T)
        # T from Strategy (30 or 15)
        T = strat['review_period_days'] # T
        
        # CORRECTED FORMULA (Periodic Review):
        # Protection Interval = T + L
        # SS = Z * sigma_D * sqrt(T + L)
        ss_qty = z * demand_std * np.sqrt(T + L)
        
        # Target Level = Demand_During(T + L) + SS
        # cycle_stock variable name kept for compatibility but now covers T+L period average demand
        cycle_stock = forecast_daily_demand * (T + L)
        
        # 3. Anticipated Expiration Loss (LSL) - RE-ENABLED via Logic Update (March 2026)
        # Goal: If existing stock will expire before the next replenishment arrives (or during coverage period),
        # we must treat it as "unavailable" for future demand and order more to compensate.
        # Protection Period = T + L
        protection_days = T + L
        anticipated_loss_qty = 0.0
        
        if inventory_batches:
            expiry_threshold = current_day + protection_days
            # Sum up qty of batches that will expire within the protection period
            # These units cannot cover the full demand of the period.
            for batch in inventory_batches:
                if batch['expiry_day'] <= expiry_threshold:
                    anticipated_loss_qty += batch['qty']
        
        # 4. Effective Inventory Position (I_eff)
        # I_eff = (Current On-Hand + Pipeline) - Anticipated Expiry
        total_inventory = current_inventory_qty + pipeline_qty
        effective_inventory = total_inventory - anticipated_loss_qty
        
        # 5. Calculate Order (OR)
        # OR = Target - I_eff
        # Target Level = SS + Cycle Stock
        target_level = ss_qty + cycle_stock
        
        order_qty = max(0.0, target_level - effective_inventory)
        return order_qty

    def check_expiration(self, inventory_batches: List[Dict[str, Any]], current_day: int) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Check for expired batches in inventory AND apply random natural loss (breakage/theft).
        Returns (expired_qty, updated_batches)
        """
        loss_qty = 0.0
        updated_batches = []
        
        # 1. Random Natural Loss Probability (0.1% chance per batch per day)
        # Simulates breakage, theft, or storage damage
        import random
        
        for batch in inventory_batches:
            # Check Expiry
            if batch['expiry_day'] <= current_day:
                loss_qty += batch['qty']
            else:
                # Check Natural Loss
                # Apply small probability of unit loss
                if batch['qty'] > 0 and random.random() < 0.05: # 5% daily chance of incident
                     loss_amount = min(batch['qty'], random.uniform(0.1, 1.0))
                     loss_qty += loss_amount
                     batch['qty'] -= loss_amount

                if batch['qty'] > 0.01:
                    updated_batches.append(batch)
                
        return loss_qty, updated_batches

    def consume_stock(self, inventory_batches: List[Dict[str, Any]], demand_qty: float, current_day: int) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Consume stock to satisfy demand.
        Logic:
        1. Try FEFO (First Expired First Out).
        2. Apply "Consumer Rejection" logic: If the FEFO batch is near expiry, 
           a portion of customers will skip it and demand fresher stock.
        """
        # Sort by expiry (FEFO) - First Expiry First Out
        inventory_batches.sort(key=lambda x: x['expiry_day'])
        
        satisfied_qty = 0.0
        remaining_demand = demand_qty
        updated_batches = []
        
        # 1. Process "Freshness Skipping" Logic
        # Iterate through batches, but allow "skipping" to simulate customers picking fresher items.
        # This leaves old stock on the shelf to eventually expire.
        
        temp_batches = [] # To store processed state
        
        for batch in inventory_batches:
            if remaining_demand <= 0:
                temp_batches.append(batch)
                continue
                
            qty_available = batch['qty']
            qty_to_take = 0.0
            
            # Smart Consumption Logic
            days_remaining = batch['expiry_day'] - current_day
            
            # Acceptance Probability based on Remaining Shelf Life
            # If > 180 days: 100% Acceptance
            # If < 30 days: 10% Acceptance (Most skip it)
            # If < 60 days: 40% Acceptance
            # If < 90 days: 70% Acceptance
            
            acceptance_rate = 1.0
            if days_remaining < 30: acceptance_rate = 0.1
            elif days_remaining < 60: acceptance_rate = 0.4
            elif days_remaining < 90: acceptance_rate = 0.7
            
            # Calculate potential purchase from this batch
            # Customer wants X, but only accepts rate*X from this "old" batch
            # The rest of the demand is deferred to the NEXT batch (if any)
            # BUT: If this is the LAST batch, they might be forced to take it (Stockout avoidance)
            # Let's assume they skip it and it becomes Lost Sale OR they take it if desperate?
            # User wants "Buy longer shelf life", implying they skip this and go to next.
            
            desired_from_this_batch = remaining_demand
            
            # "Skipping" behavior: Only a portion of demand interacts with this batch
            # Effectively, the demand "passes through" the bad batches looking for good ones.
            # But physically, we iterate FEFO.
            # So: We try to fill 'remaining_demand' from 'batch'.
            # BUT we only effectively draw `min(qty, remaining * accept)`?
            # No, if I skip, I want the next batch.
            pass # Logic implemented below
            
            # Actual implementation:
            # We split the demand into "Willing to take this" and "Must skip for fresher"
            # But since "Must skip" just adds to valid demand for next batch, it's the same as
            # "This batch can only satisfy X amount due to rejection".
            
            # However, if it's the ONLY batch left, do they take it?
            # Let's say 50% desperation rate if no fresher option.
            # For now, simplistic: Max consumption from this batch is limited by acceptance rate * its own qty?
            # No, limited by acceptance rate * REMAINING DEMAND.
            
            willing_to_buy = remaining_demand * acceptance_rate
            
            # Take what we can (limited by willingness AND availability)
            actually_taken = min(qty_available, willing_to_buy)
            
            # Update batch and demand
            batch['qty'] -= actually_taken
            satisfied_qty += actually_taken
            remaining_demand -= actually_taken # The satisfied part is gone.
            
            # The UNSATISFIED part (remaining_demand) continues to the next loop (fresh batch).
            # If batch had remaining qty (because we skipped it), it stays in inventory.
            
            if batch['qty'] > 0.001:
                temp_batches.append(batch)
                
        # If we went through all batches and still have demand (because we skipped old stuff 
        # but ran out of fresh stuff), do we go back?
        # Realistically, customers might leave (Lost Sales due to Quality).
        # Or we force-feed the old stuff if we want to minimize stockouts.
        # User said "Buy longer shelf life drug", implying if none available, they might not buy short one?
        # Let's assume strict preference. Unfulfilled demand here = Lost Sale due to "Quality/Expiry Aversion".
        # This increases Stockout Rate (technically "Quality Stockout").
        # AND it leaves old stock to expire (Loss Rate increases). This meets the user requirement perfectly.

        updated_batches = temp_batches
        return satisfied_qty, updated_batches
