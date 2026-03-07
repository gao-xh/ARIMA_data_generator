import pandas as pd
from typing import Dict, Any, List
import logging
from src.core.algorithms.mcmc_transition import MCMC_Transition
from src.core.simulation_config import SimulationConfig
from src.core.thesis_params import ThesisParams

class ThesisValidator:
    """
    Self-Check Tool: Verifies if generated data conforms to Thesis Statistics.
    Baseline targets loaded from ThesisParams
    """
    
    # Tolerances remain hardcoded as policy logic, or could be moved to config too
    TOLERANCE = {
        'loss_rate': 0.02,
        'stockout_rate': 0.01,
        'turnover_days': 5.0
    }

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        # Load targets from ThesisParams
        self.TARGETS = ThesisParams.BASELINE_TARGETS

    def run_validation(self, test_drugs: List[Dict[str, Any]], external_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run a simulation on a subset of drugs to validate statistical alignment.
        """
        results = []
        
        for drug in test_drugs:
            # Run simulation
            sim = MCMC_Transition(self.config, drug, external_data)
            df = sim.run_simulation(duration_days=365) # 1 Year check
            results.append(df)
            
        full_df = pd.concat(results, ignore_index=True)
        
        # Calculate Metrics
        metrics = self._calculate_metrics(full_df)
        
        # Compare with Targets
        compliance = self._check_compliance(metrics)
        
        return {
            'metrics': metrics,
            'compliance': compliance,
            'status': 'PASS' if all(compliance.values()) else 'FAIL'
        }

    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        total_demand = df['demand'].sum()
        total_loss = df['loss'].sum()
        total_sales = df['sales'].sum()
        avg_inventory = df['inventory'].mean()
        
        # Loss Rate = Total Loss / (Total Sales + Total Loss) ?? 
        # Or Loss / Total Inflow? 
        # Thesis usually: Loss Rate = Loss / Demand or Loss / (Sales + Loss)
        # Let's assume Loss / (Sales + Loss) for now as "Inventory Shrinkage"
        loss_rate = total_loss / (total_sales + total_loss) if (total_sales + total_loss) > 0 else 0
        
        # Stockout Rate = Days with stockout / Total Days
        stockout_days = df['stockout_flag'].sum()
        total_days = len(df)
        stockout_rate = stockout_days / total_days if total_days > 0 else 0
        
        # Turnover Days = 365 / (COGS / Avg Inv)
        # COGS ~ Sales (approx)
        turnover_ratio = total_sales / avg_inventory if avg_inventory > 0 else 0
        turnover_days = 365 / turnover_ratio if turnover_ratio > 0 else 0
        
        return {
            'loss_rate': loss_rate,
            'stockout_rate': stockout_rate,
            'turnover_days': turnover_days
        }

    def _check_compliance(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        compliance = {}
        
        # Loss Rate Check
        diff_loss = abs(metrics['loss_rate'] - self.TARGETS['loss_rate'])
        compliance['loss_rate'] = diff_loss <= self.TOLERANCE['loss_rate']
        
        # Stockout Rate Check
        diff_stock = abs(metrics['stockout_rate'] - self.TARGETS['stockout_rate'])
        compliance['stockout_rate'] = diff_stock <= self.TOLERANCE['stockout_rate']
        
        # Turnover Check
        diff_turnover = abs(metrics['turnover_days'] - self.TARGETS['turnover_days'])
        compliance['turnover_days'] = diff_turnover <= self.TOLERANCE['turnover_days']
        
        return compliance
