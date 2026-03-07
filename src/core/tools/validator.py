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
        Validates both Baseline (pre-optimization) and Optimized (post-optimization) periods.
        """
        results = []
        
        # Determine Split Date
        split_date = pd.Timestamp(ThesisParams.SAMPLE_INFO['test_split_date'])
        
        for drug in test_drugs:
            # Run simulation
            sim = MCMC_Transition(self.config, drug, external_data)
            # Ensure we cover enough time for both periods
            # Config default starts 2024-01-01. Split is 2024-09-01.
            # We run 365 days (2024)
            df = sim.run_simulation(duration_days=365) 
            results.append(df)
            
        full_df = pd.concat(results, ignore_index=True)
        
        # Split Data
        baseline_df = full_df[full_df['date'] < split_date]
        optimized_df = full_df[full_df['date'] >= split_date]
        
        # Validate Baseline
        baseline_metrics = self._calculate_metrics(baseline_df)
        baseline_compliance = self._check_compliance(baseline_metrics, ThesisParams.BASELINE_TARGETS)
        
        # Validate Optimized
        optimized_metrics = self._calculate_metrics(optimized_df)
        optimized_compliance = self._check_compliance(optimized_metrics, ThesisParams.OPTIMIZATION_TARGETS)
        
        return {
            'overall_status': 'PASS' if (all(baseline_compliance.values()) and all(optimized_compliance.values())) else 'FAIL',
            'baseline': {
                'metrics': baseline_metrics,
                'compliance': baseline_compliance,
                'targets': ThesisParams.BASELINE_TARGETS
            },
            'optimized': {
                'metrics': optimized_metrics,
                'compliance': optimized_compliance,
                'targets': ThesisParams.OPTIMIZATION_TARGETS
            }
        }

    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {
                'loss_rate': 0.0, 'stockout_rate': 0.0, 'turnover_days': 0.0,
                'backlog_rate': 0.0, 'funds_occupied': 0.0
            }
            
        total_demand = df['demand'].sum()
        total_loss = df['loss'].sum()
        total_sales = df['sales'].sum()
        avg_inventory = df['inventory'].mean()
        
        # Loss Rate (Defined as Loss / (Sales + Loss) or Loss/TotalInflow)
        # Using Loss / (Sales + Loss) as robust metric
        loss_rate = total_loss / (total_sales + total_loss) if (total_sales + total_loss) > 0 else 0
        
        # Stockout Rate
        stockout_days = df['stockout_flag'].sum()
        total_days = len(df['date'].unique()) # Correct day count
        stockout_rate = stockout_days / (total_days * len(df['drug_id'].unique())) # Per Drug-Day
        # Actually standard definition is: Prob of Stockout on a day.
        # If df contains multiple drugs, we average over all rows?
        # df contains rows for drug * days. So len(df) is total drug-days.
        stockout_rate = stockout_days / len(df) if len(df) > 0 else 0
        
        # Turnover Days
        # Turnover Ratio = Annualized COGS / Average Inventory
        # We have partial data. 
        # Daily Turnover = Sales / Avg Inv
        # Turnover Days = 1 / Daily Turnover ? No.
        # Turnover Ratio (for period) = Total Sales / Avg Inv.
        # Turnover Days = Period Days / Turnover Ratio = Period Days * Avg Inv / Total Sales
        turnover_days = (total_days * avg_inventory) / total_sales if total_sales > 0 else 0
        
        # Backlog Rate (Not explicitly tracked in basic sim, assuming stockout ~ backlog risk)
        # Or define as Stockout Rate * Factor
        backlog_rate = stockout_rate * 5 # Approximation or placeholder
        
        # Funds Occupied (Avg Inv * Unit Cost). 
        # Since we don't have Unit Cost in DF, we can't compute total value easily unless we assume avg cost.
        # Thesis Target is 28.5 Wan (285,000).
        # We calculate "Average Inventory Units" here. Need cost.
        # Check if 'run_simulation' stored drug info in DF? No.
        # But validator is initialized with Drug Info? No, run_validation takes drug_info.
        # Actually this calculates metrics for the DF generated from ONE drug or MANY?
        # If many, need sum(Inv * Price).
        
        # Simplified: Use a default price if not available to make the unit scale meaningful
        # Thesis Avg Price ~ 30-40 RMB?
        # If Funds=285,000 and turnover ~45 days. Sales=Funds*(365/45) ~ 2.3M
        # Let's assume average unit price = 35 RMB
        avg_price = 35.0
        
        funds_occupied = avg_inventory * avg_price
        
        return {
            'loss_rate': loss_rate,
            'stockout_rate': stockout_rate,
            'turnover_days': turnover_days,
            'backlog_rate': backlog_rate,
            'funds_occupied': funds_occupied
        }

    def _check_compliance(self, metrics: Dict[str, float], targets: Dict[str, float]) -> Dict[str, Any]:
        """
        Check if metrics comply with targets and return deviations.
        Returns detailed compliance info including deviation for optimization.
        """
        compliance = {}
        deviations = {}
        
        # We use a relaxed tolerance for "Pass" because simulation is stochastic
        # 20% relative tolerance?
        for key in ['loss_rate', 'stockout_rate', 'turnover_days']:
            if key in targets:
                target = targets[key]
                actual = metrics[key]
                
                # Dynamic tolerance based on magnitude
                tol = target * 0.25 # Allow 25% deviation
                
                diff = actual - target
                deviations[key] = diff
                
                compliance[key] = abs(diff) <= tol
                
        return {
            'passed': all(compliance.values()),
            'details': compliance,
            'deviations': deviations
        }
