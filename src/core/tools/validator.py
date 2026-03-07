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

    def validate_dataset(self, full_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate an existing DataFrame (e.g. from file or generation) against stats.
        Assumes DF has 'date' column.
        """
        if 'date' not in full_df.columns:
             # Try to parse or assume index?
             return {'error': 'No date column'}
             
        full_df['date'] = pd.to_datetime(full_df['date'])
        
        # Determine Split Date
        split_date = pd.Timestamp(ThesisParams.SAMPLE_INFO['test_split_date'])
        
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
            'overall_status': 'PASS' if (baseline_compliance['passed'] and optimized_compliance['passed']) else 'WARN',
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

    def run_validation(self, test_drugs: List[Dict[str, Any]], external_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run a NEW simulation on a subset of drugs to validate statistical alignment.
        Validates both Baseline (pre-optimization) and Optimized (post-optimization) periods.
        """
        results = []
        
        for drug in test_drugs:
            # Run simulation
            sim = MCMC_Transition(self.config, drug, external_data)
            # Ensure we cover enough time for both periods
            # Config default starts 2024-01-01. Split is 2024-09-01.
            # We run 365 days (2024)
            df = sim.run_simulation(duration_days=365) 
            results.append(df)
            
        full_df = pd.concat(results, ignore_index=True)
        return self.validate_dataset(full_df)

    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty:
            return {
                'loss_rate': 0.0, 'stockout_rate': 0.0, 'turnover_days': 0.0,
                'backlog_rate': 0.0, 'funds_occupied': 0.0
            }
            
        total_loss = df['loss'].sum()
        total_sales = df['sales'].sum()
        avg_inventory = df['inventory'].mean()
        total_days = len(df['date'].unique()) # Correct day count
        
        # Funds Occupied: Avg Inventory * Price
        if 'unit_price' in df.columns:
            funds_occupied = (df['inventory'] * df['unit_price']).mean()
        else:
            funds_occupied = avg_inventory * 35.0  # Fallback

        # Loss Rate: Loss / (Sales + Loss)
        throughput = total_sales + total_loss
        loss_rate = total_loss / throughput if throughput > 0 else 0
        
        # Stockout Rate: Days with stockout / Total Drug-Days
        stockout_days = df['stockout_flag'].sum()
        stockout_rate = stockout_days / len(df) if len(df) > 0 else 0
        
        # Turnover Days: 365 * Avg Inventory / Total Annualized Sales
        # Daily Sales Rate
        sales_per_day = total_sales / total_days if total_days > 0 else 0
        turnover_days = avg_inventory / sales_per_day if sales_per_day > 0 else 0
        
        # Backlog Rate (Overstock Rate): Ratio of Days with > 90 Days Supply
        overstock_threshold = 90 * sales_per_day
        # If sales per day is 0, everything is overstock if inventory > 0
        if sales_per_day == 0:
            overstock_days = len(df[df['inventory'] > 0])
        else:
            overstock_days = len(df[df['inventory'] > overstock_threshold])
            
        backlog_rate = overstock_days / len(df) if len(df) > 0 else 0
        
        return {
            'loss_rate': loss_rate,
            'stockout_rate': stockout_rate,
            'turnover_days': turnover_days,
            'backlog_rate': backlog_rate,
            'funds_occupied': funds_occupied
        }

    def generate_markdown_report(self, validation_result: Dict[str, Any]) -> str:
        """
        Generates a Markdown report comparing Generated Stats vs Thesis Targets.
        """
        baseline = validation_result['baseline']
        optimized = validation_result['optimized']
        
        lines = []
        lines.append("# Statistical Validation Report (生成数据统计验证报告)\n")
        
        def add_section(title, data, target_period):
            lines.append(f"## {title} Period ({target_period})")
            lines.append("| Metric (指标) | Generated (生成值) | Thesis Target (论文目标) | Deviation (偏差) | Status |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            
            # Helper for row
            metrics = data['metrics']
            targets = data['targets']
            
            row_map = {
                'loss_rate': ('Loss Rate (损耗率)', True, 1.0),
                'stockout_rate': ('Stockout Rate (缺货率)', True, 1.0),
                'backlog_rate': ('Backlog Rate (积压率)', True, 1.0),
                'turnover_days': ('Turnover Days (周转天数)', False, 1.0),
                'funds_occupied': ('Funds Occupied (资金占用)', False, 1.0/10000) # Show in Wan
            }
            
            for key, (label, is_pct, mult) in row_map.items():
                actual = metrics.get(key, 0.0)
                target = targets.get(key, 0.0)
                diff = actual - target
                
                if is_pct:
                    s_act = f"{actual*100:.1f}%"
                    s_tgt = f"{target*100:.1f}%"
                    s_dif = f"{diff*100:+.1f}%"
                else:
                    s_act = f"{actual*mult:.1f}"
                    s_tgt = f"{target*mult:.1f}"
                    s_dif = f"{diff*mult:+.1f}"
                    if key == 'funds_occupied': s_act += " Wan"
                
                # Simple Pass/Fail logic (e.g. within 20% relative error or absolute diff)
                # For demonstration, we just mark it.
                status = "✅" if abs(diff) < (target * 0.2 if target else 0.1) else "⚠️"
                
                lines.append(f"| {label} | {s_act} | {s_tgt} | {s_dif} | {status} |")
            lines.append("")

        add_section("1. Baseline (经验模式)", baseline, "2023.01 - 2024.08")
        add_section("2. Optimized (优化后)", optimized, "2024.09 - 2024.12")
        
        return "\n".join(lines)

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
