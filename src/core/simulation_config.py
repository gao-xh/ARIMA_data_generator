from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

@dataclass
class SimulationConfig:
    """
    Core simulation parameters.
    Allows tuning the data generation range, clinic scaling, and external sensitivity.
    """
    
    # 1. Temporal Range (Thesis requires: 2024-2025 full cycle)
    start_date: pd.Timestamp = pd.Timestamp("2024-01-01")
    end_date: pd.Timestamp = pd.Timestamp("2025-12-31")
    
    # 2. Clinic Scaling (Simulates different sizes, e.g., Street Clinic vs Village Clinic)
    # Factor > 1.0 means higher volume, < 1.0 means lower volume
    clinics: Dict[str, float] = None

    # 3. Randomness Control (Thesis: "N +- x")
    # Base noise level (coefficient of variation for daily sales)
    random_noise_sigma: float = 0.2  # +/- 20% standard deviation
    
    # 4. External Sensitivity (Thesis Conclusion 1: External factors matter)
    # These coefficients control how much Temp/Flu modifies base demand.
    temp_sensitivity: float = 1.0   # Cold weather impact multiplier
    flu_sensitivity: float = 1.2    # Flu outbreak impact multiplier (Reduced from 2.5 to avoid crazy stockouts)
    rain_sensitivity: float = 0.0   # Rainfall impact multiplier
    
    # 7. Inventory Strategy
    # Thesis Baseline: "Empirical Mode" (2023-2024)
    # Stats Target: Turnover ~44.6 days, Stockout ~3.1%, Loss ~17.2%
    # Analysis:
    # - To get 45 days turnover with Monthly Review (30 days):
    #   Avg Stock approx = SafetyStock + (OrderQty/2)
    #   OrderQty covers 30 days. Half is 15.
    #   To get 45 total, SafetyStock needs to be ~30 days.
    #   So Safety Factor ~ 2.0 (2.0 * 15? No. Target = Factor * Daily * Period?)
    #   Usuall Target = Daily * Days * Factor.
    #   If Factor=2.0, R=30. Target = 60 days.
    #   Avg Stock = 60 - 15 = 45 days. -> MATCHES 44.6 Days.
    replenishment_days: int = 30      # R=30 (Monthly)
    safety_stock_factor: float = 2.0  # High factor causes High Inventory (44.6 days) & Loss (17.2%)
    initial_stock_days: int = 14      # Initial inventory in days of supply
    validity_days: int = 180          # Shortened theoretical validity to trigger 17% loss


    # 6. Operational Constraints (New Thesis Factors)
    # "Inventory update lag 2-3 days" -> Simulates Excel logging delay
    info_lag_days_min: int = 2
    info_lag_days_max: int = 3
    
    # 8. Active Clinic Profile (Scale Factor)
    # 1.0 = Standard Community, 2.0 = High Volume, 0.5 = Village Station
    active_clinic_scale: float = 1.0
    # "Delivery cycle uncertainty 3-5 days" -> Simulates vendor inefficiency
    lead_time_min: int = 3
    lead_time_max: int = 5
    
    def __post_init__(self):
        if self.clinics is None:
            # Default 7 clinics from thesis
            self.clinics = {
                "Clinic_A": 1.2, # Large
                "Clinic_B": 1.0, # Medium
                "Clinic_C": 0.8, # Small
                "Clinic_D": 1.0,
                "Clinic_E": 0.9,
                "Clinic_F": 1.1,
                "Clinic_G": 0.7  # Very Small
            }