from typing import Dict, TypedDict, Any

class VolatilityParams(TypedDict):
    noise_mult: float
    temp_sens: float
    flu_sens: float
    season_sens: float
    rain_sens: float        # New: Rainfall sensitivity (Log rain)
    validity_days: int
    safety_factor: float    # Adjustment factor for the inventory formula
    service_level: float    # Target Service Level (e.g. 0.95)
    z_score: float          # Confidence Coefficient (Z) corresponding to Service Level

class ThesisParams:
    """
    Central repository for Thesis Statistical Constraints.
    Ref: docs/THESIS_STATISTICS_BASELINE.md & docs/REFACTORING_GUIDE.md
    This file replaces hardcoded values in algorithms to ensure consistency.
    """
    
    # Global Statistical Settings
    CONFIDENCE_LEVEL = 0.95 # 95% Confidence Interval
    SIGNIFICANCE_LEVEL = 0.05 # Alpha
    
    # 0. Study Metadata (Thesis Section 1)
    SAMPLE_INFO = {
        'clinics': 7,
        'population': 83000, # Core service population
        'daily_visits_range': (30, 80),
        'period_start': '2023-01-01',
        'period_end': '2024-12-31',
        'train_split_date': '2024-08-31', # 20 months (83%)
        'test_split_date': '2024-09-01'   # 4 months (17%)
    }

    # 1. Classification Constraints (128 Total)
    VOLATILITY_COUNTS = {
        'LOW': 41,
        'MEDIUM': 63,
        'HIGH': 24 
    }
    TOTAL_SKUS = 128

    # 2. Algorithm Parameters per Volatility Category
    # These drive Generation (Op A) and Control (Op B)
    VOLATILITY_BEHAVIOR: Dict[str, VolatilityParams] = {
        'LOW': {
            'noise_mult': 0.2,      # CV < 0.2
            'temp_sens': 0.0,
            'flu_sens': 0.0,
            'season_sens': 0.32,    # Adjusted to match r=0.32 (Thesis Section 2)
            'rain_sens': 0.0,       # Low vol not affected by rain
            'validity_days': 720,   # Long shelf life -> Low Loss
            'safety_factor': 1.65,  # Z=1.65 for 95% Service Level
            'service_level': 0.95,
            'z_score': 1.65         # Explicit Confidence Coefficient
        },
        'MEDIUM': {
            'noise_mult': 0.6,      # 0.2 <= CV <= 0.5
            'temp_sens': 0.8,
            'flu_sens': 0.8,
            'season_sens': 0.5,
            'rain_sens': 0.3,       # Moderate rain impact
            'validity_days': 360,   # Standard 1 year
            'safety_factor': 1.96,  # Z=1.96 for ~97.5% Service Level (or just 1.5 multiplier)
            'service_level': 0.98,
            'z_score': 2.05         # 98% Service Level ~2.05
        },
        'HIGH': {
            'noise_mult': 2.5,      # CV > 0.5
            'temp_sens': 1.5,
            'flu_sens': 2.0,        # Hyper Sensitive to Flu/Weather
            'season_sens': 1.0,
            'rain_sens': 0.5,       # High sensitivity to all external factors
            'validity_days': 180,   # Short shelf life -> High Loss (Target 17.2%)
            'safety_factor': 2.33,  # Z=2.33 for 99% Service Level to prevent stockout
            'service_level': 0.99,
            'z_score': 2.33         # 99% Service Level ~2.33
        }
    }

    # 3. Target Baseline Metrics (Thesis Section 3 - Current State 2023)
    BASELINE_TARGETS = {
        'loss_rate': 0.172,     # 17.2%
        'stockout_rate': 0.031, # 3.1%
        'turnover_days': 44.6,  # 44.6 Days
        'backlog_rate': 0.158,  # 15.8%
        'funds_occupied': 285000, # Approximate Funds ~28.5 Wan
    }
    
    # NEW: Optimization Targets (Thesis Section 3.3.2 - Future State Goal)
    # Comparison Periods:
    # Baseline (Empirical): 2023.09 - 2023.12
    # Optimization (Improved): 2024.09 - 2024.12
    OPTIMIZATION_TARGETS = {
        'loss_rate': 0.139,     # Target 13.9% (-19.4%)
        'stockout_rate': 0.024, # Target 2.4% (-22.6%)
        'turnover_days': 38.7,  # Target 38.7 Days (-13.2%)
        'backlog_rate': 0.112,  # Target 11.2% (-29.1%)
        'funds_occupied': 242000 # Target 24.2 Wan (-15.1%)
    }

    # 4. Replenishment Strategy Optimization (Thesis Section 3.3.1)
    # These parameters define the "Improved Strategy" (Operator B_new)
    REPLENISHMENT_STRATEGY = {
        'common_params': {
            'lead_time': 4,       # L=4 Days
            'z_score': 1.645,     # 95% Service Level
        },
        'LOW': {
            'safety_stock_target': 8,   # Avg SS = 8 units
            'review_period_days': 30,   # Monthly
            'temp_replenishment': False
        },
        'MEDIUM': {
            'safety_stock_target': 19,  # Avg SS = 19 units
            'review_period_days': 15,   # Semi-monthly
            'temp_replenishment': False
        },
        'HIGH': {
            'safety_stock_target': 39,  # Avg SS = 39 units
            'review_period_days': 15,   # Semi-monthly
            'temp_replenishment': True, # Add 1 temp check in Flu/Winter
            'mape_target': 0.105        # Target MAPE 10.5%
        }
    }
    
    # 5. Expiry Value Coefficients (Thesis Section 5)
    # Used for valuation, not necessarily physical loss, but can be used for "Economic Loss"
    EXPIRY_COEFFS = {
        'fresh': 1.0,   # > 90 days
        'risk': 0.8,    # 30-90 days
        'critical': 0.5 # <= 30 days
    }
    
    # NEW: ARIMA Model Ground Truth Targets (Thesis Section 3.2.1 & 3.2.2)
    # These dictate how "predictable" the generated data should be.
    ARIMA_TARGETS = {
        'LOW': {
            'order': (1,0,1), 
            'r_squared': 0.89,
            'aic': 186.3,
            'mape': 0.072  # Avg MAPE (for entire set, but mainly low vol is easiest)
        },
        'MEDIUM': {
            'order': (2,1,2),
            'r_squared': 0.83,
            'aic': 214.7
        },
        'HIGH': {
            'order': (3,1,3),
            'r_squared': 0.78,
            'aic': 238.5,
            'mape': 0.105  # Specific target for High Vol (from 13.7% -> 10.5%)
        }
    }
    
    # 6. VIF Constraint
    
    # 5. VIF Constraint
    # VIF < 5 means external factors (Temp, Flu) should not be perfectly correlated.
    MAX_VIF = 5.0
