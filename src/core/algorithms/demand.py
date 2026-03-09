import pandas as pd
from typing import Dict, Any, List
import random
import numpy as np
from src.core.causal_impact import CausalImpact
from src.core.simulation_config import SimulationConfig
from src.core.thesis_params import ThesisParams

class DemandModel:
    """
    Operator A: Demand Generation Function
    Thesis Logic: D_t = Base * f(Season) * f(Temp) * f(Flu) * Category_Sens + Noise
    """
    def __init__(self, config: SimulationConfig, drug_info: Dict[str, Any], volatility_cat: str):
        self.config = config
        self.volatility_cat = volatility_cat
        
        # 1. Base Demand Setup (from CSV)
        try:
            self.raw_demand = float(drug_info.get('日均销量', 5))
        except (ValueError, TypeError):
            self.raw_demand = 5.0
            
        # 2. Extract Functional Category (Thesis Logic Mapping)
        self.functional_category = self._get_functional_category(drug_info)
        
        # 3. Set Sensitivity Multipliers (Thesis Statistics Enforcement)
        # Using ThesisParams instead of hardcoding
        self._set_sensitivities()
        
        # 4. State for Correlated Noise (AR(1) Process)
        # To make demand look more realistic and "organic", less like white noise
        self.noise_state = 0.0

    def _get_functional_category(self, drug_info: Dict[str, Any]) -> str:
        cat_str = str(drug_info.get('药品品类', 'Unknown'))
        if any(x in cat_str for x in ['呼吸', '感冒', '咳', '肺', '炎', '抗生素', '解热']):
            return 'RESPIRATORY'
        elif any(x in cat_str for x in ['慢病', '心血管', '降压', '降糖', '降脂', '慢性']):
            return 'CHRONIC'
        return 'OTHER'

    def _set_sensitivities(self):
        # Get behavior parameters from ThesisParams reference
        params = ThesisParams.VOLATILITY_BEHAVIOR.get(self.volatility_cat, ThesisParams.VOLATILITY_BEHAVIOR['MEDIUM'])
        
        self.noise_mult = params['noise_mult']
        self.temp_sens = params['temp_sens']
        self.flu_sens = params['flu_sens']
        self.season_sens = params['season_sens']
        # Default to 0.0 if not yet added to all dicts in ThesisParams (though it should be)
        self.rain_sens = params.get('rain_sens', 0.0)
        self.weekend_mult = params.get('weekend_mult', 1.0)

    def generate(self, current_date: pd.Timestamp, external_factors: pd.Series, clinic_scale: float) -> float:
        """Calculate theoretical demand for the day."""
        temp = external_factors.get('平均气温2m(℃)', 20.0) # Updated column name from CSV
        if pd.isna(temp): temp = external_factors.get('平均气温', 20.0)

        flu_rate = external_factors.get('ILI%', 0.0)
        
        rain = external_factors.get('平均降水量(mm)', 0.0)
        
        # Base Demand
        demand = self.raw_demand * clinic_scale
        
        # Seasonality
        season_impact = CausalImpact.calculate_seasonality_impact(1.0, current_date.month, self.functional_category)
        # Dampen seasonality for low volatility
        if self.season_sens < 1.0:
            season_impact = (season_impact - 1.0) * self.season_sens + 1.0
        demand *= season_impact
        
        # Temp & Flu Impact (Gated by Sensitivity)
        if self.temp_sens > 0:
            eff_temp_sens = self.config.temp_sensitivity * self.temp_sens
            demand = CausalImpact.calculate_temp_impact(demand, temp, self.functional_category, eff_temp_sens)
            
        if self.flu_sens > 0:
            eff_flu_sens = self.config.flu_sensitivity * self.flu_sens
            demand = CausalImpact.calculate_flu_impact(demand, flu_rate, self.functional_category, eff_flu_sens)
            
        # Rainfall Impact (New H1 Variable)
        if self.rain_sens > 0:
            demand = CausalImpact.calculate_rainfall_impact(demand, rain, self.rain_sens)

        # Weekend Impact (New: Weekly Seasonality)
        if hasattr(self, 'weekend_mult') and self.weekend_mult != 1.0:
             demand = CausalImpact.calculate_weekend_impact(demand, current_date, self.weekend_mult)

        # 6. Apply Correlated Noise (AR(1) Process)
        # Replaces simple white noise to reduce "high frequency jitter" and make it look more organic
        
        # Base Sigma
        base_sigma = self.config.random_noise_sigma * self.noise_mult
        
        # AR(1) Parameter (Rho) - Controls smoothness/memory
        # 0.0 = White Noise, 1.0 = Random Walk
        # 0.6 is a good balance for daily sales "momentum"
        rho = 0.6 
        
        # Generate new innovation (White Noise part)
        # Use simple normal distribution
        innovation = np.random.normal(0, base_sigma)
        
        # Update State: x_t = rho * x_{t-1} + sqrt(1 - rho^2) * epsilon_t
        # This ensures the variance of the process remains equal to base_sigma^2
        self.noise_state = rho * self.noise_state + np.sqrt(1 - rho**2) * innovation
        
        # Clamp state to avoid explosion (e.g. -50% to +80%)
        curr_noise = max(-0.5, min(0.8, self.noise_state))
        
        # Apply Multiplicative Noise
        demand *= (1 + curr_noise)
        
        # 7. Occasional "Burst" Events (Rare, independent of AR process)
        # Simulate bulk orders or sudden local events
        # Probability: 1% per day
        if np.random.random() < 0.01:
             burst = np.random.uniform(1.2, 1.5) # 20-50% spike
             demand *= burst
        
        return max(0, demand)
