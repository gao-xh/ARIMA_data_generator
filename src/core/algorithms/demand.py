import pandas as pd
from typing import Dict, Any, List
import random
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

    def generate(self, current_date: pd.Timestamp, external_factors: pd.Series, clinic_scale: float) -> float:
        """Calculate theoretical demand for the day."""
        temp = external_factors.get('平均气温', 20.0)
        flu_rate = external_factors.get('ILI%', 0.0)
        
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
            
        # Noise
        eff_sigma = self.config.random_noise_sigma * self.noise_mult
        demand = CausalImpact.apply_random_noise(demand, eff_sigma)
        
        return max(0, demand)
