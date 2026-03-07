import math
import random
from typing import Dict, Optional

class CausalImpact:
    """
    Implements Thesis Conclusions: 
    External factors (Temperature, Flu) significantly affect demand,
    which manual inventory management cannot predict efficiently.
    """

    @staticmethod
    def calculate_seasonality_impact(base_demand: float, month: int, drug_category: str) -> float:
        """
        Thesis Conclusion: Seasonal patterns exist.
        Simple model:
        - Respiratory/Cold: High in Win/Spr (12, 1, 2, 3), Low in Summer.
        - Gastrointestinal: High in Summer (6, 7, 8). (Not implemented yet explicitly, treated as base)
        - Chronic: Stable.
        """
        # "RESPIRATORY" is the internal flag set in DrugState
        is_resp = drug_category == 'RESPIRATORY' or drug_category in ['呼吸系统用药', '感冒用药', '解热镇痛用药']
        
        if is_resp:
            # Winter Peak
            if month in [12, 1, 2]: return base_demand * 1.3
            if month in [3, 4, 11]: return base_demand * 1.1
            if month in [6, 7, 8]: return base_demand * 0.7
            
        return base_demand

    @staticmethod
    def calculate_temp_impact(base_demand: float, temperature: float,  
                             drug_category: str, sensitivity: float = 1.0) -> float:
        """
        Thesis Conclusion: Respiratory/Cardiovascular drugs are sensitive to cold.
        When Temp < Throwhold (Mean - Std), demand increases non-linearly.
        
        Args:
            base_demand: Normal daily sales.
            temperature: Current Daily Avg Temp.
            drug_category: E.g., 'Respiratory', 'Chronic'.
            sensitivity: User-controlled multiplier (default 1.0).
            
        Returns:
            Adjusted demand (float).
        """
        # Thesis Assumption: Cold threshold around 10C or 5C depending on region
        TEMP_THRESHOLD_COLD = 10.0
        
        is_sensitive = drug_category == 'RESPIRATORY' or drug_category in ['呼吸系统用药', '心脑血管用药', '感冒用药']
        
        # Only specific categories react
        if not is_sensitive:
            return base_demand

        if temperature < TEMP_THRESHOLD_COLD:
            # Impact Function: Exponential decay as temp drops
            # Example: As temp goes from 10 -> 0 -> -10, demand increases
            delta_t = TEMP_THRESHOLD_COLD - temperature
            # Impact factor: e.g., 1 + (0.05 * delta * sensitivity)
            # If temp is 0C (delta=10), factor = 1 + 0.5 = 1.5x
            impact_factor = 1.0 + (0.05 * delta_t * sensitivity)
            # Cap at 3x
            impact_factor = min(impact_factor, 3.0)
            return base_demand * impact_factor
            
        return base_demand

    @staticmethod
    def calculate_flu_impact(base_demand: float, flu_rate: float, 
                            drug_category: str, sensitivity: float = 1.0) -> float:
        """
        Thesis Conclusion: Flu outbreaks (ILI%) cause demand spikes for specific drugs.
        Manual ordering misses these spikes -> Stockouts.
        
        Args:
            flu_rate: ILI% (Influenza-like Illness rate), usually 0.02 - 0.10.
            sensitivity: User-controlled multiplier.
        """
        # Thesis Assumption: Outbreak threshold ~ 5% ILI
        FLU_THRESHOLD = 0.05
        
        is_sensitive = drug_category == 'RESPIRATORY' or drug_category in ['呼吸系统用药', '抗感染用药', '解热镇痛用药', '感冒用药']

        if not is_sensitive:
            return base_demand
        
        # Robust handling for % vs ratio
        effective_rate = flu_rate
        if effective_rate > 100: effective_rate /= 100 
        
        threshold = FLU_THRESHOLD 
        if effective_rate > 1.0: # If data is > 1.0, it treats as percentage points (e.g. 5.0)
            threshold = 5.0
            
        if effective_rate > threshold:
             # Exponential spike
             gap = effective_rate - threshold
             # If points (gap=5) -> 1 + 1.0 = 2x
             # If ratio (gap=0.05) -> 1 + 0.01 (Too small)? Need normalization.
             if threshold < 1.0: # Ratio mode
                 gap = gap * 100 # Convert to points
                 
             multiplier = 1.0 + (gap * 0.2 * sensitivity)
             return base_demand * multiplier
             
        return base_demand

    @staticmethod
    def apply_random_noise(demand: float, sigma: float) -> int:
        """
        Thesis Conclusion: Real-world demand has noise ('N +- x').
        
        Args:
            demand: Calculated theoretical demand.
            sigma: Standard deviation (e.g., 0.2 means 20% variation).
        """
        if demand <= 0: return 0
        
        # Gaussian noise: N(0, sigma)
        noise_factor = random.gauss(0, sigma)
        
        # Clamp noise to reasonable bounds (e.g., +/- 3 sigma) to avoid negative demand
        noise_factor = max(-0.8, min(0.8, noise_factor))
        
        final_demand = demand * (1 + noise_factor)
        return int(round(max(0, final_demand)))
