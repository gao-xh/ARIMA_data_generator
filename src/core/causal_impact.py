import math
import random
import numpy as np
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
        - Gastrointestinal: High in Summer (6, 7, 8).
        - Chronic: Mild Winter effect (Cardio etc sensitive to cold).
        """
        # Standardize Category Checks
        cat_upper = drug_category.upper()
        # Include localized terms from CSV
        is_resp = 'RESPIRATORY' in cat_upper or any(x in cat_upper for x in ['呼吸', '感冒', '解热', '抗生物', '抗生素'])
        is_chronic = 'CHRONIC' in cat_upper or any(x in cat_upper for x in ['慢病', '心脑', '高血压', '心血管', '糖尿病'])
        is_gastro = 'GASTRO' in cat_upper or any(x in cat_upper for x in ['消化', '胃', '腹'])
        
        if is_resp:
            # Strong Winter/Spring Peak
            if month in [12, 1, 2]: return base_demand * 1.3
            if month in [3, 4, 11]: return base_demand * 1.1
            if month in [6, 7, 8]: return base_demand * 0.7
            
        elif is_chronic:
            # Mild Winter Peak (Cardiovascular constricts in cold)
            if month in [12, 1, 2]: return base_demand * 1.15
            if month in [3, 11]: return base_demand * 1.05
            
        elif is_gastro:
            # Summer Peak (Food spoilage)
            if month in [6, 7, 8]: return base_demand * 1.25
            
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
        
        cat_upper = drug_category.upper()
        # Standardize matching
        is_resp = 'RESPIRATORY' in cat_upper or any(x in cat_upper for x in ['呼吸', '感冒', '解热'])
        # Chronic (Cardio) is sensitive to cold -> blood pressure issues
        is_chronic = 'CHRONIC' in cat_upper or any(x in cat_upper for x in ['心脑', '高血压', '心血管'])
        
        # Only specific categories react
        if not (is_resp or is_chronic):
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
    def calculate_rainfall_impact(base_demand: float, rainfall_mm: float,
                                sensitivity: float = 1.0) -> float:
        """
        Thesis Variable: Log-Normal Rainfall (R')
        Formula: R' = ln(Rainfall + 1)
        
        Impact:
        - High rainfall typically reduces patient visits (access barrier).
        - However, for regression, it's just a factor X.
        - Here we model it as a slight dampener on general visits unless sensitivity is tuned otherwise.
        """
        if rainfall_mm <= 0:
            return base_demand
            
        # Log transformation as per thesis
        log_rain = np.log1p(rainfall_mm)
        
        # Default behavior: Rain reduces visits slightly (-5% per log unit * sensitivity)
        # 10mm rain -> log(11) ~= 2.4 -> 2.4 * 0.05 = 12% drop
        # But for 'High Volatility' drugs in Thesis, it might be included because it correlates.
        # We assume negative correlation for general patient traffic.
        
        factor = 1.0 - (0.05 * log_rain * sensitivity)
        return max(0.0, base_demand * factor)

    @staticmethod
    def apply_random_noise(demand: float, sigma: float) -> int:
        """
        Thesis Conclusion: Real-world demand has noise ('N +- x').
        
        Args:
            demand: Calculated theoretical demand.
            sigma: Standard deviation (e.g., 0.2 means 20% variation).
        """
        if demand <= 0: return 0
        
        # Adjust sigma minimum for small demand to ensure variation
        # If mean=5, sigma=0.04 (4%) -> std=0.2 -> No variation after rounding
        # Target: For mean=5, we want to see occasional 4, 6, 3, 7.
        # So sigma needs to be at least ~0.15 (15%) for small numbers.
        
        eff_sigma = max(sigma, 0.15) if demand < 20 else sigma
        
        try:
            # 1. Base Noise (Gaussian)
            noise_factor = np.random.normal(0, eff_sigma)
            
            # 2. Burst Event Injection (Lumpy Demand)
            # Simulate real-world events: Prescription refills, small outbreaks
            # Probability depends on sigma (higher volatility = more bursts)
            burst_prob = min(0.1, sigma * 0.2) 
            if np.random.random() < burst_prob:
                # Burst Multiplier: 1.5x to 3.0x for normal items
                # For very low volume (demand < 2), it could be just +1 or +2 units
                burst_mult = np.random.uniform(1.5, 3.0)
                noise_factor += (burst_mult - 1.0)
                
            # Clamp for stability, but allow higher upside
            # Lower bound: -0.9 (can almost wipe out demand)
            # Upper bound: +3.0 (can quadruple demand)
            noise_factor = max(-0.9, min(3.0, noise_factor))
            
            final_val = demand * (1 + noise_factor)
            final_val = max(0, final_val)
            
            # Stochastic Rounding (Crucial for Low Volume Items)
            # 5.2 -> 20% chance of 6, 80% chance of 5.
            # This preserves the mean (expected value) over time.
            floor_val = math.floor(final_val)
            prob_ceil = final_val - floor_val
            
            if np.random.random() < prob_ceil:
                return int(floor_val + 1)
            else:
                return int(floor_val)
                
        except Exception:
            # Fallback
            return int(round(demand))
