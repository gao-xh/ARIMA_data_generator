import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

class EnvironmentGenerator:
    """
    Simulates external environmental factors (Temperature, Rainfall, ILI%) 
    when historical data is missing (e.g., forecasting into 2026-2027).
    
    Algorithms based on Thesis H1 (Climate Impact) & H3 (Epidemic Stochasticity).
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        
    def generate_year(self, year: int) -> pd.DataFrame:
        """
        Generate a full year of daily environmental data.
        Returns DataFrame with columns: ['Date', 'Temp', 'Rainfall', 'ILI%']
        """
        dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
        days = len(dates)
        day_of_year = np.array(range(days))
        
        # 1. Temperature Generation (Sinusoidal + Noise)
        # Thesis: Low temp drives Respiratory issues.
        # Temp ~ 15 + 15*cos(...) + Noise
        t_base = 15.0
        t_amp = 15.0
        # Peak heat ~ July 15 (Day 196), Peak cold ~ Jan 15
        # Cosine peak is at 0. So we shift by (d - 196).
        temp_season = t_base + t_amp * np.cos(2 * np.pi * (day_of_year - 196) / 365.0)
        temp_noise = self.rng.normal(0, 3.0, days) # Daily weather variation
        temps = temp_season + temp_noise
        
        # 2. Rainfall Generation (Stochastic Intermittent)
        # Prob of rain varies by season (Summer wetter)
        # P(Rain) ~ 0.2 (Winter) to 0.5 (Summer)
        p_rain_season = 0.35 + 0.15 * np.cos(2 * np.pi * (day_of_year - 180) / 365.0)
        is_raining = self.rng.random(days) < p_rain_season
        
        # Rain amount (Gamma distribution when raining)
        # Mean amount also seasonality driven (Summer storms intense)
        rain_amounts = np.zeros(days)
        gamma_shape = 2.0
        gamma_scale = 5.0 + 5.0 * np.cos(2 * np.pi * (day_of_year - 180) / 365.0) # Seasonal intensity
        
        generated_rain = self.rng.gamma(gamma_shape, gamma_scale, days)
        rain_amounts[is_raining] = generated_rain[is_raining]
        
        # 3. ILI% (Influenza-Like Illness) Generation
        # Structure: Baseline + Winter Seasonality + Stochastic Outbreaks
        ili_base = 0.1 # 0.1% baseline
        
        # Seasonality: Peak in Jan/Feb, Low in Summer
        # Peak ~ Day 30 (Jan 30)
        ili_season = 0.5 * (1 + np.cos(2 * np.pi * (day_of_year - 30) / 365.0))
        ili_season = np.maximum(0, ili_season) # Clip negative
        
        # Stochastic Outbreaks (1-2 per year)
        ili_outbreak = np.zeros(days)
        num_outbreaks = self.rng.poisson(1.5) # Avg 1.5 outbreaks/year
        
        for _ in range(num_outbreaks):
            # Outbreak parameters
            start_day = self.rng.integers(0, days - 60)
            duration = self.rng.integers(30, 90) # 1-3 months
            peak_intensity = self.rng.uniform(1.0, 5.0) # Up to 5% rate
            
            # Log-Normal shape for outbreak curve
            # Simplified: Bell curve (Gaussian) for the duration
            t = np.arange(duration)
            center = duration / 2
            sigma = duration / 6
            curve = peak_intensity * np.exp(-(t - center)**2 / (2 * sigma**2))
            
            # Add to year (handle boundary)
            end_day = min(start_day + duration, days)
            eff_len = end_day - start_day
            ili_outbreak[start_day:end_day] += curve[:eff_len]
            
        ili_total = ili_base + (0.2 * ili_season) + ili_outbreak
        
        return pd.DataFrame({
            'Date': dates,
            '平均气温2m(℃)': np.round(temps, 2),
            '平均降水量(mm)': np.round(rain_amounts, 1),
            'ILI%': np.round(ili_total, 3) 
        })

if __name__ == "__main__":
    # Test generation for 2026
    gen = EnvironmentGenerator()
    df_2026 = gen.generate_year(2026)
    print("Generated 2026 Example:")
    print(df_2026.head())
    print("\nJan Stats:")
    print(df_2026[df_2026['Date'].dt.month == 1].mean(numeric_only=True))
    print("\nJuly Stats:")
    print(df_2026[df_2026['Date'].dt.month == 7].mean(numeric_only=True))
