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
        # Calibrated from 'data lib/external_factors.csv'
        # Mean: ~13°C, Min: -8°C, Max: 29°C -> Base 13, Amp 18
        t_base = 13.0
        t_amp = 18.0
        # Peak heat ~ July 15 (Day 196), Peak cold ~ Jan 15
        temp_season = t_base + t_amp * np.cos(2 * np.pi * (day_of_year - 196) / 365.0)
        temp_noise = self.rng.normal(0, 3.0, days) # Daily weather variation
        temps = temp_season + temp_noise
        
        # 2. Rainfall Generation (Stochastic Intermittent)
        # Calibrated: Winter ~0.3mm/day, Summer ~8.0mm/day (Peak July)
        # P(Rain) ~ 0.2 (Winter) to 0.5 (Summer)
        p_rain_season = 0.25 + 0.25 * np.cos(2 * np.pi * (day_of_year - 196) / 365.0) # Peak in Summer
        is_raining = self.rng.random(days) < p_rain_season
        
        # Rain amount (Gamma distribution)
        # Mean Intensity needs to match monthly averages:
        # Winter: 0.3mm total / 0.2 freq = 1.5mm per rain event
        # Summer: 8.0mm total / 0.5 freq = 16.0mm per rain event
        # Scale varies significantly.
        base_intensity = 1.5
        amp_intensity = 14.5 # 1.5 + 14.5 = 16.0
        # Peak Summer
        gamma_scale_ts = base_intensity + amp_intensity * 0.5 * (1 + np.cos(2 * np.pi * (day_of_year - 196) / 365.0))
        
        rain_amounts = np.zeros(days)
        # Vectorized gamma generation with varying scale is tricky without loop or approximations
        # Approximation: Generate with scale=1, then multiply by scale array
        gamma_shape = 1.5 # Skewed
        raw_gamma = self.rng.gamma(gamma_shape, 1.0, days)
        rain_amounts = raw_gamma * gamma_scale_ts * is_raining
        
        # 3. ILI% (Influenza-Like Illness) Generation
        # Calibrated from 'data lib/external_factors.csv'
        # Range: 0.025 (2.5%) to 0.091 (9.1%)
        # Mean: 0.04 (4%)
        # Seasonality: Jan Peak (5.8%), Aug Low (3.0%)
        
        ili_base = 0.03 # 3.0% Baseline
        ili_amp = 0.025 # +2.5% -> Peak 5.5% Seasonally
        
        # Seasonality: Peak in Jan (Day 15)
        # Cos peak at 0. Shift by 15.
        ili_season = ili_amp * (1 + np.cos(2 * np.pi * (day_of_year - 15) / 365.0)) / 2
        # Normalize to 0-1 range before scaling: (1+cos)/2 is 0..1
        # Result: 0 to 0.025.
        
        # Stochastic Outbreaks (0-4% add-on)
        ili_outbreak = np.zeros(days)
        num_outbreaks = self.rng.poisson(1.0) # Avg 1 outbreak/year
        
        for _ in range(num_outbreaks):
            start_day = self.rng.integers(0, days - 60)
            duration = self.rng.integers(30, 60) 
            peak_intensity = self.rng.uniform(0.01, 0.04) # 1-4% add-on
            
            t = np.arange(duration)
            center = duration / 2
            sigma = duration / 4
            curve = peak_intensity * np.exp(-(t - center)**2 / (2 * sigma**2))
            
            end_day = min(start_day + duration, days)
            eff_len = end_day - start_day
            ili_outbreak[start_day:end_day] += curve[:eff_len]
            
        ili_total = ili_base + ili_season + ili_outbreak
        # Clip to observed max/min roughly
        ili_total = np.clip(ili_total, 0.02, 0.10)
        
        # 4. Deterministic Factors
        holiday_flag = np.array([1 if d.dayofweek >= 5 else 0 for d in dates]) # Simple Weekend
        
        # Season Factor (1,2,3,4)
        # 1=Spring(3-5), 2=Summer(6-8), 3=Autumn(9-11), 4=Winter(12-2)
        def get_season(month):
            if 3 <= month <= 5: return 1
            if 6 <= month <= 8: return 2
            if 9 <= month <= 11: return 3
            return 4
        season_factor = np.array([get_season(d.month) for d in dates])
        
        return pd.DataFrame({
            '日期(UTC)': dates, # Match CSV column
            '平均气温2m(℃)': np.round(temps, 2),
            '平均降水量(mm)': np.round(rain_amounts, 2),
            'ILI%': np.round(ili_total, 4),
            '节假日标记（1=节假日/周末，0=工作日）': holiday_flag,
            '季节因子【1 = 春（3-5 月）、2 = 夏（6-8 月）、3 = 秋（9-11 月）、4 = 冬（12-2 月）】': season_factor
        })

if __name__ == "__main__":
    # Test generation for 2026
    gen = EnvironmentGenerator()
    df_2026 = gen.generate_year(2026)
    print("Generated 2026 Example:")
    print(df_2026.head())
    print("\nJan Stats:")
    print(df_2026[df_2026['日期(UTC)'].dt.month == 1].mean(numeric_only=True))
    print("\nJuly Stats:")
    print(df_2026[df_2026['日期(UTC)'].dt.month == 7].mean(numeric_only=True))
