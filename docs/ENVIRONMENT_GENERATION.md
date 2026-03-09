# Environmental Factor Generation Logic (Thesis H1/H3)

## Purpose
Simulate future environmental conditions (Temperature, Rainfall, Epidemic Activity) when historical data is unavailable (e.g., forecasting 2026-2027 demand).

## Algorithms

### 1. Temperature (Seasonality + Noise)
Daily temperature follows a deterministic sinusoidal curve based on latitude, plus stochastic daily variation.
```python
T(d) = T_base + T_amp * cos(2*pi*(d - d_peak)/365) + epsilon
```
- **Base**: 13°C (Calibrated from mean 12.7°C)
- **Amplitude**: 18°C (Range -5°C to 31°C, matching historical -8°C to 28°C)
- **Peak**: July 15 (Day 196)
- **Daily Noise**: N(0, 3°C)

### 2. Rainfall (Discrete-Continuous Hybrid)
Rainfall is modeled as a two-step process:
1. **Occurrence (Bernoulli)**: Probability of rain varies by season (Summer > Winter).
   - `P(Rain) ~ 0.25 (Winter) to 0.50 (Summer)`
2. **Intensity (Gamma)**: Amount of rain when it occurs follows a Gamma distribution with seasonally varying scale.
   - Mean intensity per rainy day scales from ~1.5mm (Winter) to ~16mm (Summer) to match monthly averages.

### 3. Epidemic Activity (ILI%) (Base + Outbreaks)
Influenza-Like Illness (ILI) rate is composed of a low baseline, winter seasonality, and random outbreak events.
```python
ILI% = Base(3%) + Seasonality(Peak +2.5%) + Stochastic_Outbreaks
```
- **Observed Stats**: Mean 4.0%, Max 9.1%, Min 2.5%.
- **Model**: Base 0.03 + Seasonality (Peak Jan 0.055) + Outbreaks (Poisson process, adding 1-4%).
- **Resulting Range**: Typically 0.03 - 0.095, matching historical constraints.

## Usage
This generator provides synthetic `external_factors.csv` rows for future date ranges, allowing the core simulation to run beyond the current dataset limit.
