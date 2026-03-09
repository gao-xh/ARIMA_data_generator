# Environmental Factor Generation Logic (Thesis H1/H3)

## Purpose
Simulate future environmental conditions (Temperature, Rainfall, Epidemic Activity) when historical data is unavailable (e.g., forecasting 2026-2027 demand).

## Algorithms

### 1. Temperature (Seasonality + Noise)
Daily temperature follows a deterministic sinusoidal curve based on latitude, plus stochastic daily variation.
```python
T(d) = T_base + T_amp * cos(2*pi*(d - d_peak)/365) + epsilon
```
- **Base**: 15°C
- **Amplitude**: 15°C (Range 0°C - 30°C)
- **Peak**: July 15 (Day 196)
- **Daily Noise**: N(0, 3°C)

### 2. Rainfall (Discrete-Continuous Hybrid)
Rainfall is modeled as a two-step process:
1. **Occurrence (Bernoulli)**: Probability of rain varies by season (Summer > Winter).
   - `P(Rain) ~ 0.35 + 0.15 * cos(Season)`
2. **Intensity (Gamma)**: Amount of rain when it occurs follows a Gamma distribution.
   - `Rain ~ Gamma(k=2.0, theta=Seasonal_Intensity)`

### 3. Epidemic Activity (ILI%) (Base + Outbreaks)
Influenza-Like Illness (ILI) rate is composed of a low baseline, winter seasonality, and random outbreak events.
```python
ILI% = Base(0.1%) + Seasonality(Winter_Peak) + Stochastic_Outbreaks
```
- **Outbreaks**: Modeled as Poisson process (~1.5/year).
- **Shape**: Gaussian burst with random duration (30-90 days) and intensity (1-5%).

## Usage
This generator provides synthetic `external_factors.csv` rows for future date ranges, allowing the core simulation to run beyond the current dataset limit.
