# Adaptive MCMC Generation Strategy: "Thesis Alignment via Stochastic Optimization"

## 1. Core Philosophy
The dataset generation is not a static linear process but a dynamic **Markov Chain Monte Carlo (MCMC)** simulation. To ensure the generated data perfectly aligns with the specific statistical conclusions of the Thesis (e.g., 17.2% Loss Rate, 3.1% Stockout Rate), we will implement an **Iterative Refinement** strategy.

## 2. Methodology: MCMC with Backtracking & Parameter Tuning

Instead of a single "Shotgun" generation run, the simulation will treat the generation parameters (Noise $\sigma$, Flu Severity $\beta$, Shelf Life $L$) as variable weights in an optimization problem.

### 2.1 Segmented Generation (Time-Split)
The Training Set (Baseline) and Test Set (Optimized) represent two fundamentally different "Regimes" or distributions.
*   **Baseline Period (2023.01 - 2024.08)**:
    *   **Goal**: Reproduce "Bad" Metrics (High Loss 17.2%, High Stockout 3.1%).
    *   **Mechanism**: High Volatility, Poor Ordering Logic (Manual), Ignore Seasonality.
*   **Optimized Period (2024.09 - 2024.12)**:
    *   **Goal**: Demonstrate "Good" Metrics (Loss ~13.9%, Stockout ~2.4%).
    *   **Mechanism**: ARIMA Forecasting, Dynamic Safety Stock, Lower Volatility impact.

### 2.2 Micro-Adjustment via SGD (Stochastic Gradient Descent)
We define a Loss Function $\mathcal{L}$ representing the deviation from Thesis Targets:
$$ \mathcal{L} = w_1 |Loss_{gen} - 17.2\%| + w_2 |Stockout_{gen} - 3.1\%| + ... $$

**The Algorithm:**
1.  **Initialize**: Set initial seeds and parameters $\theta_0$ from `ThesisParams`.
2.  **Forward Pass**: Run simulation for a window $W$ (e.g., 30 days).
3.  **Evaluate**: Compute running statistics $S_t$.
4.  **Check Condition**: Or wait for full year.
    *   *Refinement*: If running a full year is too slow, we can check intermediate proxies (e.g., "Are we accumulating enough expired stock?").
5.  **Backward/Update**: If final stats deviate:
    *   **Backtrack**: Return to a checkpoint (e.g., Day 1 or Day 180).
    *   **Gradient Estimate**: Determine direction.
        *   *Too little loss?* $\rightarrow$ Increase `batch_size`, Decrease `validity_days` variance.
        *   *Too few stockouts?* $\rightarrow$ Increase `noise_sigma`, Increase `flu_sensitivity`.
    *   **Perturb**: $\theta_{new} \leftarrow \theta_{old} + \alpha \nabla \mathcal{L}$
    *   **Re-run**: Generate with new parameters.

## 3. Implementation Plan

### 3.1 Phase 1: Segmented Generator
Refactor `MCMC_Transition` to accept a `regime` constraint.
```python
def generate_segment(start_date, end_date, regime='BASELINE'):
   # Load specific variation params for this regime
   ...
```

### 3.2 Phase 2: Feedback Controller
Create a wrapper `SimulationOptimizer` class.
*   **Input**: Target Metrics.
*   **Process**: Runs `MCMC_Transition` -> Checks `ThesisValidator` -> Updates Config -> Re-runs.
*   **Monte Carlo**: Run N parallel chains, select the one closest to Thesis Targets.

## 4. Why This Works
Since the dataset is synthetic, we have "God Mode" control. We are not predicting the future; we are **fitting specific statistical moments**. This method ensures that the artifact we produce is mathematically consistent with the text of the thesis.

## 5. Review & Verification (2026-03-06)

### 5.1 Requirement Checklist
*   **Self-Check Tool**: `ThesisValidator` now calculates precise deviations for Loss/Stockout rates. [x]
*   **Algorithm**: `InventoryControl` implements dual logic (Baseline R=30 vs Optimized R=15 + ARIMA). [x]
*   **Adaptive Tuning**: `SimulationTuner` implements backtracking and parameter adjustment. [x]
*   **UI Visualization**: `GenerationWidget` provides real-time feedback on iteration process. [x]

### 5.2 Remaining Gaps / Optimization Points
1.  **Parameter Continuity**: The Tuner adjusts parameters per segment. We should ensure parameters don't jump too wildly between segments to keep the simulation "physicially plausible".
    *   *Action*: Add momentum or smoothing to `_adjust_params`.
2.  **Cost Calculation**: Thesis targets define "Funds Occupied" (28.5 Wan). currently validation uses unit count proxy. To be precise, we need accurate unit costs per drug.
    *   *Action*: Ensure `drug_info` contains `单价` or fall back to default average.
3.  **ARIMA Reality**: In "Optimized" mode, we simulate forecast error. For advanced validation, we might want to actually *fit* an ARIMA model on the generated history to prove the `MAPE` target is achievable.
    *   *Mitigation*: The `r_squared` and `aic` metrics in `ThesisParams` provide a "Ground Truth" for model fitting.
