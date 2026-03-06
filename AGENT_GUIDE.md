# 📘 ARIMA Inventory Prediction - Agent Guide & Developer Handover

> **Note**: This guide serves as the "source of truth" for AI agents and developers working on the ARIMA Inventory Prediction project.

## 1. Project Overview
**Project Name**: ARIMA Inventory Prediction (Med-Inventory-Forecast)
**Goal**: Optimize pharmaceutical inventory management for community clinics by comparing **Traditional ARIMA** vs. **Improved ARIMA (ARIMAX)** models.

*   **Core Context**:
    *   **Problem**: Base-level clinics face fluctuating demand, leading to stockouts or waste.
    *   **Solution**: Improved ARIMA with external factors (flu, weather) & dynamic parameters.
*   **Current State**: 
    *   **Phase**: UI Implementation & Validation COMPLETE.
    *   **Next Step**: Paper writing / Final testing.

## 2. Directory Structure (Updated)

```
ARIMA/
├── data lib/               # Processed Input Data (CSV) and Template
│   ├── drug_info.csv
│   ├── external_factors.csv
│   └── sales_inventory_template.csv
├── ref/                    # Archived Raw Excel Files
├── data/                   # Generated Data Storage
│   └── processed/          # output of generate_dataset.py
├── src/
│   ├── core/               # Core Logic
│   │   ├── generator.py    # Synthetic Data Simulation Logic (CLI & GUI)
│   │   ├── constants.py    # Global Config
│   ├── models/             # Modeling
│   │   └── improved_arima.py # Main Algorithm Implementation
│   ├── ui/                 # PySide6 GUI
│   │   ├── common/         # Shared Widgets (Plots, Sliders)
│   │   ├── generation/     # Data Generator Tab
│   │   ├── validation/     # Model Validation Tab
│   │   └── main_window.py  # App Entry Point
│   └── evaluation/         # Metrics (MAPE, RMSE, R2)
├── run_app.bat             # [Execution] One-click start script
├── clean_setup.bat         # [Maintenance] Environment reset script
└── AGENT_GUIDE.md          # This file
```

## 3. Key Implementations

### A. Improved ARIMA Model (`src/models/improved_arima.py`)
Implementation fully compliant with thesis requirements:
1.  **Dynamic Orders**:
    *   Low Volatility (CV < 0.2) -> $(1,0,1)$
    *   Mid Volatility (0.2 ≤ CV ≤ 0.5) -> $(2,1,2)$
    *   High Volatility (CV > 0.5) -> $(3,1,3)$
2.  **External Factors (Exogenous Variables)**:
    *   Integrates **Seasonality Index ($S_{index}$)**.
    *   Integrates **Temperature** and **Flu Rates (ILI%)**.
3.  **Validity Decay**:
    *   Formula: $\alpha = \alpha_0 \times (1 + 0.2 \times CV')$
    *   Reduces forecast quantity when drug is near expiry (30-90 days).

### B. Data Generator (`src/core/generator.py`)
Generates synthetic data to prove **Hypothesis H1** (External factors affect sales).
*   **Logic**:
    *   Base Demand: Poisson Distribution.
    *   **Flu Shock**: If Flu Rate > Threshold (5%), demand spikes exponentially.
    *   **Cold Shock**: If Temp < Threshold (5°C), demand increases (1.5x).
    *   **Naive Replenishment**: Simulates a "dumb" reordering policy (Review every 14 days) to intentionally create stockouts/waste for the model to fix.

### C. UI Architecture (`src/ui/`)
*   **Framework**: `PySide6` with `Matplotlib`.
*   **Features**:
    *   **Modular**: Split into `GenerationWidget` and `ValidationWidget`.
    *   **Threaded**: Heavy tasks run in `QThread` to prevent UI freezing.
    *   **Styled**: Custom QSS for a professional look.

## 4. Environment & Deployment
*   **Management**: `.venv` (Python 3.10+).
*   **Startup**: Always use `run_app.bat`. It handles venv creation, pip install, and execution.
*   **Troubleshooting**: Use `clean_setup.bat` to wipe the environment if imports fail.

## 5. Changelog
*   **2026-03-06**:
    *   Refactored UI to support Tabbed Layout (Generation + Validation).
    *   Added `GenerationWidget` with slider controls for simulation parameters.
    *   Fixed `statsmodels` warning regarding Date Index.
    *   Converted `README.md` to Chinese for localization.
    *   **Data Cleanup**: Moved raw Excel files to `ref/`, standardized `data lib/` to CSV only.
    *   **Generator Update**: `generate_dataset.py` now outputs strict template-compliant CSV format (no type row).
    *   Pushed to GitHub `master` branch.
