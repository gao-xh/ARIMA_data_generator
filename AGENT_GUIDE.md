# 📘 ARIMA Inventory Prediction - Agent Guide & Developer Handover

> **Note**: This guide serves as the "source of truth" for AI agents and developers working on the ARIMA Inventory Prediction project.

### 1. Project Overview
**Project Name**: ARIMA Inventory Prediction (Med-Inventory-Forecast)
**Goal**: Optimize pharmaceutical inventory management for community clinics by comparing **Traditional ARIMA** vs. **Improved ARIMA (ARIMAX)** models. The improved model incorporates external factors (weather, flu trends, holidays) and dynamic parameter adjustment to reduce stockouts and waste.

*   **Core Context**:
    *   **Problem**: Base-level clinics face fluctuating demand for common drugs, leading to inventory imbalance (stockouts/overstock).
    *   **Solution**: A "Lightweight, High-Precision" forecasting model.
    *   **Hypothesis**: Improved ARIMA (with external regressors & validity decay) > Traditional ARIMA.
*   **Workspace Structure**:
    *   `src/`: Core logic and helper functions (`data_simulator.py`, `models.py`, `config.py`).
    *   `data lib/`: Raw reference data (External factors, Drug info).
    *   `data/processed/`: Generated datasets (`synthetic_sales.csv`).
    *   `ref/`: Reference documents and target requirements.
    *   `generate_dataset.py`: Entry point for data simulation.
    *   `AGENT_GUIDE.md`: Project guide.

### 2. Development Standards
*   **Language**: Python 3.x
*   **Libraries**: `pandas`, `numpy`, `statsmodels` (for ARIMA/SARIMAX), `pmdarima` (optional auto-arima), `matplotlib`/`seaborn`.
*   **Code Comments**: **MUST BE IN ENGLISH**. No Chinese characters in the source code files (to avoid encoding issues and maintain standard).
*   **Docstrings**: Use Google or NumPy style docstrings.
*   **Type Hinting**: Strongly encouraged for all public functions in `src/`.
*   **Paths**: Use `os.path` or `pathlib` for cross-platform compatibility. **NEVER** hardcode absolute paths.

### 3. Architecture & Key Logic (Planned)

#### A. Data Pipeline (`src/data_loader.py` & `src/preprocessing.py`)
1.  **Ingestion**:
    *   Load `销量库存业务表` (Sales & Inventory Transaction Data).
    *   Load `外部影响因子表` (External Factors: Weather, Flu, Holidays).
    *   Load `药品基础信息表` (Drug Master Data: Categories, Validity).
2.  **Preprocessing & Merging**:
    *   **Aggregation**: Aggregate sales data by `Date` + `Drug ID` + `Clinic ID` (Daily Granularity).
    *   **Imputation**: Handle missing dates (fill with 0 for sales, linear interpolation for stock if needed).
    *   **Feature Engineering**:
        *   Merge External Factors (Weather, Flu) to the main time series.
        *   Create "Lag" features if necessary for ARIMAX.
        *   Encode Categorical variables (e.g., Holidays, Season).
3.  **Dataset Generation**:
    *   Output: A clean, merged "Wide Table" or "Time Series formatted" CSV for modeling.

#### B. Modeling Strategy (`src/models.py`)

1.  **Baseline (Traditional ARIMA)**:
    *   Univariate time series forecasting using only historical sales.

2.  **Improved Model (ARIMAX + Decay)**:
    *   **External Regressors (Based on Drug Volatility)**:
        *   **Low Volatility**: Seasonality Index ($S_{index}$).
        *   **Mid Volatility**: $S_{index}$ + Temp + Rain + Flu Rate.
        *   **High Volatility**: All factors.
    *   **Dynamic Parameters (AIC Optimized)**:
        *   Low: $(1,0,1)$
        *   Mid: $(2,1,2)$
        *   High: $(3,1,3)$
    *   **Validity Decay Coefficient ($\alpha$)**:
        *   Formula: $\hat{Y}_{final} = \hat{Y}_{ARIMAX} \times \alpha$
        *   $\alpha = \alpha_0 \times (1 + 0.2 \times CV')$
        *   $\alpha_0$: 1.0 (>90 days), 0.8 (30-90 days), 0.5 (<30 days).

#### C. Evaluation (`src/evaluation.py`)
*   **Metrics**: MAPE (Mean Absolute Percentage Error), RMSE, Stockout Rate, Turnover Days.
*   **Validation**: Train/Test split (e.g., 2024 for training, 2025 for testing).

### 4. Implementation Steps (Current Phase)

#### Phase 1: Dataset Generation (Current Task)
*   **Goal**: Create a robust `DatasetGenerator` class.
*   **Input**: The 3 raw Excel files.
*   **Logic**:
    *   Read files using correct engines (`xlrd` for .xls, `openpyxl` for .xlsx).
    *   Clean column names (strip whitespace).
    *   Convert dates to standard `datetime` objects.
    *   Perform Left Join: `Sales` + `Drug Info` (on Drug Code).
    *   Perform Left Join: `Result` + `External Factors` (on Date).
    *   Handle missing values (e.g., fill NaNs in sales with 0, propagate external factors).
*   **Output**: `data/processed/merged_dataset.csv`.

### 5. Critical Watchlist (Pitfalls to Avoid)

1.  **Date Alignment**: Ensure "Date" columns in all files are parsed correctly (Check for different formats like `YYYYMMDD` vs `YYYY-MM-DD`).
2.  **Granularity**: The code must handle the "Clinic" dimension. Predictions might be needed *per clinic* or *aggregated*. Default to supporting *per clinic* granularity.
3.  **Encoding Issues**: When reading Excel files with Chinese characters, ensure pandas reads them correctly (usually auto-detected, but be watchful).
4.  **Separation of Concerns**:
    *   Don't mix data cleaning visualization (plots) inside the data loader functions. Keep `src/` files pure logic.
