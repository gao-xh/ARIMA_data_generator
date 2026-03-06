# ARIMA Data Generator

## Project Overview
This project generates synthetic pharmaceutical sales and inventory data to simulate the operations of community clinics. The data is designed to support the research on **Improved ARIMA (ARIMAX)** models for inventory forecasting.

## Core Logic & Hypotheses

### 1. Sales Simulation (Hypothesis H1)
Sales data is generated based on **External Factors** to support the hypothesis that "Common drug demand is significantly affected by seasons, weather, and disease rates."

*   **High Volatility Drugs**: Sales are strongly correlated with:
    *   **Flu Rate (ILI%)**: Primary driver for cold/flu medications.
    *   **Temperature Drop**: Secondary driver.
    *   **Season**: Baseline seasonality.
*   **Low Volatility Drugs**: Sales are stable with minor random noise (White Noise).

### 2. Inventory Simulation (Problem Statement)
Inventory data simulates a **Traditional/Empirical Replenishment Strategy** to demonstrate current inefficiencies (High Stockouts, High Waste).

*   **Replenishment Logic**: "Naive Periodic Review"
    *   **Frequency**: Every 14 days (or monthly).
    *   **Order Quantity**: Based on *historical average* (e.g., average of last 3 months).
*   **Failure Modes**:
    *   **Stockouts**: During Flu outbreaks (Sudden demand spike > Historical average).
    *   **Waste/Overstock**: Post-outbreak (Demand drops, but replenishment is based on high historical average).

## Data Structure
The generator outputs a dataset matching `销量库存业务表.xlsx` with columns:
*   Date
*   Clinic ID
*   Drug Code
*   Sales (Simulated)
*   Inventory (Calculated)
*   Replenishment (Simulated)
*   Stockout Flag (Calculated)
*   Loss/Expiry (Calculated)

## Usage
1.  Place reference data in `data lib/`.
2.  Run the generator script.
3.  Output will be saved to `data/processed/`.
