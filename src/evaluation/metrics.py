from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Handles zeros.
    Returns: Percentage (e.g., 5.0 for 5%)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mask = y_true != 0
    if not np.any(mask):
        return 0.0 # Or error code
        
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_inventory_metrics(actual_demand: np.ndarray, replenished_stock: np.ndarray) -> Dict[str, float]:
    """
    Calculates operational metrics:
    - Stockout Rate
    - Turnover Days (Simulated)
    - Waste Rate (If expiry data available)
    """
    # Simple place holder
    stockout = np.sum(replenished_stock < actual_demand)
    rate = stockout / len(actual_demand) * 100
    
    return {
        'Stockout Rate': rate,
        'Mean Inventory': np.mean(replenished_stock)
    }
