from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from src.core import constants as C

logger = logging.getLogger(__name__)

class TraditionalARIMA:
    """
    Implementation of Traditional ARIMA (Univariate) for baseline comparison.
    Unlike ImprovedARIMA (ARIMAX), this model does NOT use any external factors 
    (Seasonality Index, Temperature, Flu, Rainfall, etc.).
    
    It serves as a benchmark to demonstrate the value of the "Enhanced" features.
    """

    def __init__(self, drug_info: pd.Series):
        """
        Args:
            drug_info (pd.Series): Series containing basic info like 'FluctuationClass'.
        """
        self.drug_info = drug_info
        self.model_fit = None
        
        # Performance Metrics
        self.metrics = {
            'aic': None,
            'r2': None,
            'order': None
        }
        
        # Determine Fluctuation Class for (p,d,q) selection
        # We use the same order logic as ImprovedARIMA to make a fair 
        # "Univariate vs Multivariate" comparison.
        self.fluctuation_class = drug_info.get('波动区间分类', C.FLUC_MED)
        
        self._setup_params()

    def _setup_params(self):
        """Sets up model parameters (p,d,q). No Exog."""
        # Dynamic Parameter Selection (Same as ImprovedARIMA for comparability)
        if self.fluctuation_class == C.FLUC_LOW: # Low Volatility
            self.order = (1, 0, 1)
        elif self.fluctuation_class == C.FLUC_HIGH: # High Volatility
            self.order = (3, 1, 3)
        else: # Mid Volatility
            self.order = (2, 1, 2)
            
        logger.info(f"Traditional ARIMA Init | Class: {self.fluctuation_class} | Order: {self.order} | Exog: NONE")

    def prepare_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Prepares standard dataframe for ARIMA.
        Just ensures Date index and 'Sales' column.
        """
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[C.COL_DATE]):
            df[C.COL_DATE] = pd.to_datetime(df[C.COL_DATE])
            
        df = df.sort_values(C.COL_DATE)
        df = df.set_index(C.COL_DATE)
        
        # Try to infer frequency if possible
        try:
             df = df.asfreq(pd.infer_freq(df.index))
             df = df.ffill() # Forward fill missing
        except:
             pass 
             
        return df

    def train(self, train_df: pd.DataFrame):
        """Trains the Traditional ARIMA model (Endog only)."""
        data = self.prepare_data(train_df, is_training=True)
        endog = data[C.COL_SALES] # Target variable
        
        try:
            # Initialize ARIMA (No Exog)
            self.model = ARIMA(endog, order=self.order, 
                             enforce_stationarity=False, 
                             enforce_invertibility=False)
            
            self.model_fit = self.model.fit()
            
            # Calculate Metrics
            try:
                self.metrics['aic'] = self.model_fit.aic
                # Default: Daily Fit
                daily_r2 = r2_score(endog, self.model_fit.fittedvalues)
                self.metrics['r2'] = daily_r2
                
                # Optimized: Weekly Mean Fit (for fair comparison with Improved model)
                if isinstance(endog.index, pd.DatetimeIndex):
                    fitted = pd.Series(self.model_fit.fittedvalues, index=endog.index)
                    comp_df = pd.DataFrame({'act': endog, 'pred': fitted})
                    weekly_df = comp_df.resample('W').mean()
                    
                    if len(weekly_df) > 2:
                        weekly_r2 = r2_score(weekly_df['act'], weekly_df['pred'])
                        self.metrics['r2'] = weekly_r2
                
                self.metrics['order'] = self.order
            except Exception as e:
                logger.warning(f"Failed to calculate metrics: {e}")
                self.metrics = {'aic': 0, 'r2': 0, 'order': self.order}

            logger.info("Traditional ARIMA Training Completed.")
        except Exception as e:
            logger.error(f"Training Failed: {e}")
            raise

    def predict(self, steps: int, external_data: pd.DataFrame = None) -> pd.Series:
        """
        Generates forecast.
        Ignores external_data (since it's univariate).
        """
        if self.model_fit is None:
            raise ValueError("Model not trained yet.")
            
        # Forecast
        forecast_result = self.model_fit.get_forecast(steps=steps)
        forecast_series = forecast_result.predicted_mean
        
        # Ensure non-negative
        forecast_series = forecast_series.clip(lower=0)
        
        return forecast_series
        
    def get_fitted_values(self) -> pd.Series:
        """Returns in-sample fitted values."""
        if self.model_fit is None:
            return pd.Series()
        return self.model_fit.fittedvalues
