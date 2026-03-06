from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
import logging

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from src.core import constants as C

logger = logging.getLogger(__name__)

class ImprovedARIMA:
    """
    Implementation of the Improved ARIMA (ARIMAX) model with Validity Decay.
    Based on the project documentation.
    
    Key Features:
    1. Dynamic Parameter Selection (Order) based on Fluctuation Class.
    2. External Factor Integration (Seasonality, Temperature, Flu).
    3. Validity Decay Adjustment for Forecasts.
    """

    def __init__(self, drug_info: pd.Series):
        """
        Args:
            drug_info (pd.Series): Series containing 'FluctuationClass', 'ValidityDays', etc.
        """
        self.drug_info = drug_info
        self.model_fit = None
        self.s_index_map = {}
        
        # Determine Fluctuation Class
        self.fluctuation_class = drug_info.get('波动区间分类', C.FLUC_MED)
        self.validity_days_initial = float(drug_info.get('效期（月）', 12)) * 30 
        
        # Calculate CV (Coefficient of Variation)
        # Ideally CV comes from historical data calculation, here we use a placeholder or provided value
        # If 'CV' column exists in drug_info, use it.
        self.cv = float(drug_info.get('CV', 0.5))
        
        self._setup_params()

    def _setup_params(self):
        """Sets up model parameters (p,d,q) and exogenous variables based on fluctuation class."""
        # 2.3.3 Dynamic Parameter Optimization (Fixed per doc requirements)
        if self.fluctuation_class == C.FLUC_LOW: # Low Volatility (CV < 0.2)
            self.order = (1, 0, 1)
            # Only Seasonality as Exog
            self.exog_cols = ['S_index'] 
            
        elif self.fluctuation_class == C.FLUC_HIGH: # High Volatility (CV > 0.5)
            self.order = (3, 1, 3)
            # All factors: Season, Temp, Rain, Flu, ILI
            self.exog_cols = ['S_index', C.EXT_TEMP, '平均降水量', '流感发病率', C.EXT_FLU]
            
        else: # Mid Volatility (Default: 0.2 <= CV <= 0.5)
            self.order = (2, 1, 2)
            # Season, Temp, Rain, Flu
            self.exog_cols = ['S_index', C.EXT_TEMP, '平均降水量', '流感发病率']
            
        logger.info(f"Model Init | Class: {self.fluctuation_class} | Order: {self.order} | Exog: {self.exog_cols}")

    def _calculate_s_index(self, df: pd.DataFrame) -> Dict[int, float]:
        """Calculates Seasonality Index (Monthly Sales / Avg Monthly Sales)."""
        temp_df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(temp_df[C.COL_DATE]):
             temp_df[C.COL_DATE] = pd.to_datetime(temp_df[C.COL_DATE])
             
        temp_df['Month'] = temp_df[C.COL_DATE].dt.month
        monthly_avg = temp_df.groupby('Month')[C.COL_SALES].mean()
        overall_avg = temp_df[C.COL_SALES].mean()
        
        if overall_avg == 0:
            s_index_map = {m: 1.0 for m in range(1, 13)}
        else:
            s_index_map = (monthly_avg / overall_avg).to_dict()
            
        # Fill missing months with 1.0 just in case
        for m in range(1, 13):
            if m not in s_index_map:
                s_index_map[m] = 1.0
                
        return s_index_map

    def prepare_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Prepares standard dataframe for ARIMAX.
        Expects columns: ['Date', 'Sales', '平均气温', ...]
        """
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[C.COL_DATE]):
            df[C.COL_DATE] = pd.to_datetime(df[C.COL_DATE])
            
        df = df.sort_values(C.COL_DATE)
        
        # Calculate S_index
        if is_training:
            self.s_index_map = self._calculate_s_index(df)
            
        # Apply Map
        df['Month'] = df[C.COL_DATE].dt.month
        # Use map with fillna(1.0)
        df['S_index'] = df['Month'].map(self.s_index_map).fillna(1.0)
        
        # Ensure Exog columns exist, fill missing with 0
        for col in self.exog_cols:
            if col not in df.columns:
                # S_index is already handled, skip warning for it
                if col != 'S_index':
                   # For missing columns (e.g. Rain/Flu), fill with 0 or mean? 0 is safer default if missing.
                   df[col] = 0.0
                   
        # Set Date as Index for ARIMA
        df = df.set_index(C.COL_DATE)
        
        # Try to infer frequency if possible (e.g. Daily 'D')
        try:
             df = df.asfreq(pd.infer_freq(df.index))
             df = df.fillna(method='ffill') # Forward fill missing dates if gap
        except:
             pass # If fails, just use index as is
             
        return df

    def train(self, train_df: pd.DataFrame):
        """Trains the ARIMAX model."""
        data = self.prepare_data(train_df, is_training=True)
        
        # Define endogenous and exogenous
        endog = data[C.COL_SALES]
        exog = data[self.exog_cols]
        
        try:
            # Initialize ARIMA with Exogenous variables
            # Order is defined in __init__
            # statsmodels ARIMA handles indices for time series
            if isinstance(data.index, pd.DatetimeIndex):
                 self.model = ARIMA(endog, exog=exog, order=self.order)
            else:
                 self.model = ARIMA(endog, exog=exog, order=self.order)
                 
            self.model_fit = self.model.fit()
            logger.info("Model Training Completed.")
        except Exception as e:
            logger.error(f"Training Failed: {e}")
            raise

    def entropy_weight_decay(self, forecast_val: float, remaining_days: float, current_cv: float) -> float:
        """
        2.3.4 Validity Decay Coefficient Design
        alpha = alpha0 * (1 + beta * CV')
        """
        # alpha0 logic
        if remaining_days > 90:
            alpha0 = 1.0
        elif remaining_days >= 30:
            alpha0 = 0.8
        else:
            alpha0 = 0.5
            
        beta = 0.2
        
        # CV' (Normalized CV) - Assumed 0-1 range usually, let's clamp
        cv_prime = min(max(current_cv, 0), 1.0)
        
        alpha = alpha0 * (1 + beta * cv_prime)
        
        decayed_val = forecast_val * alpha
        return decayed_val

    def predict(self, steps: int, future_exog_df: pd.DataFrame, current_stock_validity_days: float = None) -> List[float]:
        """
        Predicts future values.
        Args:
            steps (int): Number of steps to predict.
            future_exog_df (pd.DataFrame): Dataframe containing future exog features.
            current_stock_validity_days (float, optional): Remaining validity of current batch. 
                                                           Used for decay calculation.
        """
        if not self.model_fit:
            raise ValueError("Model not trained yet.")
            
        # Prepare future exog
        future_exog = future_exog_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(future_exog[C.COL_DATE]):
            future_exog[C.COL_DATE] = pd.to_datetime(future_exog[C.COL_DATE])
            
        # Set index to align with model expectations
        future_exog = future_exog.set_index(C.COL_DATE)
            
        # Apply Saved S_index
        future_exog['Month'] = future_exog.index.month
        # Use existing map or default to 1.0
        s_map = getattr(self, 's_index_map', {m: 1.0 for m in range(1, 13)})
        future_exog['S_index'] = future_exog['Month'].map(s_map).fillna(1.0)
        
        # Ensure Exog columns exist
        for col in self.exog_cols:
            if col not in future_exog.columns:
                if col != 'S_index':
                    future_exog[col] = 0.0
        
        # Select correct columns + rows
        # Assumes future_exog_df is already aligned with steps or larger
        exog_ready = future_exog[self.exog_cols].iloc[:steps]
        
        # Forecast
        forecast_res = self.model_fit.forecast(steps=steps, exog=exog_ready)
        
        # Convert series or array to list
        forecast_list = forecast_res.tolist() if hasattr(forecast_res, 'tolist') else list(forecast_res)

        # Apply Validity Decay if parameters provided
        if current_stock_validity_days is not None:
             final_forecast = []
             for i, val in enumerate(forecast_list):
                 # Validity decreases daily? Assume step=1day
                 rem_days = max(0, current_stock_validity_days - i)
                 decayed = self.entropy_weight_decay(val, rem_days, self.cv)
                 final_forecast.append(decayed)
             return final_forecast
             
        return forecast_list
