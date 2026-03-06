import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedARIMA:
    """
    Implementation of the Improved ARIMA (ARIMAX) model with Validity Decay.
    Based on the project documentation.
    """

    def __init__(self, drug_info):
        """
        Args:
            drug_info (dict): Dictionary identifying 'FluctuationClass', 'ValidityDays', etc.
        """
        self.drug_info = drug_info
        self.model_fit = None
        self.fluctuation_class = drug_info.get('波动区间分类', '中波动')
        self.validity_days = float(drug_info.get('效期（月）', 12)) * 30 # Convert months to days approx
        
        # Calculate CV (Coefficient of Variation) if available or use a default
        # Ideally CV comes from historical data, here we might estimate or use a placeholder
        self.cv = 0.5 # Default placeholder if not calculated from data
        
        self._setup_params()

    def _setup_params(self):
        """Sets up model parameters based on fluctuation class."""
        # 2.3.3 Dynamic Parameter Optimization (Fixed per doc requirements)
        if '低' in self.fluctuation_class: # Low Volatility
            self.order = (1, 0, 1)
            self.exog_cols = ['S_index'] # Only Seasonality
        elif '高' in self.fluctuation_class: # High Volatility
            self.order = (3, 1, 3)
            # All factors: Season, Temp, Rain, Flu, ILI
            self.exog_cols = ['S_index', '平均气温', '平均降水量', '流感发病率', '流感ILI%']
        else: # Mid Volatility (Default)
            self.order = (2, 1, 2)
            # Season, Temp, Rain, Flu
            self.exog_cols = ['S_index', '平均气温', '平均降水量', '流感发病率']
            
        logger.info(f"Model Init | Class: {self.fluctuation_class} | Order: {self.order} | Exog: {self.exog_cols}")

    def _calculate_s_index(self, df):
        """Calculates Seasonality Index (Monthly Sales / Avg Monthly Sales)."""
        temp_df = df.copy()
        temp_df['Month'] = temp_df['Date'].dt.month
        monthly_avg = temp_df.groupby('Month')['Sales'].mean()
        overall_avg = temp_df['Sales'].mean()
        
        if overall_avg == 0:
            s_index_map = {m: 1.0 for m in range(1, 13)}
        else:
            s_index_map = (monthly_avg / overall_avg).to_dict()
            
        # Fill missing months with 1.0 just in case
        for m in range(1, 13):
            if m not in s_index_map:
                s_index_map[m] = 1.0
                
        return s_index_map

    def prepare_data(self, df, is_training=True):
        """
        Prepares standard dataframe for ARIMAX.
        Expects columns: ['Date', 'Sales', '平均气温', ...]
        """
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
            
        df = df.sort_values('Date')
        
        # Calculate S_index
        if is_training:
            self.s_index_map = self._calculate_s_index(df)
            
        # Apply Map
        df['Month'] = df['Date'].dt.month
        # Use map with fillna(1.0)
        df['S_index'] = df['Month'].map(self.s_index_map).fillna(1.0)
        
        # Ensure Exog columns exist, fill missing with 0
        for col in self.exog_cols:
            if col not in df.columns:
                # S_index is already handled, skip warning for it
                if col != 'S_index':
                   logger.warning(f"Missing exog column {col}, filling with 0")
                   df[col] = 0.0
                
        return df

    def train(self, train_df):
        """Trains the ARIMAX model."""
        data = self.prepare_data(train_df, is_training=True)
        
        # Define endogenous and exogenous
        endog = data['Sales']
        exog = data[self.exog_cols]
        
        try:
            # Initialize ARIMA with Exogenous variables
            # Order is defined in __init__
            self.model = ARIMA(endog, exog=exog, order=self.order)
            self.model_fit = self.model.fit()
            logger.info("Model Training Completed.")
        except Exception as e:
            logger.error(f"Training Failed: {e}")
            raise

    def entropy_weight_decay(self, forecast_val, remaining_days, current_cv):
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
        
        # Final decay shouldn't strictly increase forecast typically for validity risk...
        # Wait, formula says (1 + ...), so alpha > alpha0 if High Volatility?
        # Maybe High Volatility implies *riskier* to hold, so we want to *reduce* stock? 
        # Actually doc says: "realize 'demand forecast + validity control' double target".
        # If correlation is positive, it means for high volatility we keep MORE? 
        # Or maybe alpha is a *reduction* factor?
        # Base alpha0 is < 1 for short validity. (0.5).
        # (1 + 0.2*CV) makes it slightly larger (e.g. 0.5 * 1.2 = 0.6).
        # This implies: High volatility items with short validity are slightly *safer* to stock than low volatility short validity?
        # Or maybe logic is: High volatility = chance of selling is higher, so don't cut down prediction as much?
        # Let's stick to the formula explicitly.
        
        decayed_val = forecast_val * alpha
        return decayed_val

    def predict(self, steps, future_exog_df, current_stock_validity_days=None):
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
        if not pd.api.types.is_datetime64_any_dtype(future_exog['Date']):
            future_exog['Date'] = pd.to_datetime(future_exog['Date'])
            
        # Apply Saved S_index
        future_exog['Month'] = future_exog['Date'].dt.month
        # Use existing map or default to 1.0
        s_map = getattr(self, 's_index_map', {m: 1.0 for m in range(1, 13)})
        future_exog['S_index'] = future_exog['Month'].map(s_map).fillna(1.0)
        
        # Ensure Exog columns exist
        for col in self.exog_cols:
            if col not in future_exog.columns:
                if col != 'S_index':
                    logger.warning(f"Predict: Missing exog column {col}, filling with 0")
                    future_exog[col] = 0.0
        
        # Select correct columns + rows
        # Assumes future_exog_df is already aligned with steps or larger
        exog_ready = future_exog[self.exog_cols].iloc[:steps]
        
        # Forecast
        forecast_res = self.model_fit.forecast(steps=steps, exog=exog_ready)
        
        # Apply Validity Decay if parameters provided
        if current_stock_validity_days is not None:
             final_forecast = []
             for i, val in enumerate(forecast_res):
                 # Validity decreases daily? Assume step=1day
                 rem_days = max(0, current_stock_validity_days - i)
                 decayed = self.entropy_weight_decay(val, rem_days, self.cv)
                 final_forecast.append(decayed)
                 
             return pd.Series(final_forecast, index=forecast_res.index)
        
        return forecast_res

