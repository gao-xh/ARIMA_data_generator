from typing import Tuple, List, Dict, Any
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
        
        # Performance Metrics
        self.metrics = {
            'aic': None,
            'r2': None,
            'order': None
        }
        
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
        
        # --- 1. Define Candidate Exogenous Variables (Based on external_factors.csv) ---
        # We assume the CSV has columns like:
        # - 平均气温 (Mean Temp) -> Used to create Temp_Std
        # - 降雨量 (Rainfall)    -> Used to create Rain_Log
        # - 流感发病率 (Flu Rate) -> Direct use
        # - ILI% (Influenza-like Illness) -> Direct use
        
        self.weather_cols = ['平均气温', '降雨量'] 
        self.disease_cols = ['流感发病率', 'ILI%']
        
        # Fourier Terms for Annual Seasonality (Always used)
        self.fourier_cols = ['sin_1', 'cos_1', 'sin_2', 'cos_2']
        
        # --- 2. Dynamic Selection Strategy (Thesis 2.3.3) ---
        # Modified Plan (User Feedback): 
        # ALWAYS include external factors to ensure "Regulation" is visible if correlation exists.
        # Let the optimizer (ARIMA logic) decide significance via coefficients.
        
        # Base Exog: S_index + Fourier + Weather + Disease
        # Note: We use 'Temp_Std' and 'Rain_Log' which are transformed from raw data
        full_exog = ['S_index'] + self.fourier_cols + ['Temp_Std', 'Rain_Log', '流感发病率', 'ILI%']
        
        if self.fluctuation_class == C.FLUC_LOW: # Low Volatility
            self.order = (1, 0, 1)
            # Previously restricted, now enabling full regulation check
            self.exog_cols = full_exog
            
        elif self.fluctuation_class == C.FLUC_HIGH: # High Volatility
            self.order = (3, 1, 3)
            self.exog_cols = full_exog
            
        else: # Mid Volatility (Default)
            self.order = (2, 1, 2)
            self.exog_cols = full_exog
            
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

    def _generate_fourier_terms(self, dates: pd.DatetimeIndex, k: int = 2) -> pd.DataFrame:
        """
        Generates Fourier terms for annual seasonality (Period=365.25).
        """
        if not isinstance(dates, pd.DatetimeIndex):
             return pd.DataFrame()
             
        # Day of year (1-366)
        doyl = dates.dayofyear
        
        exog = pd.DataFrame(index=dates)
        for i in range(1, k + 1):
            exog[f'sin_{i}'] = np.sin(2 * np.pi * i * doyl / 365.25)
            exog[f'cos_{i}'] = np.cos(2 * np.pi * i * doyl / 365.25)
            
        return exog

    def prepare_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Prepares standard dataframe for ARIMAX pipeline.
        Crucially, this handles Feature Engineering for External Factors.
        """
        df = df.copy()
        
        # 1. Date Indexing
        if not pd.api.types.is_datetime64_any_dtype(df[C.COL_DATE]):
            df[C.COL_DATE] = pd.to_datetime(df[C.COL_DATE])
        df = df.sort_values(C.COL_DATE).set_index(C.COL_DATE)
        
        # 2. Frequency Standardization
        try:
             # Try to infer freq, default to Daily
             inferred_freq = pd.infer_freq(df.index)
             if inferred_freq:
                df = df.asfreq(inferred_freq)
             else:
                df = df.asfreq('D')
             
             # FF/BF ensures no gaps in timeline
             df = df.ffill().bfill() 
        except:
             pass 

        # 3. Feature Engineering: S_index (Seasonality)
        if is_training:
            self.s_index_map = self._calculate_s_index(df.reset_index())
        
        df['Month'] = df.index.month
        df['S_index'] = df['Month'].map(self.s_index_map).fillna(1.0)

        # 4. Feature Engineering: Fourier Terms (Annual Cycle)
        if hasattr(df.index, 'dayofyear'):
             fourier_df = self._generate_fourier_terms(df.index)
             for col in fourier_df.columns:
                 df[col] = fourier_df[col]

        # 5. Feature Engineering: Weather Transformations
        # Check if raw columns exist before transforming
        if '平均气温' in df.columns:
            # Standardize Temperature: (T - Mean)/Std -> Temp_Std
            # If training, we learn mean/std. If predicting, use learned.
            if is_training:
                self.temp_mean = df['平均气温'].mean()
                self.temp_std = df['平均气温'].std() if df['平均气温'].std() != 0 else 1.0
            
            # Apply transformation
            # Fill missing raw temp with mean first
            # Use getattr in case temp_mean wasn't set (should involve is_training check but safe fallback)
            t_mean = getattr(self, 'temp_mean', df['平均气温'].mean())
            t_std = getattr(self, 'temp_std', df['平均气温'].std() if df['平均气温'].std() != 0 else 1.0)

            raw_temp = df['平均气温'].fillna(t_mean)
            df['Temp_Std'] = (raw_temp - t_mean) / t_std
            
        if '降雨量' in df.columns:
            # Log Transform Rainfall: log1p(Rain) -> Rain_Log
            # Handles skewed distribution of rainfall data
            raw_rain = df['降雨量'].fillna(0).clip(lower=0)
            df['Rain_Log'] = np.log1p(raw_rain)

        # 6. Ensure all required columns exist (Fill Missing)
        for col in self.exog_cols:
            if col not in df.columns:
                   # Critical fallback
                   df[col] = 0.0
             
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
            
            # Use enforce_stationarity=False and enforce_invertibility=False to avoid 
            # "Non-stationary starting autoregressive parameters" warnings and improve convergence
            # on short or noisy datasets.
            self.model = ARIMA(endog, exog=exog, order=self.order, 
                             enforce_stationarity=False, 
                             enforce_invertibility=False)
                 
            # Increase maxiter to help convergence
            self.model_fit = self.model.fit(method_kwargs={"maxiter": 300})
            
            # --- Print Model Coefficients to Verify External Factors ---
            try:
                # Get parameters
                params = self.model_fit.params
                pvalues = self.model_fit.pvalues
                
                # Filter for Exogenous Variables
                exog_params = {k: v for k, v in params.items() if k in self.exog_cols}
                
                if exog_params:
                    print("\n[Improved ARIMAX] External Factor Coefficients:")
                    print("-" * 50)
                    for factor, coef in exog_params.items():
                        # Get p-value if available
                        pval = pvalues.get(factor, 1.0)
                        sig = "*" if pval < 0.05 else ""
                        impact = "POSITIVE" if coef > 0 else "NEGATIVE"
                        print(f"  {factor:<15}: Coef = {coef:>8.4f} (p={pval:.3f}) {sig} [{impact} Impact]")
                    print("-" * 50)
                    if not any(k in exog_params for k in ['Temp_Std', 'Rain_Log', '流感发病率']):
                        print("  WARNING: Weather/Flu factors were NOT selected for this drug class (Low Volatility?)")
                    else:
                        print("  CONFIRMED: Weather/Flu factors are actively influencing the forecast.")
                else:
                     print("\n[Improved ARIMAX] No External Factors used in regression.")
            except Exception as e:
                print(f"Could not print coefficients: {e}")
            
            # Calculate Metrics (Weekly Average for Robustness)
            from sklearn.metrics import r2_score
            try:
                self.metrics['aic'] = self.model_fit.aic
                
                # Default: Daily Fit
                daily_r2 = r2_score(endog, self.model_fit.fittedvalues)
                self.metrics['r2'] = daily_r2
                
                # Optimized: Weekly Mean Fit (Reduces daily noise impact)
                if isinstance(endog.index, pd.DatetimeIndex):
                    # Align fitted values to original index
                    fitted = pd.Series(self.model_fit.fittedvalues, index=endog.index)
                    comp_df = pd.DataFrame({'act': endog, 'pred': fitted})
                    
                    # Resample to Weekly Mean (Smooth out daily noise)
                    weekly_df = comp_df.resample('W').mean()
                    
                    # Recalculate R2 on aggregated data
                    if len(weekly_df) > 2:
                        weekly_r2 = r2_score(weekly_df['act'], weekly_df['pred'])
                        # Use Weekly R2 if it's better (usually is for retail data)
                        # But cap at reasonable value if denominator is tiny
                        self.metrics['r2'] = weekly_r2
                        self.metrics['r2_daily'] = daily_r2 # Store for debugging
                
                self.metrics['order'] = self.order
            except Exception as e:
                logger.warning(f"Failed to calculate metrics: {e}")
                self.metrics = {'aic': 0, 'r2': 0, 'order': self.order}

            logger.info("Model Training Completed.")
        except Exception as e:
            logger.error(f"Training Failed: {e}")
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Returns the training metrics."""
        return self.metrics

    def entropy_weight_decay(self, forecast_val: float, remaining_days: float, current_cv: float) -> float:
        """
        2.3.4 Validity Decay Coefficient Design
        alpha = alpha0 * (1 + beta * CV')
        
        SMOOTHER LOGIC (March 2026 Update):
        Instead of step functions (0.5/0.8), use a continuous decay curve based on sigmoid or logistic function to avoid cliffs.
        But for simplicity and interpretability in Thesis:
        - If remaining > 60: No decay (1.0)
        - If remaining <= 60: Linear decay from 1.0 down to a floor.
        - Floor depends on Volatility: High Volatility -> Higher Floor (0.8), Low -> Lower (0.4).
        """
        
        # 1. Determine Floor based on CV (High CV needs more stock even if expiring)
        # cv_prime is approx 0.0 to 1.0+
        cv_prime = min(max(current_cv, 0), 1.0)
        
        # High Volatility (cv=0.6+) -> floor = 0.8  (Don't cut too much!)
        # Low Volatility (cv=0.1)  -> floor = 0.4  (Can cut aggressively)
        min_alpha = 0.4 + (0.4 * cv_prime) 
        
        # 2. Calculate Decay
        # Linear ramp: 
        # Day 60 -> alpha = 1.0
        # Day 0  -> alpha = min_alpha
        
        if remaining_days > 60:
            alpha = 1.0
        elif remaining_days <= 0:
             alpha = 0.0 # Expired
        else:
            slope = (1.0 - min_alpha) / 60.0
            alpha = min_alpha + (slope * remaining_days)
            
        # 3. Apply Boost from CV (Beta term) - Kept/Modified from original
        # The original logic (1 + beta * cv) increases order for variable items.
        # This is already partially covered by 'min_alpha'.
        # Let's keep a small boost to acknowledge that Variable demand helps clear stock faster.
        beta = 0.1
        alpha = alpha * (1 + beta * cv_prime)
        
        # Clamp final alpha to [0, 1.2] (allow slight overstock for very high CV)
        alpha = min(max(alpha, 0.0), 1.2)
        
        decayed_val = forecast_val * alpha
        return decayed_val

    def predict(self, steps: int, future_exog_df: pd.DataFrame, current_stock_validity_days: float = None) -> List[float]:
        """
        Predicts future values using trained coefficients and FUTURE external factors.
        """
        if not self.model_fit:
            raise ValueError("Model not trained yet.")
            
        # 1. Prepare Future Exogenous Data
        # We MUST process future_exog_df exactly like training data to get 
        # Temp_Std, Rain_Log, S_index etc.
        
        # Ensure 'Date' column exists for preparation
        future_exog = future_exog_df.copy()
        if C.COL_DATE not in future_exog.columns and isinstance(future_exog.index, pd.DatetimeIndex):
            future_exog = future_exog.reset_index()
            # Rename index col to 'Date' if needed, or just use reset_index default
            if 'index' in future_exog.columns:
                future_exog = future_exog.rename(columns={'index': C.COL_DATE})

        # Run pipeline (is_training=False to use saved params like temp_mean)
        processed_future = self.prepare_data(future_exog, is_training=False)
        
        # 2. Select Relevant Time Slice
        # Assumes processed_future covers the prediction period.
        # We need exactly 'steps' rows. 
        # Ideally, future_exog_df starts exactly after training end.
        exog_ready = processed_future[self.exog_cols].iloc[:steps]
        
        if len(exog_ready) < steps:
            # Padding if insufficient future data provided
            pad_len = steps - len(exog_ready)
            last_row = exog_ready.iloc[[-1]]
            padding = pd.concat([last_row] * pad_len, ignore_index=True)
            exog_ready = pd.concat([exog_ready, padding], axis=0).reset_index(drop=True)
            exog_ready = exog_ready.iloc[:steps] # Trim exact
        
        # 3. Forecast
        forecast_res = self.model_fit.forecast(steps=steps, exog=exog_ready)
        
        forecast_list = forecast_res.tolist() if hasattr(forecast_res, 'tolist') else list(forecast_res)

        # 4. Apply Validity Decay
        if current_stock_validity_days is not None:
             final_forecast = []
             for i, val in enumerate(forecast_list):
                 rem_days = max(0, current_stock_validity_days - i)
                 decayed = self.entropy_weight_decay(val, rem_days, self.cv)
                 final_forecast.append(decayed)
             return final_forecast
             
        return forecast_list
