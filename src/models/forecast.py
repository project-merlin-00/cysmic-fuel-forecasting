"""
ML Model Module
Implements ensemble forecasting models: Prophet, LSTM, XGBoost
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DemandForecaster:
    """Main class for demand forecasting using ML ensemble."""
    
    def __init__(self):
        self.models = {}
        self.is_fitted = False
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = "volume_liters") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for modeling.
        
        Args:
            df: Raw dataframe
            target_col: Target column name
            
        Returns:
            Features DataFrame and target Series
        """
        # Create time-based features
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Lag features
        for lag in [1, 7, 14, 30]:
            df[f'lag_{lag}'] = df.groupby('station_id')[target_col].shift(lag)
        
        # Rolling features
        df['rolling_mean_7'] = df.groupby('station_id')[target_col].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        df['rolling_mean_30'] = df.groupby('station_id')[target_col].transform(
            lambda x: x.rolling(30, min_periods=1).mean()
        )
        
        # Drop rows with NaN (from lag features)
        df = df.dropna()
        
        feature_cols = [
            'day_of_week', 'day_of_month', 'month', 'quarter', 
            'is_weekend', 'lag_1', 'lag_7', 'lag_14', 'lag_30',
            'rolling_mean_7', 'rolling_mean_30'
        ]
        
        return df[feature_cols], df[target_col]
    
    def train_prophet(self, df: pd.DataFrame, target_col: str = "volume_liters") -> Any:
        """
        Train Facebook Prophet model.
        
        Args:
            df: Time series data
            target_col: Target column
            
        Returns:
            Trained Prophet model
        """
        logger.info("Training Prophet model...")
        
        try:
            from prophet import Prophet
            
            # Prepare data for Prophet
            prophet_df = df[['date', target_col]].rename(
                columns={'date': 'ds', target_col: 'y'}
            )
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            model.fit(prophet_df)
            
            self.models['prophet'] = model
            logger.info("Prophet model trained successfully")
            return model
            
        except ImportError:
            logger.warning("Prophet not installed, skipping...")
            return None
    
    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """
        Train LSTM model for sequence prediction.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained LSTM model
        """
        logger.info("Training LSTM model...")
        
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            # Reshape for LSTM [samples, timesteps, features]
            X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            
            model = Sequential([
                LSTM(64, activation='relu', input_shape=(1, X_train.shape[2])),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            
            self.models['lstm'] = model
            logger.info("LSTM model trained successfully")
            return model
            
        except ImportError:
            logger.warning("TensorFlow not installed, skipping LSTM...")
            return None
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Trained XGBoost model
        """
        logger.info("Training XGBoost model...")
        
        try:
            import xgboost as xgb
            
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            self.models['xgboost'] = model
            logger.info("XGBoost model trained successfully")
            return model
            
        except ImportError:
            logger.warning("XGBoost not installed, skipping...")
            return None
    
    def predict_ensemble(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Generate predictions from all models.
        
        Args:
            X: Feature array
            
        Returns:
            Dictionary of predictions from each model
        """
        predictions = {}
        
        if 'xgboost' in self.models:
            predictions['xgboost'] = self.models['xgboost'].predict(X)
        
        if 'lstm' in self.models:
            X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
            predictions['lstm'] = self.models['lstm'].predict(X_lstm).flatten()
        
        # Prophet requires different prediction method
        if 'prophet' in self.models:
            logger.info("Prophet predictions require date input")
        
        return predictions
    
    def weighted_ensemble(self, predictions: Dict[str, np.ndarray], 
                         weights: Dict[str, float] = None) -> np.ndarray:
        """
        Create weighted ensemble prediction.
        
        Args:
            predictions: Dictionary of model predictions
            weights: Optional weights for each model
            
        Returns:
            Weighted average prediction
        """
        if weights is None:
            # Default equal weights
            weights = {k: 1.0/len(predictions) for k in predictions.keys()}
        
        ensemble = np.zeros_like(list(predictions.values())[0])
        
        for model_name, pred in predictions.items():
            ensemble += weights.get(model_name, 0) * pred
        
        return ensemble
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "MAPE": mape
        }


class FeatureEngineer:
    """Feature engineering for demand forecasting."""
    
    @staticmethod
    def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """Add calendar-based features."""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df[date_col].dt.month
        df['quarter'] = df[date_col].dt.quarter
        df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        
        return df
    
    @staticmethod
    def add_holiday_features(df: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
        """Add holiday/event features."""
        df = df.copy()
        
        # Create holiday lookup
        holiday_dates = events_df[events_df['type'] == 'holiday']['date'].tolist()
        election_dates = events_df[events_df['type'] == 'election']['date'].tolist()
        
        df['is_holiday'] = df['date'].isin(pd.to_datetime(holiday_dates)).astype(int)
        df['is_election'] = df['date'].isin(pd.to_datetime(election_dates)).astype(int)
        
        # Days until nearest holiday
        df['days_to_holiday'] = df['date'].apply(
            lambda x: min([abs((x - pd.to_datetime(d)).days) for d in holiday_dates] + [30])
        )
        
        return df
    
    @staticmethod
    def add_weather_features(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Add weather features."""
        df = df.copy()
        
        df = df.merge(
            weather_df[['date', 'temperature', 'precipitation', 'humidity']],
            on='date',
            how='left'
        )
        
        # Fill missing weather data
        df['temperature'] = df['temperature'].fillna(df['temperature'].mean())
        df['precipitation'] = df['precipitation'].fillna(0)
        df['humidity'] = df['humidity'].fillna(df['humidity'].mean())
        
        return df


if __name__ == "__main__":
    # Quick test
    forecaster = DemandForecaster()
    print("DemandForecaster initialized")
