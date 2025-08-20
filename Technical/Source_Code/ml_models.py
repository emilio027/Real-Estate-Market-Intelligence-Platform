#!/usr/bin/env python3
"""
Real Estate Market Intelligence Platform - ML Models Module
Advanced Machine Learning Models for Property Valuation and Market Analysis

Author: Emilio Cardenas
Institution: MIT PhD AI Automation | Harvard MBA
Version: 2.0.0 Enterprise
License: Proprietary
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')

# Core ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import joblib

# Advanced ML Libraries
try:
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False
    logging.warning("Advanced ML libraries (XGBoost, LightGBM, CatBoost) not available")

# Geospatial Libraries
try:
    import geopandas as gpd
    from shapely.geometry import Point
    import folium
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    logging.warning("Geospatial libraries not available")

# Deep Learning (Optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    logging.warning("TensorFlow not available - deep learning models disabled")

# Model Interpretability
try:
    import shap
    import lime
    import lime.tabular
    INTERPRETABILITY_AVAILABLE = True
except ImportError:
    INTERPRETABILITY_AVAILABLE = False
    logging.warning("Model interpretability libraries not available")


class RealEstateMLManager:
    """
    Comprehensive machine learning model manager for real estate market intelligence.
    Provides property valuation, market forecasting, and investment optimization models.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the Real Estate ML Manager."""
        self.random_state = random_state
        self.logger = self._setup_logging()
        
        # Model storage
        self.valuation_models = {}
        self.forecasting_models = {}
        self.classification_models = {}
        self.clustering_models = {}
        
        # Performance tracking
        self.model_performance = {}
        self.feature_importance = {}
        
        # Data processors
        self.scalers = {}
        self.encoders = {}
        
        # Best model tracking
        self.best_valuation_model = None
        self.best_forecasting_model = None
        
        self.logger.info("Real Estate ML Manager initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('RealEstateMLManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def prepare_property_data(self, property_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and engineer features for property valuation models."""
        self.logger.info("Preparing property data for modeling...")
        
        df = property_df.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Engineer temporal features
        if 'sale_date' in df.columns:
            df['sale_date'] = pd.to_datetime(df['sale_date'])
            df['sale_year'] = df['sale_date'].dt.year
            df['sale_month'] = df['sale_date'].dt.month
            df['sale_quarter'] = df['sale_date'].dt.quarter
            df['days_since_sale'] = (datetime.now() - df['sale_date']).dt.days
            
        # Property age and condition features
        if 'year_built' in df.columns:
            current_year = datetime.now().year
            df['property_age'] = current_year - df['year_built']
            df['age_category'] = pd.cut(df['property_age'], 
                                      bins=[0, 10, 20, 50, 100], 
                                      labels=['New', 'Modern', 'Mature', 'Old'])
        
        # Size and layout features
        if 'square_footage' in df.columns and 'lot_size' in df.columns:
            df['size_efficiency'] = df['square_footage'] / df['lot_size']
            df['size_category'] = pd.cut(df['square_footage'],
                                       bins=[0, 1000, 2000, 3000, 5000, np.inf],
                                       labels=['Small', 'Medium', 'Large', 'XLarge', 'Mansion'])
        
        # Financial features
        if 'annual_rent' in df.columns and 'property_value' in df.columns:
            df['gross_yield'] = (df['annual_rent'] / df['property_value']) * 100
            df['price_per_sqft'] = df['property_value'] / df['square_footage']
        
        # Location features (if coordinates available)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            df = self._add_location_features(df)
        
        # Market features
        df = self._add_market_features(df)
        
        self.logger.info(f"Feature engineering completed. Dataset shape: {df.shape}")
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Numerical columns - fill with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical columns - fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        return df
    
    def _add_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add location-based features."""
        if not GEOSPATIAL_AVAILABLE:
            self.logger.warning("Geospatial libraries not available - skipping location features")
            return df
        
        # Distance to city center (simplified - using Manhattan as example)
        city_center_lat, city_center_lon = 40.7128, -74.0060  # NYC coordinates
        
        df['distance_to_center'] = np.sqrt(
            (df['latitude'] - city_center_lat)**2 + 
            (df['longitude'] - city_center_lon)**2
        ) * 111  # Approximate km conversion
        
        # Create location clusters
        if len(df) > 100:
            coords = df[['latitude', 'longitude']].values
            kmeans = KMeans(n_clusters=min(20, len(df)//50), random_state=self.random_state)
            df['location_cluster'] = kmeans.fit_predict(coords)
            self.clustering_models['location'] = kmeans
        
        return df
    
    def _add_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market-level features."""
        # Property type distributions
        if 'property_type' in df.columns:
            type_counts = df['property_type'].value_counts()
            df['type_rarity'] = df['property_type'].map(lambda x: 1.0 / type_counts[x])
        
        # Neighborhood price levels
        if 'neighborhood' in df.columns and 'property_value' in df.columns:
            neighborhood_stats = df.groupby('neighborhood')['property_value'].agg(['mean', 'std', 'count'])
            df['neighborhood_avg_price'] = df['neighborhood'].map(neighborhood_stats['mean'])
            df['neighborhood_price_volatility'] = df['neighborhood'].map(neighborhood_stats['std'])
            df['neighborhood_liquidity'] = df['neighborhood'].map(neighborhood_stats['count'])
        
        return df
    
    def create_valuation_models(self) -> Dict[str, Any]:
        """Create comprehensive property valuation models."""
        base_config = {'random_state': self.random_state, 'n_jobs': -1}
        
        models = {
            # Traditional Regression Models
            'linear_regression': {
                'model': LinearRegression(),
                'params': {'fit_intercept': [True, False]}
            },
            'ridge_regression': {
                'model': Ridge(**base_config),
                'params': {'alpha': [0.1, 1.0, 10.0, 100.0]}
            },
            'lasso_regression': {
                'model': Lasso(**base_config),
                'params': {'alpha': [0.01, 0.1, 1.0, 10.0]}
            },
            'elastic_net': {
                'model': ElasticNet(**base_config),
                'params': {
                    'alpha': [0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.5, 0.7, 0.9]
                }
            },
            
            # Tree-Based Models
            'random_forest': {
                'model': RandomForestRegressor(**base_config),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'max_features': ['sqrt', 'log2']
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=self.random_state),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            
            # Support Vector Regression
            'svr': {
                'model': SVR(),
                'params': {
                    'kernel': ['linear', 'rbf'],
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto']
                }
            },
            
            # Neural Network
            'mlp': {
                'model': MLPRegressor(random_state=self.random_state, max_iter=2000),
                'params': {
                    'hidden_layer_sizes': [(100,), (100, 50), (200, 100, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01]
                }
            }
        }
        
        # Add advanced models if available
        if ADVANCED_ML_AVAILABLE:
            models.update({
                'xgboost': {
                    'model': xgb.XGBRegressor(random_state=self.random_state),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0]
                    }
                },
                'lightgbm': {
                    'model': lgb.LGBMRegressor(random_state=self.random_state),
                    'params': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'num_leaves': [31, 50, 100]
                    }
                },
                'catboost': {
                    'model': cb.CatBoostRegressor(random_state=self.random_state, verbose=False),
                    'params': {
                        'iterations': [100, 200, 300],
                        'depth': [3, 6, 9],
                        'learning_rate': [0.01, 0.1, 0.2]
                    }
                }
            })
        
        return models
    
    def train_valuation_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame = None, y_val: pd.Series = None,
                             cv_folds: int = 5) -> Dict[str, Dict]:
        """Train property valuation models with hyperparameter optimization."""
        self.logger.info("Training property valuation models...")
        
        # Encode categorical variables
        X_train_processed, X_val_processed = self._encode_features(X_train, X_val)
        
        # Get model configurations
        model_configs = self.create_valuation_models()
        results = {}
        
        for model_name, config in model_configs.items():
            self.logger.info(f"Training {model_name}...")
            
            try:
                # Create pipeline with scaling
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', config['model'])
                ])
                
                # Update parameter names for pipeline
                param_grid = {}
                for param, values in config['params'].items():
                    param_grid[f'model__{param}'] = values
                
                # Grid search with cross-validation
                grid_search = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=cv_folds,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=1,  # Avoid nested parallelism
                    verbose=0
                )
                
                grid_search.fit(X_train_processed, y_train)
                
                # Get best model
                best_model = grid_search.best_estimator_
                
                # Evaluate performance
                if X_val_processed is not None and y_val is not None:
                    y_pred = best_model.predict(X_val_processed)
                    mse = mean_squared_error(y_val, y_pred)
                    mae = mean_absolute_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)
                    mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
                else:
                    # Use cross-validation scores
                    cv_scores = cross_val_score(best_model, X_train_processed, y_train,
                                              cv=cv_folds, scoring='neg_root_mean_squared_error')
                    mse = (-cv_scores.mean()) ** 2
                    mae = None
                    r2 = None
                    mape = None
                
                # Store results
                results[model_name] = {
                    'model': best_model,
                    'best_params': grid_search.best_params_,
                    'cv_score': grid_search.best_score_,
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'mae': mae,
                    'r2': r2,
                    'mape': mape
                }
                
                # Store in class attributes
                self.valuation_models[model_name] = best_model
                self.model_performance[f'valuation_{model_name}'] = results[model_name]
                
                self.logger.info(f"{model_name} - RMSE: {np.sqrt(mse):.2f}, R²: {r2:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Select best model
        if results:
            best_model_name = min(results.keys(), key=lambda x: results[x]['mse'])
            self.best_valuation_model = self.valuation_models[best_model_name]
            self.logger.info(f"Best valuation model: {best_model_name}")
        
        return results
    
    def create_market_forecasting_models(self) -> Dict[str, Any]:
        """Create time series forecasting models for market analysis."""
        models = {
            'arima_baseline': {
                'description': 'ARIMA baseline model',
                'seasonal': True
            },
            'lstm_deep': {
                'description': 'LSTM neural network for time series',
                'sequence_length': 12,
                'features': ['price', 'volume', 'inventory']
            },
            'ensemble_forecast': {
                'description': 'Ensemble of multiple forecasting methods',
                'methods': ['linear_trend', 'seasonal_decompose', 'exponential_smoothing']
            }
        }
        
        return models
    
    def train_market_forecasting_models(self, market_data: pd.DataFrame,
                                      target_column: str = 'median_price',
                                      forecast_horizon: int = 12) -> Dict[str, Any]:
        """Train market forecasting models for price and trend prediction."""
        self.logger.info("Training market forecasting models...")
        
        results = {}
        
        # Prepare time series data
        ts_data = self._prepare_time_series_data(market_data, target_column)
        
        # Simple trend model
        if len(ts_data) >= 24:  # Need at least 2 years of data
            trend_model = self._train_trend_model(ts_data)
            results['trend_model'] = trend_model
            self.forecasting_models['trend'] = trend_model
        
        # LSTM model if deep learning is available
        if DEEP_LEARNING_AVAILABLE and len(ts_data) >= 36:
            lstm_model = self._train_lstm_forecasting_model(ts_data, forecast_horizon)
            if lstm_model:
                results['lstm_model'] = lstm_model
                self.forecasting_models['lstm'] = lstm_model
        
        # Seasonal decomposition model
        seasonal_model = self._train_seasonal_model(ts_data)
        results['seasonal_model'] = seasonal_model
        self.forecasting_models['seasonal'] = seasonal_model
        
        return results
    
    def _encode_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Encode categorical features for machine learning."""
        X_train_encoded = X_train.copy()
        X_val_encoded = X_val.copy() if X_val is not None else None
        
        # Identify categorical columns
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col not in self.encoders:
                # Use label encoding for now (could be extended to one-hot encoding)
                encoder = LabelEncoder()
                encoder.fit(X_train[col].astype(str))
                self.encoders[col] = encoder
            
            # Transform training data
            X_train_encoded[col] = self.encoders[col].transform(X_train[col].astype(str))
            
            # Transform validation data if provided
            if X_val_encoded is not None:
                # Handle unseen categories
                val_values = X_val[col].astype(str)
                known_categories = set(self.encoders[col].classes_)
                val_values_mapped = val_values.map(
                    lambda x: x if x in known_categories else self.encoders[col].classes_[0]
                )
                X_val_encoded[col] = self.encoders[col].transform(val_values_mapped)
        
        return X_train_encoded, X_val_encoded
    
    def _prepare_time_series_data(self, market_data: pd.DataFrame, target_column: str) -> pd.Series:
        """Prepare time series data for forecasting."""
        if 'date' in market_data.columns:
            market_data = market_data.sort_values('date')
            ts_data = market_data.set_index('date')[target_column]
        else:
            ts_data = market_data[target_column]
        
        # Remove any NaN values
        ts_data = ts_data.dropna()
        
        return ts_data
    
    def _train_trend_model(self, ts_data: pd.Series) -> Dict[str, Any]:
        """Train simple trend forecasting model."""
        from sklearn.linear_model import LinearRegression
        
        # Create time index
        X = np.arange(len(ts_data)).reshape(-1, 1)
        y = ts_data.values
        
        # Fit linear trend
        trend_model = LinearRegression()
        trend_model.fit(X, y)
        
        # Calculate performance
        y_pred = trend_model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        return {
            'model': trend_model,
            'mse': mse,
            'r2': r2,
            'model_type': 'linear_trend'
        }
    
    def _train_lstm_forecasting_model(self, ts_data: pd.Series, forecast_horizon: int) -> Optional[Dict[str, Any]]:
        """Train LSTM model for time series forecasting."""
        if not DEEP_LEARNING_AVAILABLE:
            return None
        
        try:
            # Prepare sequences for LSTM
            sequence_length = min(12, len(ts_data) // 4)
            X, y = self._create_sequences(ts_data.values, sequence_length)
            
            if len(X) < 10:  # Need minimum samples
                return None
            
            # Split into train/test
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Reshape for LSTM
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
                Dropout(0.2),
                LSTM(25, return_sequences=False),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=16,
                validation_data=(X_test, y_test),
                verbose=0,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
            )
            
            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return {
                'model': model,
                'mse': mse,
                'r2': r2,
                'history': history,
                'sequence_length': sequence_length,
                'model_type': 'lstm'
            }
            
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {str(e)}")
            return None
    
    def _train_seasonal_model(self, ts_data: pd.Series) -> Dict[str, Any]:
        """Train seasonal decomposition model."""
        # Simple seasonal model using moving averages
        if len(ts_data) >= 12:
            # Calculate seasonal components
            seasonal_period = 12 if len(ts_data) >= 24 else len(ts_data) // 2
            
            # Moving average trend
            trend = ts_data.rolling(window=seasonal_period, center=True).mean()
            
            # Seasonal component
            detrended = ts_data - trend
            seasonal = detrended.groupby(detrended.index % seasonal_period).mean()
            
            # Residual
            seasonal_full = pd.Series(index=ts_data.index, dtype=float)
            for i in range(len(ts_data)):
                seasonal_full.iloc[i] = seasonal.iloc[i % seasonal_period]
            
            residual = ts_data - trend - seasonal_full
            
            return {
                'trend': trend,
                'seasonal': seasonal,
                'residual': residual,
                'seasonal_period': seasonal_period,
                'model_type': 'seasonal_decomposition'
            }
        else:
            # Not enough data for seasonal analysis
            return {
                'trend': ts_data.rolling(window=3).mean(),
                'seasonal': None,
                'residual': None,
                'seasonal_period': None,
                'model_type': 'simple_trend'
            }
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def predict_property_value(self, property_features: pd.DataFrame, 
                             model_name: str = None) -> Union[float, np.ndarray]:
        """Predict property value using trained valuation models."""
        if model_name is None:
            model = self.best_valuation_model
            if model is None:
                raise ValueError("No valuation models trained yet")
        else:
            if model_name not in self.valuation_models:
                raise ValueError(f"Model {model_name} not found")
            model = self.valuation_models[model_name]
        
        # Encode features
        features_encoded, _ = self._encode_features(property_features)
        
        # Make prediction
        prediction = model.predict(features_encoded)
        
        return prediction[0] if len(prediction) == 1 else prediction
    
    def forecast_market_trends(self, historical_data: pd.DataFrame,
                             forecast_horizon: int = 12,
                             model_name: str = None) -> Dict[str, Any]:
        """Forecast market trends using trained forecasting models."""
        if model_name is None:
            # Use best available model
            model_name = 'seasonal' if 'seasonal' in self.forecasting_models else list(self.forecasting_models.keys())[0]
        
        if model_name not in self.forecasting_models:
            raise ValueError(f"Forecasting model {model_name} not found")
        
        model = self.forecasting_models[model_name]
        
        # Generate forecasts based on model type
        if model.get('model_type') == 'linear_trend':
            return self._forecast_trend_model(model, historical_data, forecast_horizon)
        elif model.get('model_type') == 'seasonal_decomposition':
            return self._forecast_seasonal_model(model, historical_data, forecast_horizon)
        elif model.get('model_type') == 'lstm':
            return self._forecast_lstm_model(model, historical_data, forecast_horizon)
        else:
            raise ValueError(f"Unknown model type for {model_name}")
    
    def _forecast_trend_model(self, model: Dict[str, Any], historical_data: pd.DataFrame,
                            forecast_horizon: int) -> Dict[str, Any]:
        """Generate forecasts using trend model."""
        trend_model = model['model']
        
        # Future time points
        last_time_point = len(historical_data)
        future_X = np.arange(last_time_point, last_time_point + forecast_horizon).reshape(-1, 1)
        
        # Generate predictions
        forecasts = trend_model.predict(future_X)
        
        return {
            'forecasts': forecasts,
            'model_type': 'linear_trend',
            'confidence_interval': None  # Could add confidence intervals
        }
    
    def _forecast_seasonal_model(self, model: Dict[str, Any], historical_data: pd.DataFrame,
                               forecast_horizon: int) -> Dict[str, Any]:
        """Generate forecasts using seasonal decomposition model."""
        seasonal_period = model.get('seasonal_period', 12)
        
        if seasonal_period is None:
            # Simple trend continuation
            last_values = historical_data.iloc[-3:]['median_price'].mean()
            forecasts = np.full(forecast_horizon, last_values)
        else:
            # Use seasonal pattern
            seasonal = model['seasonal']
            trend = model['trend']
            
            # Project trend forward
            trend_values = trend.dropna()
            if len(trend_values) >= 2:
                trend_slope = (trend_values.iloc[-1] - trend_values.iloc[-2])
                last_trend = trend_values.iloc[-1]
            else:
                trend_slope = 0
                last_trend = historical_data['median_price'].mean()
            
            # Generate forecasts
            forecasts = []
            for i in range(forecast_horizon):
                seasonal_component = seasonal.iloc[i % seasonal_period]
                trend_component = last_trend + (i + 1) * trend_slope
                forecast = trend_component + seasonal_component
                forecasts.append(forecast)
            
            forecasts = np.array(forecasts)
        
        return {
            'forecasts': forecasts,
            'model_type': 'seasonal_decomposition',
            'seasonal_period': seasonal_period
        }
    
    def _forecast_lstm_model(self, model: Dict[str, Any], historical_data: pd.DataFrame,
                           forecast_horizon: int) -> Dict[str, Any]:
        """Generate forecasts using LSTM model."""
        if not DEEP_LEARNING_AVAILABLE:
            raise ValueError("TensorFlow not available for LSTM forecasting")
        
        lstm_model = model['model']
        sequence_length = model['sequence_length']
        
        # Prepare last sequence
        last_sequence = historical_data['median_price'].iloc[-sequence_length:].values
        last_sequence = last_sequence.reshape((1, sequence_length, 1))
        
        # Generate forecasts iteratively
        forecasts = []
        current_sequence = last_sequence.copy()
        
        for _ in range(forecast_horizon):
            next_pred = lstm_model.predict(current_sequence, verbose=0)[0, 0]
            forecasts.append(next_pred)
            
            # Update sequence
            new_sequence = np.roll(current_sequence[0], -1, axis=0)
            new_sequence[-1, 0] = next_pred
            current_sequence = new_sequence.reshape((1, sequence_length, 1))
        
        return {
            'forecasts': np.array(forecasts),
            'model_type': 'lstm',
            'sequence_length': sequence_length
        }
    
    def get_feature_importance(self, model_name: str = None, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from tree-based models."""
        if model_name is None:
            model = self.best_valuation_model
            model_name = 'best_model'
        else:
            if model_name not in self.valuation_models:
                raise ValueError(f"Model {model_name} not found")
            model = self.valuation_models[model_name]
        
        # Extract the actual model from pipeline
        if hasattr(model, 'named_steps'):
            actual_model = model.named_steps['model']
        else:
            actual_model = model
        
        # Get feature importance
        if hasattr(actual_model, 'feature_importances_'):
            importances = actual_model.feature_importances_
            
            # Create DataFrame (assuming we have feature names stored)
            feature_names = getattr(self, 'feature_names', [f'feature_{i}' for i in range(len(importances))])
            
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(importances)],
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
            return importance_df
        else:
            self.logger.warning(f"Model {model_name} does not support feature importance")
            return pd.DataFrame()
    
    def save_models(self, filepath: str) -> None:
        """Save all trained models to disk."""
        self.logger.info(f"Saving models to {filepath}")
        
        model_data = {
            'valuation_models': self.valuation_models,
            'forecasting_models': self.forecasting_models,
            'classification_models': self.classification_models,
            'clustering_models': self.clustering_models,
            'model_performance': self.model_performance,
            'feature_importance': self.feature_importance,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'best_valuation_model': self.best_valuation_model,
            'best_forecasting_model': self.best_forecasting_model
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info("Models saved successfully")
    
    def load_models(self, filepath: str) -> None:
        """Load trained models from disk."""
        self.logger.info(f"Loading models from {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.valuation_models = model_data.get('valuation_models', {})
        self.forecasting_models = model_data.get('forecasting_models', {})
        self.classification_models = model_data.get('classification_models', {})
        self.clustering_models = model_data.get('clustering_models', {})
        self.model_performance = model_data.get('model_performance', {})
        self.feature_importance = model_data.get('feature_importance', {})
        self.scalers = model_data.get('scalers', {})
        self.encoders = model_data.get('encoders', {})
        self.best_valuation_model = model_data.get('best_valuation_model')
        self.best_forecasting_model = model_data.get('best_forecasting_model')
        
        self.logger.info("Models loaded successfully")
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all trained models."""
        summary_data = []
        
        for model_key, metrics in self.model_performance.items():
            if isinstance(metrics, dict):
                summary_data.append({
                    'Model': model_key,
                    'RMSE': metrics.get('rmse', np.nan),
                    'MAE': metrics.get('mae', np.nan),
                    'R²': metrics.get('r2', np.nan),
                    'MAPE': metrics.get('mape', np.nan),
                    'CV_Score': metrics.get('cv_score', np.nan)
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values('RMSE', na_last=True)
            return summary_df
        else:
            return pd.DataFrame()


def main():
    """Main function demonstrating Real Estate ML Manager usage."""
    print("=" * 80)
    print("Real Estate Market Intelligence Platform")
    print("Machine Learning Models Module")
    print("Author: Emilio Cardenas | MIT PhD AI Automation | Harvard MBA")
    print("=" * 80)
    
    # Initialize ML Manager
    ml_manager = RealEstateMLManager(random_state=42)
    
    # Generate sample real estate data for demonstration
    np.random.seed(42)
    n_properties = 1000
    
    # Create synthetic property dataset
    property_data = pd.DataFrame({
        'property_id': [f'PROP_{i:04d}' for i in range(n_properties)],
        'square_footage': np.random.normal(2000, 500, n_properties).clip(500, 5000),
        'year_built': np.random.randint(1950, 2024, n_properties),
        'bedrooms': np.random.choice([2, 3, 4, 5], n_properties, p=[0.2, 0.4, 0.3, 0.1]),
        'bathrooms': np.random.uniform(1, 4, n_properties).round(1),
        'lot_size': np.random.exponential(0.3, n_properties).clip(0.1, 2.0),
        'property_type': np.random.choice(['Single Family', 'Condo', 'Townhouse'], n_properties),
        'neighborhood': np.random.choice(['Downtown', 'Suburbs', 'Waterfront', 'Historic'], n_properties),
        'latitude': np.random.uniform(40.7, 40.8, n_properties),
        'longitude': np.random.uniform(-74.1, -73.9, n_properties),
        'sale_date': pd.date_range(start='2020-01-01', end='2023-12-31', periods=n_properties)
    })
    
    # Generate realistic property values
    base_price_per_sqft = 300
    property_data['property_value'] = (
        property_data['square_footage'] * base_price_per_sqft *
        (1 + np.random.normal(0, 0.3, n_properties)) *
        (2024 - property_data['year_built'] + 1) / 50  # Age adjustment
    )
    
    # Add rental income
    property_data['annual_rent'] = property_data['property_value'] / 120  # GRM of 120
    
    print(f"\nGenerated {len(property_data)} synthetic properties")
    print(f"Property value range: ${property_data['property_value'].min():,.0f} - ${property_data['property_value'].max():,.0f}")
    
    # Prepare data for modeling
    prepared_data = ml_manager.prepare_property_data(property_data)
    print(f"Prepared dataset shape: {prepared_data.shape}")
    
    # Define features and target
    feature_columns = [
        'square_footage', 'property_age', 'bedrooms', 'bathrooms', 'lot_size',
        'size_efficiency', 'gross_yield', 'price_per_sqft', 'distance_to_center'
    ]
    
    # Filter to available columns
    available_features = [col for col in feature_columns if col in prepared_data.columns]
    
    # Add categorical features
    categorical_features = ['property_type', 'neighborhood', 'age_category', 'size_category']
    available_categorical = [col for col in categorical_features if col in prepared_data.columns]
    
    all_features = available_features + available_categorical
    print(f"Using {len(all_features)} features: {all_features}")
    
    # Prepare training data
    X = prepared_data[all_features].copy()
    y = prepared_data['property_value']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set: {len(X_train)} properties")
    print(f"Test set: {len(X_test)} properties")
    
    # Store feature names for later use
    ml_manager.feature_names = all_features
    
    # Train valuation models
    print("\nTraining property valuation models...")
    valuation_results = ml_manager.train_valuation_models(X_train, y_train, X_test, y_test)
    
    print(f"\nTrained {len(valuation_results)} valuation models")
    
    # Show model performance summary
    print("\nModel Performance Summary:")
    summary = ml_manager.get_model_summary()
    if not summary.empty:
        print(summary.to_string(index=False))
    
    # Test prediction
    print("\nTesting property value prediction...")
    sample_property = X_test.iloc[[0]]
    predicted_value = ml_manager.predict_property_value(sample_property)
    actual_value = y_test.iloc[0]
    
    print(f"Predicted value: ${predicted_value:,.0f}")
    print(f"Actual value: ${actual_value:,.0f}")
    print(f"Prediction error: {abs(predicted_value - actual_value) / actual_value * 100:.1f}%")
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    feature_importance = ml_manager.get_feature_importance(top_n=10)
    if not feature_importance.empty:
        for _, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Create market data for forecasting
    market_data = pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', end='2023-12-31', freq='M'),
    })
    
    # Generate synthetic market trends
    trend = np.linspace(400000, 500000, len(market_data))
    seasonal = 10000 * np.sin(2 * np.pi * np.arange(len(market_data)) / 12)
    noise = np.random.normal(0, 5000, len(market_data))
    market_data['median_price'] = trend + seasonal + noise
    market_data['sales_volume'] = np.random.randint(100, 500, len(market_data))
    
    print(f"\nTraining market forecasting models with {len(market_data)} months of data...")
    
    # Train forecasting models
    forecasting_results = ml_manager.train_market_forecasting_models(market_data)
    print(f"Trained {len(forecasting_results)} forecasting models")
    
    # Test forecasting
    if forecasting_results:
        print("\nTesting market trend forecasting...")
        forecast_results = ml_manager.forecast_market_trends(market_data, forecast_horizon=6)
        
        print(f"6-month forecast (starting from ${market_data['median_price'].iloc[-1]:,.0f}):")
        for i, forecast in enumerate(forecast_results['forecasts']):
            print(f"  Month {i+1}: ${forecast:,.0f}")
    
    print("\nReal Estate ML Manager demonstration completed successfully!")


if __name__ == "__main__":
    main()