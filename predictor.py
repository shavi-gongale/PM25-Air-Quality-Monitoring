import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_forecast_data(forecast_data):
    """
    Prepare forecast data for prediction.
    
    Args:
        forecast_data (list): List of dictionaries containing forecast data
        
    Returns:
        np.ndarray: Prepared forecast data for prediction
    """
    # Convert forecast data to DataFrame
    forecast_df = pd.DataFrame(forecast_data)
    
    # Extract date information
    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
    forecast_df['Month'] = forecast_df['Date'].dt.month
    forecast_df['DayOfWeek'] = forecast_df['Date'].dt.dayofweek
    forecast_df['DayOfYear'] = forecast_df['Date'].dt.dayofyear
    
    # Add seasonal features
    forecast_df['Month_sin'] = np.sin(2 * np.pi * forecast_df['Month'] / 12)
    forecast_df['Month_cos'] = np.cos(2 * np.pi * forecast_df['Month'] / 12)
    forecast_df['DayOfWeek_sin'] = np.sin(2 * np.pi * forecast_df['DayOfWeek'] / 7)
    forecast_df['DayOfWeek_cos'] = np.cos(2 * np.pi * forecast_df['DayOfWeek'] / 7)
    forecast_df['DayOfYear_sin'] = np.sin(2 * np.pi * forecast_df['DayOfYear'] / 365)
    forecast_df['DayOfYear_cos'] = np.cos(2 * np.pi * forecast_df['DayOfYear'] / 365)
    
    # Select features for prediction (same order as in training)
    features = [
        'Temperature', 'Solar_Radiation', 'Wind_Speed', 'Relative_Humidity',
        'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 
        'DayOfYear_sin', 'DayOfYear_cos'
    ]
    
    # Scale the features
    scaler = StandardScaler()
    X_forecast = scaler.fit_transform(forecast_df[features])
    
    return X_forecast

def predict_pm25(models, forecast_data):
    """
    Predict PM2.5 values using trained models and forecast data.
    
    Args:
        models (dict): Trained models
        forecast_data (list): List of dictionaries containing forecast data
        
    Returns:
        dict: Predictions from each model
    """
    # Prepare forecast data
    X_forecast = prepare_forecast_data(forecast_data)
    
    # Make predictions with each model
    predictions = {}
    
    for name, model in models.items():
        predictions[name] = model.predict(X_forecast)
    
    return predictions
