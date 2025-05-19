import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

def load_data():
    """
    Load the PM2.5 dataset from the data directory.
    If file doesn't exist, create a synthetic dataset for demonstration.
    
    Returns:
        pandas.DataFrame: The loaded or generated dataset
    """
    try:
        # Try to load the dataset
        data_path = os.path.join('data', 'pm25_dataset.csv')
        data = pd.read_csv(data_path)
        return data
    except (FileNotFoundError, pd.errors.EmptyDataError):
        # If dataset doesn't exist or is empty, create a synthetic dataset
        print("Dataset not found. Creating a synthetic dataset for demonstration.")
        return generate_synthetic_dataset()

def generate_synthetic_dataset():
    """
    Generate a synthetic PM2.5 dataset for demonstration purposes.
    
    Returns:
        pandas.DataFrame: A synthetic dataset with realistic patterns
    """
    # Set a seed for reproducibility
    np.random.seed(42)
    
    # Generate dates from 2021 to present day (daily data)
    start_date = pd.to_datetime('2021-01-01')
    end_date = pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create dataframes for two stations
    stations = ['Fort William', 'Bidhannagar']
    data_frames = []
    
    for station in stations:
        # Base values and seasonal patterns for parameters
        temp_base = 25 if station == 'Fort William' else 27
        solar_base = 200 if station == 'Fort William' else 210
        wind_base = 3.5 if station == 'Fort William' else 3.0
        humidity_base = 65 if station == 'Fort William' else 70
        pm25_base = 35 if station == 'Fort William' else 40
        
        # Create dataframe for this station
        station_data = pd.DataFrame({
            'Date': dates,
            'Station': station
        })
        
        # Add seasonality and random variation to parameters
        # Temperature: higher in summer, lower in winter
        station_data['Temperature'] = temp_base + 10 * np.sin(2 * np.pi * (station_data['Date'].dt.dayofyear / 365)) + np.random.normal(0, 2, len(station_data))
        
        # Solar Radiation: higher in summer, lower in winter, daily variation
        station_data['Solar_Radiation'] = solar_base + 100 * np.sin(2 * np.pi * (station_data['Date'].dt.dayofyear / 365)) + np.random.normal(0, 30, len(station_data))
        station_data['Solar_Radiation'] = station_data['Solar_Radiation'].clip(0)  # No negative solar radiation
        
        # Wind Speed: slight seasonal variation with random fluctuations
        station_data['Wind_Speed'] = wind_base + np.sin(2 * np.pi * (station_data['Date'].dt.dayofyear / 365)) + np.random.normal(0, 1, len(station_data))
        station_data['Wind_Speed'] = station_data['Wind_Speed'].clip(0.5)  # Minimum wind speed
        
        # Relative Humidity: higher in monsoon, lower in winter
        station_data['Relative_Humidity'] = humidity_base + 15 * np.sin(2 * np.pi * ((station_data['Date'].dt.dayofyear + 60) % 365 / 365)) + np.random.normal(0, 5, len(station_data))
        station_data['Relative_Humidity'] = station_data['Relative_Humidity'].clip(30, 100)  # Humidity range
        
        # PM2.5: influenced by other parameters with some lag and randomness
        # Higher when temperature is extreme, wind is low, and humidity is high
        station_data['PM2.5'] = (
            pm25_base 
            + 0.5 * (abs(station_data['Temperature'] - temp_base)) 
            - 3 * station_data['Wind_Speed'] 
            + 0.2 * station_data['Relative_Humidity']
            + 0.05 * station_data['Solar_Radiation']
            + np.random.normal(0, 10, len(station_data))
        )
        station_data['PM2.5'] = station_data['PM2.5'].clip(5)  # Minimum PM2.5 level
        
        # Add weekly pattern (higher on weekdays, lower on weekends)
        weekday_effect = np.where(station_data['Date'].dt.dayofweek < 5, 5, -5)
        station_data['PM2.5'] += weekday_effect
        
        data_frames.append(station_data)
    
    # Combine data from both stations
    combined_data = pd.concat(data_frames, ignore_index=True)
    
    # Sort by date and station
    combined_data = combined_data.sort_values(['Date', 'Station']).reset_index(drop=True)
    
    # Save the synthetic dataset to a file
    os.makedirs('data', exist_ok=True)
    combined_data.to_csv(os.path.join('data', 'pm25_dataset.csv'), index=False)
    
    return combined_data

def preprocess_data(data):
    """
    Preprocess the dataset for model training.
    
    Args:
        data (pandas.DataFrame): The dataset to preprocess
        
    Returns:
        pandas.DataFrame: The preprocessed dataset
    """
    # Create a copy of the data to avoid modifying the original
    processed_data = data.copy()
    
    # Convert date to datetime if it's not already
    processed_data['Date'] = pd.to_datetime(processed_data['Date'])
    
    # Extract time-based features
    processed_data['Month'] = processed_data['Date'].dt.month
    processed_data['Day'] = processed_data['Date'].dt.day
    processed_data['DayOfWeek'] = processed_data['Date'].dt.dayofweek
    processed_data['DayOfYear'] = processed_data['Date'].dt.dayofyear
    
    # Add seasonal features (sin and cos transformations for cyclical features)
    processed_data['Month_sin'] = np.sin(2 * np.pi * processed_data['Month'] / 12)
    processed_data['Month_cos'] = np.cos(2 * np.pi * processed_data['Month'] / 12)
    processed_data['DayOfWeek_sin'] = np.sin(2 * np.pi * processed_data['DayOfWeek'] / 7)
    processed_data['DayOfWeek_cos'] = np.cos(2 * np.pi * processed_data['DayOfWeek'] / 7)
    processed_data['DayOfYear_sin'] = np.sin(2 * np.pi * processed_data['DayOfYear'] / 365)
    processed_data['DayOfYear_cos'] = np.cos(2 * np.pi * processed_data['DayOfYear'] / 365)
    
    # Drop unnecessary columns for model training
    columns_to_drop = ['Date', 'Station', 'Month', 'Day', 'DayOfWeek', 'DayOfYear']
    model_data = processed_data.drop(columns=columns_to_drop)
    
    return model_data

def split_data(data, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    
    Args:
        data (pandas.DataFrame): The preprocessed dataset
        test_size (float): The proportion of the dataset to include in the test split
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - The split datasets
    """
    # Define features and target
    X = data.drop('PM2.5', axis=1)
    y = data['PM2.5']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test
