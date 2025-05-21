import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from model_trainer import train_models
from data_handler import preprocess_data, split_data

def load_excel_data(file_path, start_date=None, end_date=None):
    """
    Load data from Excel file with proper formatting
    
    Args:
        file_path (str): Path to the Excel file
        start_date (str): Optional start date filter in 'YYYY-MM-DD' format
        end_date (str): Optional end date filter in 'YYYY-MM-DD' format
    
    Returns:
        pd.DataFrame: Processed dataframe ready for model training
    """
    # Load the Excel file skipping the first row (header)
    df = pd.read_excel(file_path, header=1)
    
    print("Excel columns:", df.columns.tolist())
    print("First few rows:")
    print(df.head())
    
    # Rename columns to standard names used in our application
    # Extract the proper columns based on what's actually in the file
    data = pd.DataFrame()
    
    # Find the right columns in the file
    date_column = next((col for col in df.columns if 'Date' in str(col) and 'From' in str(col)), None)
    pm25_column = next((col for col in df.columns if 'PM2.5' in str(col)), None)
    temp_column = next((col for col in df.columns if 'Temp' in str(col)), None)
    rh_column = next((col for col in df.columns if 'RH' in str(col)), None)
    ws_column = next((col for col in df.columns if 'WS' in str(col)), None)
    sr_column = next((col for col in df.columns if 'SR' in str(col)), None)
    
    # Map columns to our standard names
    if date_column:
        data['Date'] = df[date_column]
    if pm25_column:
        data['PM2.5'] = df[pm25_column]
    if temp_column:
        data['Temperature'] = df[temp_column]
    if rh_column:
        data['Relative_Humidity'] = df[rh_column]
    if ws_column:
        data['Wind_Speed'] = df[ws_column]
    if sr_column:
        data['Solar_Radiation'] = df[sr_column]
    
    # If we didn't find the columns, try a different approach
    if len(data.columns) == 0:
        # The file might not have proper headers, so try to use the data directly
        data = pd.DataFrame()
        
        # Try fixed positions
        if len(df.columns) >= 6:
            data['Date'] = df.iloc[:, 0]
            data['PM2.5'] = df.iloc[:, 2]
            data['Temperature'] = df.iloc[:, 3]
            data['Relative_Humidity'] = df.iloc[:, 4]
            data['Wind_Speed'] = df.iloc[:, 5]
            data['Solar_Radiation'] = df.iloc[:, 6] if len(df.columns) > 6 else np.nan
    
    # Add station information
    data['Station'] = 'Kolkata'
    
    # Convert date to datetime with the correct format (DD-MM-YYYY)
    try:
        data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y %H:%M')
    except:
        try:
            # Try alternative format
            data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
        except:
            print("Warning: Date conversion failed. Using original dates.")
    
    # Filter by date range if provided
    if start_date:
        start_date = pd.to_datetime(start_date)
        data = data[data['Date'] >= start_date]
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        data = data[data['Date'] <= end_date]
    
    # Sort by date
    data = data.sort_values('Date').reset_index(drop=True)
    
    print(f"Loaded dataset with {len(data)} records")
    if len(data) > 0:
        print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    
    return data

def clean_and_prepare_data(data):
    """
    Clean the dataset by handling missing values and outliers
    
    Args:
        data (pandas.DataFrame): The original dataset
        
    Returns:
        pandas.DataFrame: Cleaned dataset
    """
    # Make a copy to avoid modifying the original
    cleaned_data = data.copy()
    
    # Check for missing values
    missing_counts = cleaned_data.isnull().sum()
    print("Missing value counts before cleaning:")
    print(missing_counts)
    
    # Handle missing values
    # For numeric columns, fill with the median value
    numeric_cols = cleaned_data.select_dtypes(include=['float', 'int']).columns
    for col in numeric_cols:
        if cleaned_data[col].isnull().sum() > 0:
            median_val = cleaned_data[col].median()
            cleaned_data[col] = cleaned_data[col].fillna(median_val)
            print(f"Filled {col} missing values with median: {median_val}")
    
    # Check if cleaning was successful
    missing_counts_after = cleaned_data.isnull().sum()
    print("Missing value counts after cleaning:")
    print(missing_counts_after)
    
    # Handle outliers (optional)
    # You can add outlier detection and handling here if needed
    
    return cleaned_data

def main():
    # Define paths
    excel_file = 'attached_assets/edited dataset of kolkata.xlsx'
    model_output = 'pm_model.pkl'
    
    # Define date range
    start_date = '2021-01-01'
    end_date = '2025-01-15'
    
    print("Loading dataset...")
    # Load data from Excel file
    data = load_excel_data(excel_file, start_date, end_date)
    
    print("Cleaning data...")
    # Clean the data (handle missing values)
    cleaned_data = clean_and_prepare_data(data)
    
    print("Preprocessing data...")
    # Preprocess the data
    processed_data = preprocess_data(cleaned_data)
    
    # Manual imputation step for any remaining NaN values
    # This ensures all features are complete before model training
    imputer = SimpleImputer(strategy='median')
    
    # Get feature names (excluding target)
    feature_names = [col for col in processed_data.columns if col != 'PM2.5']
    
    # Apply imputer to features
    feature_data = processed_data[feature_names]
    feature_data_imputed = pd.DataFrame(
        imputer.fit_transform(feature_data),
        columns=feature_names
    )
    
    # Reconstruct the complete dataset
    processed_data_imputed = pd.DataFrame()
    processed_data_imputed[feature_names] = feature_data_imputed
    processed_data_imputed['PM2.5'] = processed_data['PM2.5'].fillna(processed_data['PM2.5'].median())
    
    print("Splitting data for training...")
    # Split data for training
    X_train, X_test, y_train, y_test = split_data(processed_data_imputed, test_size=0.2)
    
    print("Training models...")
    # Train models
    models = train_models(X_train, y_train)
    
    print("Saving trained model to", model_output)
    # Save the trained models
    with open(model_output, 'wb') as f:
        pickle.dump(models, f)
    
    print("Model training and saving complete!")

if __name__ == "__main__":
    main()