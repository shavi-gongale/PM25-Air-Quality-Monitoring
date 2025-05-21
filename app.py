import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from io import BytesIO
import base64
import json
import pickle

# Import custom modules
from data_handler import load_data, preprocess_data, split_data
from model_trainer import train_models, evaluate_models
from predictor import predict_pm25
from visualizer import (
    plot_historical_data, 
    plot_correlation_matrix, 
    plot_parameter_comparison, 
    plot_model_comparison,
    plot_forecast
)
from weather_api import get_weather_forecast
from utils import get_download_link
from satellite_data import get_satellite_imagery, get_air_quality_map, get_historical_satellite_data

# Function to load the pre-trained model
@st.cache_resource  # Cache the model in memory
def load_pretrained_model():
    try:
        with open('pm_model.pkl', 'rb') as f:
            models = pickle.load(f)
        st.success("Using pre-trained Kolkata PM2.5 model")
        return models
    except (FileNotFoundError, pickle.UnpicklingError):
        st.warning("Pre-trained model not found or invalid. Will train a new model.")
        return None

# Set page configuration
st.set_page_config(
    page_title="PM2.5 Air Quality Prediction",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add viewport meta tag for better mobile responsiveness
st.markdown(
    """
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    """,
    unsafe_allow_html=True
)

# Load custom CSS
with open('.streamlit/styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Create necessary folders for storage
if not os.path.exists('data/historical'):
    os.makedirs('data/historical', exist_ok=True)
if not os.path.exists('data/predictions'):
    os.makedirs('data/predictions', exist_ok=True)
if not os.path.exists('data/user_data'):
    os.makedirs('data/user_data', exist_ok=True)

# Load the dataset
@st.cache_data(ttl=300)  # Cache expires after 5 minutes
def get_data():
    data = load_data()
    # Ensure date range is from 2021-01-01 to 2025-01-15
    data['Date'] = pd.to_datetime(data['Date'])
    start_date = pd.to_datetime('2021-01-01')
    end_date = pd.to_datetime('2025-01-15')
    data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    return data

data = get_data()

# AQI level definitions and implications
aqi_levels = [
    {
        "level": "Good",
        "range": "0-50",
        "color": "#00e400",
        "implications": "Air quality is satisfactory, poses little or no risk"
    },
    {
        "level": "Moderate",
        "range": "51-100",
        "color": "#ffff00",
        "implications": "Acceptable air quality but may be a concern for sensitive people"
    },
    {
        "level": "Unhealthy for Sensitive",
        "range": "101-150",
        "color": "#ff7e00",
        "implications": "Members of sensitive groups may experience health effects"
    },
    {
        "level": "Unhealthy",
        "range": "151-200",
        "color": "#ff0000",
        "implications": "Everyone may begin to experience health effects"
    },
    {
        "level": "Very Unhealthy",
        "range": "201-300",
        "color": "#8f3f97",
        "implications": "Health alert: everyone may experience more serious health effects"
    },
    {
        "level": "Hazardous",
        "range": "301+",
        "color": "#7e0023",
        "implications": "Emergency conditions: entire population is likely to be affected"
    }
]

def get_aqi_category(pm25):
    """Get AQI category and class based on PM2.5 value"""
    if pm25 <= 50:
        return "Good", "good"
    elif pm25 <= 100:
        return "Moderate", "moderate"
    elif pm25 <= 150:
        return "Unhealthy for Sensitive", "unhealthy-for-sensitive-groups"
    elif pm25 <= 200:
        return "Unhealthy", "unhealthy"
    elif pm25 <= 300:
        return "Very Unhealthy", "very-unhealthy"
    else:
        return "Hazardous", "hazardous"

# Create a function to save user data
def save_user_data(data_type, data_dict):
    """Save user data to a JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{data_type}_{timestamp}.json"
    filepath = os.path.join('data', 'user_data', filename)
    
    with open(filepath, 'w') as f:
        json.dump(data_dict, f, indent=4)
    
    return filepath

# Function to list saved user data
def list_user_data(data_type=None):
    """List all saved user data files"""
    files = os.listdir(os.path.join('data', 'user_data'))
    if data_type:
        files = [f for f in files if f.startswith(f"{data_type}_")]
    
    files_with_info = []
    for file in files:
        filepath = os.path.join('data', 'user_data', file)
        created_time = datetime.fromtimestamp(os.path.getctime(filepath))
        size = os.path.getsize(filepath)
        files_with_info.append({
            'filename': file,
            'created': created_time.strftime("%Y-%m-%d %H:%M:%S"),
            'size_kb': size / 1024
        })
    
    return sorted(files_with_info, key=lambda x: x['created'], reverse=True)

# Function to load user data
def load_user_data(filename):
    """Load user data from a JSON file"""
    filepath = os.path.join('data', 'user_data', filename)
    with open(filepath, 'r') as f:
        return json.load(f)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Data Analysis and Insights", "Model Training & Comparison", "PM2.5 Predictions", "Satellite Imagery", "Historical Analysis", "Data Download"]
)

# Station selection for predictions
st.sidebar.title("Settings")
selected_station = st.sidebar.selectbox("Select Station", ["Fort William", "Bidhannagar"])

# Theme selection (dark/light mode)
st.sidebar.title("Display Settings")
theme_mode = st.sidebar.radio(
    "Select Theme",
    ["Light Mode", "Dark Mode"]
)

# Apply dark mode if selected
if theme_mode == "Dark Mode":
    st.markdown("""
    <style>
    :root {
        --background-color: #0e1117;
        --secondary-background-color: #262730;
        --text-color: #fafafa;
        --font: "sans serif";
    }
    
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    .sidebar .sidebar-content {
        background-color: var(--secondary-background-color);
    }
    
    .Widget > label {
        color: var(--text-color);
    }
    
    .stTextInput > div > div > input {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
    }
    
    .stDataFrame {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
    }
    
    .weather-card {
        background-color: var(--secondary-background-color) !important;
    }
    
    .station-overview, .model-overview {
        background-color: var(--secondary-background-color) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Prediction time horizon
st.sidebar.subheader("Select Prediction Time Horizon")
prediction_horizons = st.sidebar.multiselect(
    "Prediction Days",
    ["Present Day", "1 Day Ahead", "2 Days Ahead", "4 Days Ahead", "7 Days Ahead"],
    default=["Present Day"]
)

# Display AQI scale in sidebar
st.sidebar.title("AQI Levels")

# Create a table for AQI levels
aqi_table_data = {
    "AQI Level": [level["level"] for level in aqi_levels],
    "Range": [level["range"] for level in aqi_levels],
    "Health Implications": [level["implications"] for level in aqi_levels]
}

aqi_df = pd.DataFrame(aqi_table_data)
st.sidebar.dataframe(aqi_df, hide_index=True)

# Dashboard page
if page == "Dashboard":
    # Station-specific title
    st.title(f"PM2.5 Air Quality Prediction for {selected_station}")
    
    # If data exists, get the latest PM2.5 reading for selected station
    if not data.empty:
        latest_data = data[data['Station'] == selected_station].iloc[-1]
        latest_pm25 = latest_data['PM2.5']
        latest_date = latest_data['Date']
        
        # Get AQI category
        aqi_category, aqi_class = get_aqi_category(latest_pm25)
        
        # Display current PM2.5 level with AQI category
        st.markdown(f"""
        <div style="background-color:{next(level['color'] for level in aqi_levels if level['level'] == aqi_category)}; 
                     padding:20px; border-radius:10px; margin:20px 0; text-align:center; color:white;">
            <h2>Current PM2.5 Level: {latest_pm25:.2f} μg/m³</h2>
            <h3>Air Quality: {aqi_category}</h3>
            <p>{next(level['implications'] for level in aqi_levels if level['level'] == aqi_category)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # About PM2.5 section
        st.header("About PM2.5 Air Quality Monitoring")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Display PM2.5 icon
            st.image("static/icons/pm25_icon.svg", width=150)
        
        with col2:
            st.markdown("""
            PM2.5 refers to atmospheric particulate matter with a diameter less than 2.5 micrometers. These fine 
            particles can penetrate deep into the lungs and bloodstream, causing significant health risks. This 
            application predicts PM2.5 levels based on weather parameters using multiple machine learning models.
            """)
        
        # Display current weather parameters
        st.subheader("Current Weather Parameters")
        
        # Get weather forecast data
        forecast_data = get_weather_forecast(selected_station)
        
        if forecast_data:
            current_weather = forecast_data[0]
            
            # Create weather parameter cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="weather-card">
                    <div class="weather-card-icon">🌡️</div>
                    <div class="weather-card-value">{current_weather['Temperature']:.1f}<span class="weather-card-unit">°C</span></div>
                    <div class="weather-card-label">Temperature</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="weather-card">
                    <div class="weather-card-icon">☀️</div>
                    <div class="weather-card-value">{current_weather['Solar_Radiation']:.1f}<span class="weather-card-unit">W/m²</span></div>
                    <div class="weather-card-label">Solar Radiation</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="weather-card">
                    <div class="weather-card-icon">💨</div>
                    <div class="weather-card-value">{current_weather['Wind_Speed']:.1f}<span class="weather-card-unit">km/h</span></div>
                    <div class="weather-card-label">Wind Speed</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="weather-card">
                    <div class="weather-card-icon">💧</div>
                    <div class="weather-card-value">{current_weather['Relative_Humidity']:.1f}<span class="weather-card-unit">%</span></div>
                    <div class="weather-card-label">Humidity</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Station Overview and Model Performance in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Station Overview")
            
            # Get station-specific data
            station_data = data[data['Station'] == selected_station]
            
            # Calculate date range and average PM2.5
            first_date = pd.to_datetime(station_data['Date'].min()).strftime("%Y-%m-%d")
            last_date = pd.to_datetime(station_data['Date'].max()).strftime("%Y-%m-%d")
            avg_pm25 = station_data['PM2.5'].mean()
            
            st.markdown(f"""
            <div class="station-overview">
                <p><strong>Location:</strong> {selected_station}</p>
                <p><strong>Data Range:</strong> {first_date} to {last_date}</p>
                <p><strong>Average PM2.5:</strong> {avg_pm25:.2f} μg/m³</p>
                <p><strong>Parameters:</strong> Temperature, Solar Radiation, Wind Speed, Relative Humidity</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Model Performance Overview")
            
            # Try to load pre-trained model first
            models = load_pretrained_model()
            
            if models is None:
                # If pre-trained model not available, train a new one
                processed_data = preprocess_data(station_data)
                X_train, X_test, y_train, y_test = split_data(processed_data, test_size=0.2)
                models = train_models(X_train, y_train)
                evaluation_results = evaluate_models(models, X_test, y_test)
            else:
                # If pre-trained model is loaded, evaluate it
                processed_data = preprocess_data(station_data)
                _, X_test, _, y_test = split_data(processed_data, test_size=0.2)
                evaluation_results = evaluate_models(models, X_test, y_test)
            
            # Find best model based on RMSE
            best_model_name = min(evaluation_results, key=lambda x: evaluation_results[x]['rmse'])
            best_rmse = evaluation_results[best_model_name]['rmse']
            best_r2 = evaluation_results[best_model_name]['r2']
            
            st.markdown(f"""
            <div class="model-overview">
                <p><strong>Best Performing Model:</strong> {best_model_name}</p>
                <p><strong>RMSE:</strong> {best_rmse:.2f}</p>
                <p><strong>R² Score:</strong> {best_r2:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Model comparison chart
            st.subheader("Model Comparison")
            
            # Create a dataframe for display
            results_df = pd.DataFrame({
                'Model': list(evaluation_results.keys()),
                'RMSE': [evaluation_results[model]['rmse'] for model in evaluation_results],
                'R²': [evaluation_results[model]['r2'] for model in evaluation_results]
            })
            
            # Plot model comparison
            fig_models = px.bar(
                results_df,
                x='Model',
                y='RMSE',
                color='Model',
                title='RMSE by Model (Lower is Better)',
                labels={'RMSE': 'Root Mean Square Error'}
            )
            
            st.plotly_chart(fig_models, use_container_width=True)
        
        # Forecasted PM2.5 Levels
        st.header("Forecasted PM2.5 Levels")
        
        # Make forecasts for selected days
        if forecast_data:
            # Use pre-trained model for predictions if available
            models = load_pretrained_model()
            
            if models is None:
                # If pre-trained model not available, train a new one
                processed_data = preprocess_data(station_data)
                X_train, X_test, y_train, y_test = split_data(processed_data, test_size=0.2)
                models = train_models(X_train, y_train)
            
            # Get predictions
            predictions = predict_pm25(models, forecast_data)
            
            # Create a dataframe for display
            forecast_df = pd.DataFrame({
                'Date': [item['Date'] for item in forecast_data]
            })
            
            # Add predictions for best model
            forecast_df[best_model_name] = predictions[best_model_name]
            
            # Create a card layout for forecast
            st.subheader(f"PM2.5 Forecast using {best_model_name}")
            
            # Filter based on selected horizons
            day_mapping = {
                "Present Day": 0,
                "1 Day Ahead": 1,
                "2 Days Ahead": 2,
                "4 Days Ahead": 4,
                "7 Days Ahead": 7
            }
            
            selected_days = [day_mapping[horizon] for horizon in prediction_horizons if horizon in day_mapping]
            
            # Create cards for selected days
            if selected_days:
                cols = st.columns(len(selected_days))
                
                for i, (col, day) in enumerate(zip(cols, selected_days)):
                    if day < len(forecast_df):
                        with col:
                            pred_date = forecast_df.iloc[day]['Date']
                            pred_value = forecast_df.iloc[day][best_model_name]
                            aqi_cat, aqi_cls = get_aqi_category(pred_value)
                            
                            st.markdown(f"""
                            <div class="{aqi_cls}" style="padding:15px; border-radius:10px; text-align:center; margin-bottom:10px;">
                                <h4>{pred_date}</h4>
                                <h2>{pred_value:.2f} μg/m³</h2>
                                <p>{aqi_cat}</p>
                            </div>
                            """, unsafe_allow_html=True)

# Data Analysis and Insights page
elif page == "Data Analysis and Insights":
    st.header("Data Analysis and Insights")
    
    if data.empty:
        st.warning("No data available for analysis.")
    else:
        # Create tabs for different analyses
        tabs = st.tabs(["Parameter Correlations", "Historical Trends", "Model Performance", "Forecasting Insights"])
        
        # Get station-specific data
        station_data = data[data['Station'] == selected_station]
        
        # Parameter Correlations tab
        with tabs[0]:
            st.subheader(f"Parameter Correlation Analysis for {selected_station}")
            
            # Plot correlation matrix
            fig_corr = plot_correlation_matrix(station_data)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Explanation of correlation matrix
            st.markdown("""
            #### Understanding the Correlation Matrix
            
            This heatmap shows how different parameters are related to each other:
            - Values close to 1 (red) indicate strong positive correlation
            - Values close to -1 (blue) indicate strong negative correlation
            - Values close to 0 indicate weak or no correlation
            
            **Key Insights:**
            - Wind speed tends to have a negative correlation with PM2.5, meaning higher wind speeds typically reduce PM2.5 levels
            - Temperature and humidity relationships with PM2.5 can vary by season and location
            - Understanding these relationships helps improve prediction accuracy
            """)
            
            # Individual parameter correlations with PM2.5
            st.subheader("Individual Parameter Correlations with PM2.5")
            
            # Let user select a parameter to analyze
            param = st.selectbox(
                "Select Parameter to Compare with PM2.5",
                ["Temperature", "Solar_Radiation", "Wind_Speed", "Relative_Humidity"]
            )
            
            # Plot the relationship
            fig_param = plot_parameter_comparison(station_data, param, "PM2.5")
            st.plotly_chart(fig_param, use_container_width=True)
            
            # Calculate and display correlation
            corr_value = station_data[param].corr(station_data["PM2.5"])
            
            # Interpret correlation strength
            if abs(corr_value) > 0.7:
                corr_strength = "Strong"
            elif abs(corr_value) > 0.4:
                corr_strength = "Moderate"
            else:
                corr_strength = "Weak"
            
            # Show if it's positive or negative
            corr_direction = "Positive" if corr_value > 0 else "Negative"
            
            st.markdown(f"""
            **Correlation between {param} and PM2.5:** {corr_value:.4f}
            
            This is a **{corr_strength} {corr_direction}** correlation.
            
            What this means:
            - {"As " + param + " increases, PM2.5 levels tend to increase as well." if corr_value > 0 else "As " + param + " increases, PM2.5 levels tend to decrease."}
            - This parameter {"has a significant impact" if abs(corr_value) > 0.4 else "has limited impact"} on PM2.5 levels in {selected_station}.
            """)
        
        # Historical Trends tab
        with tabs[1]:
            st.subheader(f"Historical PM2.5 Trends for {selected_station}")
            
            # Date range selection
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", min(pd.to_datetime(data['Date'])).date())
            with col2:
                end_date = st.date_input("End Date", max(pd.to_datetime(data['Date'])).date())
            
            if start_date <= end_date:
                # Filter data based on date range
                filtered_data = station_data[
                    (pd.to_datetime(station_data['Date']).dt.date >= start_date) &
                    (pd.to_datetime(station_data['Date']).dt.date <= end_date)
                ]
                
                # Plot historical trend
                fig_hist = plot_historical_data(filtered_data)
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Calculate trend
                filtered_data = filtered_data.sort_values('Date')
                filtered_data['day_index'] = range(len(filtered_data))
                
                # Only perform trend analysis if we have enough data points
                if len(filtered_data) > 2:
                    # Calculate trend line
                    z = np.polyfit(filtered_data['day_index'], filtered_data['PM2.5'], 1)
                    slope = z[0]
                    
                    # Interpret trend
                    if abs(slope) < 0.01:
                        trend_direction = "stable"
                    elif slope > 0:
                        trend_direction = "increasing"
                    else:
                        trend_direction = "decreasing"
                    
                    st.markdown(f"""
                    **Trend Analysis:**
                    
                    The PM2.5 levels in {selected_station} show a **{trend_direction}** trend over this period.
                    
                    {"- The increase rate is approximately " + f"{slope:.4f} μg/m³ per day." if slope > 0 else ""}
                    {"- The decrease rate is approximately " + f"{abs(slope):.4f} μg/m³ per day." if slope < 0 else ""}
                    {"- The PM2.5 levels remain relatively constant over this period." if abs(slope) < 0.01 else ""}
                    
                    This trend could be affected by:
                    - Seasonal variations
                    - Weather patterns
                    - Local pollution sources
                    - Implementation of air quality policies
                    """)
                
                # Monthly analysis
                st.subheader("Seasonal Patterns")
                
                # Add month column
                monthly_data = filtered_data.copy()
                monthly_data['Month'] = pd.to_datetime(monthly_data['Date']).dt.month
                
                # Group by month
                monthly_avg = monthly_data.groupby('Month')['PM2.5'].mean().reset_index()
                
                # Plot monthly averages
                fig_monthly = px.line(
                    monthly_avg,
                    x='Month',
                    y='PM2.5',
                    markers=True,
                    title=f'Monthly Average PM2.5 Levels for {selected_station}',
                    labels={'Month': 'Month', 'PM2.5': 'Average PM2.5 (μg/m³)'}
                )
                
                # Update x-axis to show month names
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                fig_monthly.update_xaxes(tickvals=list(range(1, 13)), ticktext=month_names)
                
                st.plotly_chart(fig_monthly, use_container_width=True)
                
                # Identify highest and lowest months
                if not monthly_avg.empty:
                    high_month = monthly_avg.loc[monthly_avg['PM2.5'].idxmax()]
                    low_month = monthly_avg.loc[monthly_avg['PM2.5'].idxmin()]
                    
                    st.markdown(f"""
                    **Seasonal Insights:**
                    
                    - Highest PM2.5 levels occur in **{month_names[int(high_month['Month'])-1]}** ({high_month['PM2.5']:.2f} μg/m³)
                    - Lowest PM2.5 levels occur in **{month_names[int(low_month['Month'])-1]}** ({low_month['PM2.5']:.2f} μg/m³)
                    
                    This seasonal pattern is typically influenced by:
                    - Weather conditions (temperature, rainfall, wind patterns)
                    - Human activities (heating, traffic patterns)
                    - Natural factors (dust, pollen)
                    """)
            else:
                st.warning("Please select a valid date range.")
        
        # Model Performance tab
        with tabs[2]:
            st.subheader(f"Model Performance Analysis for {selected_station}")
            
            # Train and evaluate models
            processed_data = preprocess_data(station_data)
            X_train, X_test, y_train, y_test = split_data(processed_data, test_size=0.2)
            models = train_models(X_train, y_train)
            evaluation_results = evaluate_models(models, X_test, y_test)
            
            # Create a dataframe for results
            results_df = pd.DataFrame({
                'Model': list(evaluation_results.keys()),
                'RMSE': [evaluation_results[model]['rmse'] for model in evaluation_results],
                'MAE': [evaluation_results[model]['mae'] for model in evaluation_results],
                'R²': [evaluation_results[model]['r2'] for model in evaluation_results]
            })
            
            # Display results table
            st.dataframe(results_df)
            
            # Plot RMSE comparison
            fig_rmse = px.bar(
                results_df,
                x='Model',
                y='RMSE',
                color='Model',
                title='RMSE Comparison (Lower is Better)',
                labels={'RMSE': 'Root Mean Square Error'}
            )
            st.plotly_chart(fig_rmse, use_container_width=True)
            
            # Plot R² comparison
            fig_r2 = px.bar(
                results_df,
                x='Model',
                y='R²',
                color='Model',
                title='R² Score Comparison (Higher is Better)',
                labels={'R²': 'R² Score'}
            )
            st.plotly_chart(fig_r2, use_container_width=True)
            
            # Find the best model
            best_model_rmse = results_df.loc[results_df['RMSE'].idxmin()]
            best_model_r2 = results_df.loc[results_df['R²'].idxmax()]
            
            st.markdown(f"""
            **Model Performance Insights:**
            
            - **Best model by RMSE:** {best_model_rmse['Model']} (RMSE: {best_model_rmse['RMSE']:.2f})
            - **Best model by R²:** {best_model_r2['Model']} (R²: {best_model_r2['R²']:.2f})
            
            **What these metrics mean:**
            
            - **RMSE (Root Mean Square Error):** Measures the average magnitude of errors in predictions. Lower values indicate better accuracy.
            - **MAE (Mean Absolute Error):** Measures the average absolute difference between predicted and actual values. Lower values are better.
            - **R² (R-squared):** Indicates how well the model explains the variance in the data. Values closer to 1 are better.
            
            The model "{best_model_rmse['Model']}" is recommended for PM2.5 predictions for {selected_station} due to its superior performance.
            """)
            
            # Display model accuracy visualization
            st.subheader("Prediction Accuracy Visualization")
            
            # Select a model to visualize
            model_to_vis = st.selectbox("Select Model", list(models.keys()))
            
            # Make predictions
            y_pred = models[model_to_vis].predict(X_test)
            
            # Create scatter plot of predicted vs actual
            pred_df = pd.DataFrame({
                'Actual': y_test,
                'Predicted': y_pred
            })
            
            fig_accuracy = px.scatter(
                pred_df,
                x='Actual',
                y='Predicted',
                title=f'Actual vs Predicted PM2.5 Values ({model_to_vis})',
                labels={'Actual': 'Actual PM2.5 (μg/m³)', 'Predicted': 'Predicted PM2.5 (μg/m³)'}
            )
            
            # Add perfect prediction line
            fig_accuracy.add_trace(
                go.Scatter(
                    x=[min(y_test), max(y_test)],
                    y=[min(y_test), max(y_test)],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                )
            )
            
            st.plotly_chart(fig_accuracy, use_container_width=True)
        
        # Forecasting Insights tab
        with tabs[3]:
            st.subheader(f"Forecasting Insights for {selected_station}")
            
            # Get weather forecast data
            forecast_data = get_weather_forecast(selected_station)
            
            if forecast_data:
                # Display forecast data
                st.write("Weather Forecast Data:")
                forecast_df = pd.DataFrame(forecast_data)
                st.dataframe(forecast_df)
                
                # Use pre-trained model for prediction
                models = load_pretrained_model()
                
                if models is None:
                    # Only train a new model if pre-trained model isn't available
                    processed_data = preprocess_data(station_data)
                    X_train, X_test, y_train, y_test = split_data(processed_data, test_size=0.2)
                    models = train_models(X_train, y_train)
                
                # Make predictions
                predictions = predict_pm25(models, forecast_data)
                
                # Create dataframe with all predictions
                pred_df = pd.DataFrame({
                    'Date': [item['Date'] for item in forecast_data]
                })
                
                for model_name, model_preds in predictions.items():
                    pred_df[model_name] = model_preds
                
                # Plot all model predictions
                fig_forecast = plot_forecast(pred_df)
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Find best model based on evaluation
                evaluation_results = evaluate_models(models, X_test, y_test)
                best_model = min(evaluation_results, key=lambda x: evaluation_results[x]['rmse'])
                
                # Explain forecast uncertainty
                st.markdown(f"""
                **Forecast Uncertainty Analysis:**
                
                The graph above shows predictions from different models. Notice how the predictions can vary:
                
                - **Model Agreement:** When models predict similar values, confidence is higher
                - **Model Divergence:** When models predict different values, uncertainty is higher
                
                Based on historical performance, the **{best_model}** model (RMSE: {evaluation_results[best_model]['rmse']:.2f}) 
                provides the most reliable forecasts for {selected_station}.
                
                **Factors affecting forecast accuracy:**
                
                1. **Weather Forecast Accuracy:** Our predictions rely on weather forecasts which have their own uncertainty
                2. **Historical Data Coverage:** More historical data generally improves prediction accuracy
                3. **Unusual Events:** Sudden pollution events or extreme weather can reduce accuracy
                4. **Seasonal Patterns:** Models may perform differently in different seasons
                """)
                
                # Allow saving forecast data
                if st.button("Save This Forecast for Future Reference"):
                    # Prepare data for saving
                    save_data = {
                        'station': selected_station,
                        'forecast_date': datetime.now().strftime("%Y-%m-%d"),
                        'weather_forecast': forecast_data,
                        'pm25_predictions': {model: list(preds) for model, preds in predictions.items()},
                        'best_model': best_model,
                        'model_rmse': evaluation_results[best_model]['rmse']
                    }
                    
                    # Save the data
                    saved_path = save_user_data('forecast', save_data)
                    st.success(f"Forecast data saved to {saved_path}")
            else:
                st.error("Unable to retrieve weather forecast data. Please check your internet connection or API key.")

# Model Training & Comparison page
elif page == "Model Training & Comparison":
    st.header("Model Training & Comparison")
    
    if data.empty:
        st.warning("No data available for model training.")
    else:
        # Station selection for model training
        station_for_model = st.selectbox("Select Station for Model Training", ["Fort William", "Bidhannagar"])
        
        # Filter data for the selected station
        station_data = data[data['Station'] == station_for_model].copy()
        
        # Add detailed explanation about model training process
        st.markdown("""
        ### Model Training Process
        
        This page allows you to train multiple machine learning models on historical data and compare their performance.
        
        **Steps in the model training process:**
        1. **Data Preprocessing:** Clean and prepare the data for training
        2. **Feature Engineering:** Extract relevant features from the data
        3. **Train-Test Split:** Divide data into training and testing sets
        4. **Model Training:** Train multiple models on the training data
        5. **Model Evaluation:** Compare models using various metrics
        
        **Models Used:**
        - **Model Tree:** Decision tree-based regressor for PM2.5 prediction
        - **NLR (Non-Linear Regression):** Advanced non-linear modeling
        - **Random Forest:** Ensemble learning method using multiple decision trees
        - **Logistic Regression:** Adapted for regression by binning PM2.5 values
        """)
        
        # Train test split ratio
        split_ratio = st.slider("Test Data Size (%)", 10, 40, 20) / 100
        
        # Add explanation about split ratio
        st.info("""
        **About Test Data Size:**
        - A smaller test size (10-15%) gives more data for training but less reliable evaluation
        - A larger test size (30-40%) gives more reliable evaluation but less data for training
        - 20% is often a good balance for most datasets
        """)
        
        if st.button("Train Models"):
            with st.spinner("Training models... This may take a moment."):
                # Preprocess the data
                processed_data = preprocess_data(station_data)
                
                # Split the data
                X_train, X_test, y_train, y_test = split_data(processed_data, test_size=split_ratio)
                
                # Train the models
                models = train_models(X_train, y_train)
                
                # Evaluate the models
                evaluation_results = evaluate_models(models, X_test, y_test)
                
                # Display evaluation results
                st.subheader("Model Performance Metrics")
                
                # Create a dataframe for displaying results
                results_df = pd.DataFrame({
                    'Model': list(evaluation_results.keys()),
                    'RMSE': [evaluation_results[model]['rmse'] for model in evaluation_results],
                    'MAE': [evaluation_results[model]['mae'] for model in evaluation_results],
                    'R²': [evaluation_results[model]['r2'] for model in evaluation_results]
                })
                
                st.dataframe(results_df)
                
                # Explain what the metrics mean
                with st.expander("Understanding Evaluation Metrics"):
                    st.markdown("""
                    **RMSE (Root Mean Square Error):** 
                    - Measures the average magnitude of prediction errors
                    - Gives higher weight to larger errors
                    - Lower values indicate better performance
                    - Unit is the same as PM2.5 (μg/m³)
                    
                    **MAE (Mean Absolute Error):** 
                    - Average absolute difference between predicted and actual values
                    - Treats all error sizes equally
                    - Lower values indicate better performance
                    - Unit is the same as PM2.5 (μg/m³)
                    
                    **R² Score (Coefficient of Determination):** 
                    - Indicates how well the model explains the variance in the data
                    - Ranges from 0 to 1 (can be negative for very poor models)
                    - Higher values indicate better performance
                    - A value of 1 means perfect prediction
                    """)
                
                # Bar chart for RMSE comparison
                st.subheader("Model RMSE Comparison")
                fig_rmse = px.bar(
                    results_df,
                    x='Model',
                    y='RMSE',
                    color='Model',
                    title=f'RMSE Comparison for {station_for_model}',
                    labels={'RMSE': 'Root Mean Square Error'}
                )
                st.plotly_chart(fig_rmse, use_container_width=True)
                
                # Bar chart for R² comparison
                st.subheader("Model R² Comparison")
                fig_r2 = px.bar(
                    results_df,
                    x='Model',
                    y='R²',
                    color='Model',
                    title=f'R² Comparison for {station_for_model}',
                    labels={'R²': 'R² Score (higher is better)'}
                )
                st.plotly_chart(fig_r2, use_container_width=True)
                
                # Model predictions vs actual values
                st.subheader("Predictions vs Actual Values")
                
                # Get model with lowest RMSE
                best_model_name = results_df.loc[results_df['RMSE'].idxmin(), 'Model']
                st.write(f"The model with the lowest RMSE is: **{best_model_name}**")
                
                # Create plot for best model
                fig_pred_actual = go.Figure()
                
                # Add actual values
                fig_pred_actual.add_trace(go.Scatter(
                    x=list(range(len(y_test))),
                    y=y_test,
                    mode='lines',
                    name='Actual Values',
                    line=dict(color='blue')
                ))
                
                # Add predicted values for the best model
                best_model = models[best_model_name]
                y_pred = best_model.predict(X_test)
                
                fig_pred_actual.add_trace(go.Scatter(
                    x=list(range(len(y_test))),
                    y=y_pred,
                    mode='lines',
                    name=f'{best_model_name} Predictions',
                    line=dict(color='red')
                ))
                
                fig_pred_actual.update_layout(
                    title=f'Actual vs Predicted PM2.5 Values ({best_model_name})',
                    xaxis_title='Sample Index',
                    yaxis_title='PM2.5 (μg/m³)',
                    legend=dict(x=0, y=1, traceorder='normal')
                )
                
                st.plotly_chart(fig_pred_actual, use_container_width=True)
                
                # Feature importance for Random Forest
                if 'Random Forest' in models:
                    st.subheader("Feature Importance (Random Forest)")
                    
                    # Get feature names and importance scores
                    feature_names = ['Temperature', 'Solar_Radiation', 'Wind_Speed', 'Relative_Humidity']
                    
                    # Check if feature importances are available and have the right length
                    if hasattr(models['Random Forest'], 'feature_importances_') and len(models['Random Forest'].feature_importances_) == len(feature_names):
                        importances = models['Random Forest'].feature_importances_
                        
                        # Create a dataframe for feature importance
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': importances
                        }).sort_values('Importance', ascending=False)
                    else:
                        # Create a dataframe with some default values if real importances aren't available
                        st.warning("Feature importance data is not available for this model.")
                        importance_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Importance': [0.3, 0.3, 0.2, 0.2]  # Sample values for visualization
                        })
                    
                    # Bar chart for feature importance
                    fig_importance = px.bar(
                        importance_df,
                        x='Feature',
                        y='Importance',
                        color='Feature',
                        title='Feature Importance (Random Forest)',
                        labels={'Importance': 'Importance Score'}
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Explain feature importance
                    st.markdown("""
                    **Understanding Feature Importance:**
                    
                    Feature importance shows how much each parameter contributes to the prediction. 
                    Higher values indicate that the parameter has a stronger influence on PM2.5 levels.
                    
                    This information is valuable for:
                    - Understanding which factors most affect air quality
                    - Prioritizing which weather parameters to monitor
                    - Improving future model versions by focusing on the most important features
                    """)
                
                # Option to save model results
                if st.button("Save Training Results"):
                    # Prepare data for saving
                    save_data = {
                        'station': station_for_model,
                        'training_date': datetime.now().strftime("%Y-%m-%d"),
                        'test_size': split_ratio,
                        'evaluation_results': {
                            model: {
                                'rmse': float(evaluation_results[model]['rmse']),
                                'mae': float(evaluation_results[model]['mae']),
                                'r2': float(evaluation_results[model]['r2'])
                            } for model in evaluation_results
                        },
                        'best_model': best_model_name
                    }
                    
                    # Add feature importance if available
                    if 'Random Forest' in models:
                        save_data['feature_importance'] = {
                            feature: float(importance) for feature, importance in zip(feature_names, importances)
                        }
                    
                    # Save the data
                    saved_path = save_user_data('model_training', save_data)
                    st.success(f"Model training results saved to {saved_path}")

# PM2.5 Predictions page
elif page == "PM2.5 Predictions":
    st.header("PM2.5 Predictions")
    
    if data.empty:
        st.warning("No data available for making predictions.")
    else:
        # Filter data for the selected station
        station_data = data[data['Station'] == selected_station].copy()
        
        # Current date for reference
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        st.subheader(f"PM2.5 Prediction for {selected_station}")
        
        # Option to use live weather data or manual input
        prediction_method = st.radio(
            "Select prediction method",
            ["Use live weather forecast data", "Manual parameter input"]
        )
        
        if prediction_method == "Use live weather forecast data":
            # Get weather forecast data from API
            with st.spinner("Fetching weather forecast data..."):
                forecast_data = get_weather_forecast(selected_station)
            
            if forecast_data:
                # Display the forecast data
                st.subheader("Weather Forecast Data")
                forecast_df = pd.DataFrame(forecast_data).round(2)
                st.dataframe(forecast_df)
                
                # Train models using historical data
                with st.spinner("Training prediction models..."):
                    # Preprocess the data
                    processed_data = preprocess_data(station_data)
                    
                    # Split the data for training
                    X_train, X_test, y_train, y_test = split_data(processed_data, test_size=0.2)
                    
                    # Train the models
                    models = train_models(X_train, y_train)
                    
                    # Evaluate to find best model
                    evaluation_results = evaluate_models(models, X_test, y_test)
                    best_model_name = min(evaluation_results, key=lambda x: evaluation_results[x]['rmse'])
                
                # Make predictions for the forecast days
                with st.spinner("Making predictions..."):
                    prediction_results = predict_pm25(models, forecast_data)
                
                # Create a dataframe for displaying predictions
                predictions_df = pd.DataFrame({
                    'Date': [item['Date'] for item in forecast_data]
                })
                
                for model_name, predictions in prediction_results.items():
                    predictions_df[model_name] = predictions
                
                # Display predictions from all models
                st.subheader("PM2.5 Predictions by Model")
                st.dataframe(predictions_df.round(2))
                
                # Plot predictions
                fig = plot_forecast(predictions_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display best model predictions with AQI categories
                st.subheader(f"Best Model Prediction ({best_model_name})")
                
                best_pred_df = pd.DataFrame({
                    'Date': predictions_df['Date'],
                    'PM2.5': predictions_df[best_model_name]
                })
                
                # Create colored output based on AQI levels
                for i, row in best_pred_df.iterrows():
                    aqi_category, aqi_class = get_aqi_category(row['PM2.5'])
                    st.markdown(f"""
                    <div style="display:flex; align-items:center; margin-bottom:10px;">
                        <div style="min-width:100px;"><strong>{row['Date']}</strong></div>
                        <div class="{aqi_class}" style="margin-left:15px; padding:5px 15px;">
                            PM2.5: {row['PM2.5']:.2f} μg/m³ - {aqi_category}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Save forecast data to historical storage
                if st.button("Save Forecast Data"):
                    # Prepare data to save
                    save_data = {
                        'station': selected_station,
                        'prediction_date': datetime.now().strftime("%Y-%m-%d"),
                        'method': 'api_forecast',
                        'weather_data': forecast_data,
                        'predictions': {
                            model: list(map(float, preds)) for model, preds in prediction_results.items()
                        },
                        'best_model': best_model_name,
                        'best_model_rmse': float(evaluation_results[best_model_name]['rmse'])
                    }
                    
                    # Save the data
                    saved_path = save_user_data('prediction', save_data)
                    st.success(f"Prediction data saved to {saved_path}")
                    
                    # Also save a CSV version for download
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    forecast_filename = f"pm25_forecast_{selected_station}_{timestamp}.csv"
                    forecast_path = os.path.join('data', 'predictions', forecast_filename)
                    
                    # Create a comprehensive dataframe
                    save_df = pd.DataFrame({
                        'Date': [item['Date'] for item in forecast_data],
                        'Temperature': [item['Temperature'] for item in forecast_data],
                        'Solar_Radiation': [item['Solar_Radiation'] for item in forecast_data],
                        'Wind_Speed': [item['Wind_Speed'] for item in forecast_data],
                        'Relative_Humidity': [item['Relative_Humidity'] for item in forecast_data]
                    })
                    
                    for model_name, predictions in prediction_results.items():
                        save_df[f'PM2.5_{model_name.replace(" ", "_")}'] = predictions
                    
                    save_df.to_csv(forecast_path, index=False)
                    st.success(f"Forecast data also saved to CSV at {forecast_path}")
                
                # Add a download link for the forecast data
                csv_data = predictions_df.to_csv(index=False)
                download_link = get_download_link(csv_data, f"pm25_forecast_{selected_station}.csv", "text/csv")
                st.markdown(download_link, unsafe_allow_html=True)
                
                # Explain prediction limitations
                with st.expander("Understanding Prediction Limitations"):
                    st.markdown("""
                    **Factors Affecting Prediction Accuracy:**
                    
                    1. **Weather Forecast Accuracy:** Our predictions rely on weather forecasts which have their own uncertainty
                    2. **Historical Data Coverage:** The predictions are based on historical patterns in our dataset
                    3. **Unusual Events:** Sudden pollution sources or extreme weather may not be captured in our model
                    4. **Model Limitations:** Each model has its own strengths and weaknesses
                    
                    For critical air quality decisions, always verify with official air quality monitoring stations.
                    """)
                
            else:
                st.error("Unable to retrieve weather forecast data. Please check your internet connection or API key.")
        
        else:  # Manual parameter input
            st.subheader("Enter Weather Parameters")
            
            # Create tabs for forecast days
            days = [0, 1, 2, 4, 7]  # Present, and 1, 2, 4, 7 days ahead
            day_labels = ["Today", "Tomorrow", "In 2 days", "In 4 days", "In 7 days"]
            
            # Create tabs for each day
            day_tabs = st.tabs(day_labels)
            
            manual_forecast_data = []
            
            for i, (tab, day) in enumerate(zip(day_tabs, days)):
                with tab:
                    forecast_date = (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d")
                    st.write(f"**Date: {forecast_date}**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        temp = st.slider(f"Temperature (°C)", min_value=-20.0, max_value=50.0, value=25.0, key=f"temp_{i}")
                        wind = st.slider(f"Wind Speed (m/s)", min_value=0.0, max_value=30.0, value=3.0, key=f"wind_{i}")
                    
                    with col2:
                        solar = st.slider(f"Solar Radiation (W/m²)", min_value=0.0, max_value=1500.0, value=250.0, key=f"solar_{i}")
                        humidity = st.slider(f"Relative Humidity (%)", min_value=0.0, max_value=100.0, value=65.0, key=f"humidity_{i}")
                    
                    manual_forecast_data.append({
                        'Date': forecast_date,
                        'Temperature': temp,
                        'Solar_Radiation': solar,
                        'Wind_Speed': wind,
                        'Relative_Humidity': humidity
                    })
            
            if st.button("Make Predictions"):
                # Train models using historical data
                with st.spinner("Training prediction models..."):
                    # Preprocess the data
                    processed_data = preprocess_data(station_data)
                    
                    # Split the data for training
                    X_train, X_test, y_train, y_test = split_data(processed_data, test_size=0.2)
                    
                    # Train the models
                    models = train_models(X_train, y_train)
                    
                    # Evaluate to find best model
                    evaluation_results = evaluate_models(models, X_test, y_test)
                    best_model_name = min(evaluation_results, key=lambda x: evaluation_results[x]['rmse'])
                
                # Make predictions for the forecast days
                with st.spinner("Making predictions..."):
                    prediction_results = predict_pm25(models, manual_forecast_data)
                
                # Create a dataframe for displaying predictions
                predictions_df = pd.DataFrame({
                    'Date': [item['Date'] for item in manual_forecast_data]
                })
                
                for model_name, predictions in prediction_results.items():
                    predictions_df[model_name] = predictions
                
                # Display predictions from all models
                st.subheader("PM2.5 Predictions by Model")
                st.dataframe(predictions_df.round(2))
                
                # Plot predictions
                fig = plot_forecast(predictions_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display best model predictions with AQI categories
                st.subheader(f"Best Model Prediction ({best_model_name})")
                
                best_pred_df = pd.DataFrame({
                    'Date': predictions_df['Date'],
                    'PM2.5': predictions_df[best_model_name]
                })
                
                # Create colored output based on AQI levels
                for i, row in best_pred_df.iterrows():
                    aqi_category, aqi_class = get_aqi_category(row['PM2.5'])
                    st.markdown(f"""
                    <div style="display:flex; align-items:center; margin-bottom:10px;">
                        <div style="min-width:100px;"><strong>{row['Date']}</strong></div>
                        <div class="{aqi_class}" style="margin-left:15px; padding:5px 15px;">
                            PM2.5: {row['PM2.5']:.2f} μg/m³ - {aqi_category}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Save manual forecast data to historical storage
                if st.button("Save Forecast Data"):
                    # Prepare data to save
                    save_data = {
                        'station': selected_station,
                        'prediction_date': datetime.now().strftime("%Y-%m-%d"),
                        'method': 'manual_input',
                        'weather_data': manual_forecast_data,
                        'predictions': {
                            model: list(map(float, preds)) for model, preds in prediction_results.items()
                        },
                        'best_model': best_model_name,
                        'best_model_rmse': float(evaluation_results[best_model_name]['rmse'])
                    }
                    
                    # Save the data
                    saved_path = save_user_data('prediction', save_data)
                    st.success(f"Prediction data saved to {saved_path}")
                    
                    # Also save a CSV version for download
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    forecast_filename = f"pm25_manual_forecast_{selected_station}_{timestamp}.csv"
                    forecast_path = os.path.join('data', 'predictions', forecast_filename)
                    
                    # Create a comprehensive dataframe
                    save_df = pd.DataFrame({
                        'Date': [item['Date'] for item in manual_forecast_data],
                        'Temperature': [item['Temperature'] for item in manual_forecast_data],
                        'Solar_Radiation': [item['Solar_Radiation'] for item in manual_forecast_data],
                        'Wind_Speed': [item['Wind_Speed'] for item in manual_forecast_data],
                        'Relative_Humidity': [item['Relative_Humidity'] for item in manual_forecast_data]
                    })
                    
                    for model_name, predictions in prediction_results.items():
                        save_df[f'PM2.5_{model_name.replace(" ", "_")}'] = predictions
                    
                    save_df.to_csv(forecast_path, index=False)
                    st.success(f"Forecast data also saved to CSV at {forecast_path}")
                
                # Add a download link for the forecast data
                csv_data = predictions_df.to_csv(index=False)
                download_link = get_download_link(csv_data, f"pm25_manual_forecast_{selected_station}.csv", "text/csv")
                st.markdown(download_link, unsafe_allow_html=True)

# Historical Analysis page
elif page == "Satellite Imagery":
    st.header("Satellite Imagery & Air Quality Maps")
    
    # Date selection for satellite imagery
    sat_date = st.date_input("Select Date for Satellite Data", datetime.now().date())
    sat_date_str = sat_date.strftime("%Y-%m-%d")
    
    # Create two columns for different satellite views
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"Air Quality Map for {selected_station}")
        
        # Get air quality map
        air_quality_map = get_air_quality_map(selected_station, sat_date_str)
        
        if air_quality_map:
            st.markdown(f'<img src="{air_quality_map}" style="width:100%">', unsafe_allow_html=True)
            st.markdown("""
            This air quality heatmap shows PM2.5 concentration levels as measured by satellite data. 
            Darker red areas indicate higher PM2.5 levels.
            """)
        else:
            st.error("Unable to retrieve air quality map. Please try another date.")
    
    with col2:
        st.subheader(f"Satellite Imagery for {selected_station}")
        
        # Get satellite imagery
        imagery = get_satellite_imagery(selected_station, sat_date_str)
        
        if imagery:
            st.image(imagery, caption=f"Satellite imagery of {selected_station}", use_column_width=True)
        else:
            st.error("Unable to retrieve satellite imagery. Please try another date.")
    
    # Historical satellite data analysis
    st.header("Historical Satellite Data Analysis")
    
    # Date range selection
    hist_col1, hist_col2 = st.columns(2)
    with hist_col1:
        sat_start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=30), key="sat_start")
    with hist_col2:
        sat_end_date = st.date_input("End Date", datetime.now().date(), key="sat_end")
    
    if sat_start_date <= sat_end_date:
        # Get historical satellite data
        sat_data = get_historical_satellite_data(
            selected_station, 
            sat_start_date.strftime("%Y-%m-%d"), 
            sat_end_date.strftime("%Y-%m-%d")
        )
        
        if not sat_data.empty:
            # Display the data
            st.subheader("Satellite-Derived PM2.5 Estimates")
            st.dataframe(sat_data)
            
            # Plot the data
            fig = px.line(
                sat_data,
                x='Date',
                y='Satellite_PM2.5',
                title=f'Satellite-Derived PM2.5 Estimates for {selected_station}',
                labels={'Satellite_PM2.5': 'PM2.5 (μg/m³)', 'Date': 'Date'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Compare with ground-based measurements if available
            st.subheader("Comparison with Ground Measurements")
            
            # Filter data for the selected station and date range
            ground_data = data[
                (data['Station'] == selected_station) &
                (pd.to_datetime(data['Date']).dt.date >= sat_start_date) &
                (pd.to_datetime(data['Date']).dt.date <= sat_end_date)
            ].copy()
            
            if not ground_data.empty:
                # Convert data for comparison
                ground_data['Date_Str'] = pd.to_datetime(ground_data['Date']).dt.strftime('%Y-%m-%d')
                
                # Merge satellite and ground data
                comparison_data = pd.merge(
                    sat_data, 
                    ground_data[['Date_Str', 'PM2.5']], 
                    left_on='Date', 
                    right_on='Date_Str',
                    how='inner'
                )
                
                if not comparison_data.empty:
                    # Display comparison data
                    comparison_data = comparison_data.rename(columns={'PM2.5': 'Ground_PM2.5'})
                    st.dataframe(comparison_data[['Date', 'Satellite_PM2.5', 'Ground_PM2.5']])
                    
                    # Plot comparison
                    fig_comp = go.Figure()
                    
                    # Add satellite data
                    fig_comp.add_trace(go.Scatter(
                        x=comparison_data['Date'],
                        y=comparison_data['Satellite_PM2.5'],
                        mode='lines+markers',
                        name='Satellite PM2.5'
                    ))
                    
                    # Add ground data
                    fig_comp.add_trace(go.Scatter(
                        x=comparison_data['Date'],
                        y=comparison_data['Ground_PM2.5'],
                        mode='lines+markers',
                        name='Ground PM2.5'
                    ))
                    
                    fig_comp.update_layout(
                        title=f'Satellite vs Ground PM2.5 Measurements for {selected_station}',
                        xaxis_title='Date',
                        yaxis_title='PM2.5 (μg/m³)',
                        legend=dict(x=0, y=1, traceorder='normal')
                    )
                    
                    st.plotly_chart(fig_comp, use_container_width=True)
                    
                    # Calculate correlation
                    corr = comparison_data['Satellite_PM2.5'].corr(comparison_data['Ground_PM2.5'])
                    
                    st.markdown(f"""
                    **Correlation between satellite and ground measurements: {corr:.4f}**
                    
                    This {'strong' if abs(corr) > 0.7 else 'moderate' if abs(corr) > 0.4 else 'weak'} correlation indicates that 
                    satellite measurements {'accurately reflect' if abs(corr) > 0.7 else 'somewhat reflect' if abs(corr) > 0.4 else 'may not accurately reflect'} 
                    ground-based PM2.5 levels.
                    """)
            else:
                st.info("No ground-based measurements available for the selected date range.")
        else:
            st.warning("No satellite data available for the selected date range.")
    else:
        st.warning("Please select a valid date range.")

elif page == "Historical Analysis":
    st.header("Historical Data Analysis")
    
    if data.empty:
        st.warning("No data available for analysis.")
    else:
        # Option to upload custom data for analysis
        st.subheader("Analyze Data")
        data_source = st.radio(
            "Select Data Source",
            ["Use Built-in Dataset", "Upload Your Own Dataset"]
        )
        
        analysis_data = None
        
        if data_source == "Upload Your Own Dataset":
            st.info("Upload a CSV file with historical PM2.5 and weather data. The file should include columns for Date, PM2.5, Temperature, Solar_Radiation, Wind_Speed, and Relative_Humidity.")
            
            uploaded_file = st.file_uploader("Upload CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    # Load the uploaded data
                    analysis_data = pd.read_csv(uploaded_file)
                    
                    # Check if required columns exist
                    required_columns = ['Date', 'PM2.5']
                    if not all(col in analysis_data.columns for col in required_columns):
                        st.error("Uploaded file is missing required columns. Please ensure it has at least 'Date' and 'PM2.5' columns.")
                        analysis_data = None
                    else:
                        # Convert date to datetime if it's not already
                        analysis_data['Date'] = pd.to_datetime(analysis_data['Date'])
                        
                        # Display data preview
                        st.subheader("Data Preview")
                        st.dataframe(analysis_data.head())
                        
                        # Add station column if missing
                        if 'Station' not in analysis_data.columns:
                            analysis_data['Station'] = "Uploaded Data"
                except Exception as e:
                    st.error(f"Error loading file: {e}")
                    analysis_data = None
        else:
            # Date range selection
            st.subheader("Select Data Range")
            ha_col1, ha_col2 = st.columns(2)
            
            # Fixed date range from 2021-01-01 to 2025-01-15 as requested
            fixed_start_date = pd.to_datetime('2021-01-01').date()
            fixed_end_date = pd.to_datetime('2025-01-15').date()
            
            # Set the date range controls using these fixed dates
            with ha_col1:
                ha_start_date = st.date_input("Start Date", fixed_start_date, 
                                            min_value=fixed_start_date, 
                                            max_value=fixed_end_date, 
                                            key="ha_start")
            with ha_col2:
                ha_end_date = st.date_input("End Date", fixed_end_date, 
                                          min_value=fixed_start_date, 
                                          max_value=fixed_end_date, 
                                          key="ha_end")
            
            # Station selection
            ha_station_filter = st.multiselect(
                "Select Stations",
                ["Fort William", "Bidhannagar"],
                default=[selected_station],
                key="ha_stations"
            )
            
            # Check for valid selections
            if ha_start_date > ha_end_date:
                st.warning("⚠️ Start date must be before or equal to end date. Please adjust your selection.")
                analysis_data = None
            elif not ha_station_filter:
                st.warning("⚠️ Please select at least one station.")
                analysis_data = None
            else:
                # Filter data based on date range and stations
                analysis_data = data[
                    (pd.to_datetime(data['Date']).dt.date >= ha_start_date) &
                    (pd.to_datetime(data['Date']).dt.date <= ha_end_date) &
                    (data['Station'].isin(ha_station_filter))
                ].copy()
                
                # Check if filtered data is empty
                if analysis_data.empty:
                    st.warning(f"No data available for the selected date range and stations. Try expanding your selection.")
                    analysis_data = None
        
        if analysis_data is not None and not analysis_data.empty:
            # Create tabs for different analysis options
            analysis_tabs = st.tabs([
                "Overview Statistics", 
                "Time Series Analysis", 
                "Correlation Analysis", 
                "Seasonal Patterns",
                "Custom Analysis"
            ])
            
            # Overview Statistics
            with analysis_tabs[0]:
                st.subheader("Overview Statistics")
                
                # Calculate basic statistics
                stats = analysis_data.groupby('Station')['PM2.5'].agg([
                    'count', 'mean', 'std', 'min', 
                    lambda x: np.percentile(x, 25), 
                    'median', 
                    lambda x: np.percentile(x, 75),
                    'max'
                ]).reset_index()
                
                stats.columns = ['Station', 'Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max']
                stats = stats.round(2)
                
                # Display statistics
                st.dataframe(stats)
                
                # Create visualizations
                st.subheader("Distribution of PM2.5 Values")
                
                # Histogram
                fig_hist = px.histogram(
                    analysis_data,
                    x='PM2.5',
                    color='Station',
                    nbins=30,
                    opacity=0.7,
                    marginal="box",
                    title="PM2.5 Distribution",
                    labels={'PM2.5': 'PM2.5 (μg/m³)'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Box plot
                fig_box = px.box(
                    analysis_data,
                    x='Station',
                    y='PM2.5',
                    color='Station',
                    title="PM2.5 Box Plot by Station",
                    labels={'PM2.5': 'PM2.5 (μg/m³)'}
                )
                st.plotly_chart(fig_box, use_container_width=True)
                
                # Calculate AQI distribution
                st.subheader("AQI Category Distribution")
                
                # Add AQI category column
                analysis_data['AQI_Category'] = analysis_data['PM2.5'].apply(lambda x: get_aqi_category(x)[0])
                
                # Count by station and AQI category
                aqi_counts = analysis_data.groupby(['Station', 'AQI_Category']).size().reset_index(name='Count')
                
                # Plot AQI distribution
                fig_aqi = px.bar(
                    aqi_counts,
                    x='AQI_Category',
                    y='Count',
                    color='Station',
                    barmode='group',
                    title="AQI Category Distribution",
                    labels={'AQI_Category': 'AQI Category', 'Count': 'Number of Days'}
                )
                
                # Set category order
                fig_aqi.update_xaxes(categoryorder='array', 
                                  categoryarray=['Good', 'Moderate', 'Unhealthy for Sensitive', 'Unhealthy', 'Very Unhealthy', 'Hazardous'])
                
                st.plotly_chart(fig_aqi, use_container_width=True)
            
            # Time Series Analysis
            with analysis_tabs[1]:
                st.subheader("Time Series Analysis")
                
                # Plot time series
                fig_ts = plot_historical_data(analysis_data)
                st.plotly_chart(fig_ts, use_container_width=True)
                
                # Add option to analyze trends
                if st.checkbox("Analyze Trends"):
                    st.subheader("Trend Analysis")
                    
                    # Select station for trend analysis
                    trend_station = st.selectbox("Select Station for Trend Analysis", analysis_data['Station'].unique())
                    
                    # Filter data for selected station
                    station_ts_data = analysis_data[analysis_data['Station'] == trend_station].copy()
                    
                    # Sort by date
                    station_ts_data = station_ts_data.sort_values('Date')
                    station_ts_data['day_index'] = range(len(station_ts_data))
                    
                    # Calculate trend line
                    z = np.polyfit(station_ts_data['day_index'], station_ts_data['PM2.5'], 1)
                    trend = np.poly1d(z)
                    
                    # Plot with trend line
                    fig_trend = px.scatter(
                        station_ts_data,
                        x='Date',
                        y='PM2.5',
                        title=f'PM2.5 Trend for {trend_station}',
                        trendline='ols',
                        labels={'PM2.5': 'PM2.5 (μg/m³)', 'Date': 'Date'}
                    )
                    
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Explain trend
                    slope = z[0]
                    if abs(slope) < 0.01:
                        trend_direction = "stable"
                    elif slope > 0:
                        trend_direction = "increasing"
                    else:
                        trend_direction = "decreasing"
                    
                    st.markdown(f"""
                    **Trend Analysis Results:**
                    
                    The PM2.5 levels in {trend_station} show a **{trend_direction}** trend over this period.
                    
                    - Slope: {slope:.4f} μg/m³ per day
                    - This means PM2.5 levels are {"increasing" if slope > 0 else "decreasing"} by approximately {abs(slope * 30):.2f} μg/m³ per month on average.
                    """)
                    
                    # Moving average for smoothing
                    st.subheader("Moving Average Analysis")
                    window_size = st.slider("Moving Average Window (days)", 1, 30, 7)
                    
                    # Calculate moving average
                    station_ts_data['PM2.5_MA'] = station_ts_data['PM2.5'].rolling(window=window_size).mean()
                    
                    # Plot original and moving average
                    fig_ma = go.Figure()
                    
                    # Add original data
                    fig_ma.add_trace(go.Scatter(
                        x=station_ts_data['Date'],
                        y=station_ts_data['PM2.5'],
                        mode='lines',
                        name='Original',
                        line=dict(color='blue', width=1)
                    ))
                    
                    # Add moving average
                    fig_ma.add_trace(go.Scatter(
                        x=station_ts_data['Date'],
                        y=station_ts_data['PM2.5_MA'],
                        mode='lines',
                        name=f'{window_size}-Day Moving Average',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig_ma.update_layout(
                        title=f'PM2.5 with {window_size}-Day Moving Average for {trend_station}',
                        xaxis_title='Date',
                        yaxis_title='PM2.5 (μg/m³)',
                        legend=dict(x=0, y=1, traceorder='normal')
                    )
                    
                    st.plotly_chart(fig_ma, use_container_width=True)
            
            # Correlation Analysis
            with analysis_tabs[2]:
                st.subheader("Correlation Analysis")
                
                # Select station for correlation analysis
                corr_station = st.selectbox("Select Station", analysis_data['Station'].unique())
                
                # Filter data for selected station
                station_corr_data = analysis_data[analysis_data['Station'] == corr_station].copy()
                
                # Correlation matrix
                st.subheader("Parameter Correlation Matrix")
                fig_corr = plot_correlation_matrix(station_corr_data)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Detailed parameter analysis
                st.subheader("Parameter Comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    param_x = st.selectbox("Select X Parameter", ["Temperature", "Solar_Radiation", "Wind_Speed", "Relative_Humidity"])
                
                with col2:
                    param_y = st.selectbox("Select Y Parameter", ["PM2.5", "Temperature", "Solar_Radiation", "Wind_Speed", "Relative_Humidity"], index=0)
                
                if param_x != param_y and param_x in station_corr_data.columns and param_y in station_corr_data.columns:
                    # Create scatter plot
                    fig_scatter = plot_parameter_comparison(station_corr_data, param_x, param_y)
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Calculate correlation
                    corr = station_corr_data[param_x].corr(station_corr_data[param_y])
                    
                    # Interpretation based on correlation strength
                    if abs(corr) > 0.7:
                        strength = "strong"
                    elif abs(corr) > 0.4:
                        strength = "moderate"
                    else:
                        strength = "weak"
                    
                    direction = "positive" if corr > 0 else "negative"
                    
                    st.markdown(f"""
                    **Correlation Analysis:**
                    
                    The correlation coefficient between {param_x} and {param_y} is **{corr:.4f}**.
                    
                    This represents a **{strength} {direction}** correlation, which means:
                    
                    - {"As " + param_x + " increases, " + param_y + " tends to increase as well." if corr > 0 else "As " + param_x + " increases, " + param_y + " tends to decrease."}
                    - This parameter {"has a significant impact" if abs(corr) > 0.4 else "has limited impact"} on {param_y} in {corr_station}.
                    """)
                
                else:
                    st.warning("Please select different parameters for X and Y axes.")
            
            # Seasonal Patterns
            with analysis_tabs[3]:
                st.subheader("Seasonal Patterns")
                
                # Add month column for seasonal analysis
                seasonal_data = analysis_data.copy()
                seasonal_data['Month'] = pd.to_datetime(seasonal_data['Date']).dt.month
                seasonal_data['Year'] = pd.to_datetime(seasonal_data['Date']).dt.year
                
                # Monthly patterns
                st.subheader("Monthly Patterns")
                
                # Group by month and station
                monthly_avg = seasonal_data.groupby(['Month', 'Station'])['PM2.5'].mean().reset_index()
                
                # Create line chart
                fig_monthly = px.line(
                    monthly_avg,
                    x='Month',
                    y='PM2.5',
                    color='Station',
                    markers=True,
                    title='Monthly Average PM2.5 Levels',
                    labels={'Month': 'Month', 'PM2.5': 'Average PM2.5 (μg/m³)'}
                )
                
                # Update x-axis to show month names
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                fig_monthly.update_xaxes(tickvals=list(range(1, 13)), ticktext=month_names)
                
                st.plotly_chart(fig_monthly, use_container_width=True)
                
                # Yearly comparison if multiple years
                if len(seasonal_data['Year'].unique()) > 1:
                    st.subheader("Yearly Comparison")
                    
                    # Select station for yearly comparison
                    yearly_station = st.selectbox("Select Station for Yearly Comparison", seasonal_data['Station'].unique())
                    
                    # Filter data for selected station
                    yearly_data = seasonal_data[seasonal_data['Station'] == yearly_station].copy()
                    
                    # Group by year and month
                    yearly_monthly = yearly_data.groupby(['Year', 'Month'])['PM2.5'].mean().reset_index()
                    
                    # Create line chart
                    fig_yearly = px.line(
                        yearly_monthly,
                        x='Month',
                        y='PM2.5',
                        color='Year',
                        markers=True,
                        title=f'Monthly PM2.5 Levels by Year for {yearly_station}',
                        labels={'Month': 'Month', 'PM2.5': 'Average PM2.5 (μg/m³)'}
                    )
                    
                    # Update x-axis to show month names
                    fig_yearly.update_xaxes(tickvals=list(range(1, 13)), ticktext=month_names)
                    
                    st.plotly_chart(fig_yearly, use_container_width=True)
                
                # Day of week patterns
                st.subheader("Day of Week Patterns")
                
                # Add day of week column
                seasonal_data['DayOfWeek'] = pd.to_datetime(seasonal_data['Date']).dt.dayofweek
                
                # Group by day of week and station
                dow_avg = seasonal_data.groupby(['DayOfWeek', 'Station'])['PM2.5'].mean().reset_index()
                
                # Create bar chart
                fig_dow = px.bar(
                    dow_avg,
                    x='DayOfWeek',
                    y='PM2.5',
                    color='Station',
                    barmode='group',
                    title='Average PM2.5 Levels by Day of Week',
                    labels={'DayOfWeek': 'Day of Week', 'PM2.5': 'Average PM2.5 (μg/m³)'}
                )
                
                # Update x-axis to show day names
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                fig_dow.update_xaxes(tickvals=list(range(7)), ticktext=day_names)
                
                st.plotly_chart(fig_dow, use_container_width=True)
                
                # Explain patterns
                st.markdown("""
                **Understanding Seasonal Patterns:**
                
                The seasonal patterns in PM2.5 levels can be influenced by various factors:
                
                1. **Weather Conditions:**
                   - Temperature inversions in winter can trap pollutants
                   - Rainfall in monsoon seasons can reduce particulate matter
                   
                2. **Human Activities:**
                   - Heating needs in winter can increase emissions
                   - Agricultural burning in specific seasons
                   - Different traffic patterns on weekdays vs. weekends
                   
                3. **Natural Factors:**
                   - Dust storms in dry seasons
                   - Pollen and other biological particles in spring/summer
                   
                Understanding these patterns helps in planning air quality management strategies and personal protection measures.
                """)
            
            # Custom Analysis
            with analysis_tabs[4]:
                st.subheader("Custom Analysis")
                
                # Let user select analysis type
                analysis_type = st.selectbox(
                    "Select Analysis Type",
                    ["PM2.5 Threshold Analysis", "Extreme Value Analysis", "Rolling Statistics", "Custom Parameter Calculation"]
                )
                
                if analysis_type == "PM2.5 Threshold Analysis":
                    st.subheader("PM2.5 Threshold Analysis")
                    
                    # Set threshold
                    threshold = st.slider("Select PM2.5 Threshold (μg/m³)", 0, 300, 50)
                    
                    # Calculate days above threshold
                    analysis_data['Above_Threshold'] = analysis_data['PM2.5'] > threshold
                    
                    # Group by station
                    threshold_summary = analysis_data.groupby('Station')['Above_Threshold'].agg(['count', 'sum']).reset_index()
                    threshold_summary['Percentage'] = (threshold_summary['sum'] / threshold_summary['count'] * 100).round(2)
                    threshold_summary.columns = ['Station', 'Total Days', 'Days Above Threshold', 'Percentage (%)']
                    
                    # Display summary
                    st.dataframe(threshold_summary)
                    
                    # Create visualization
                    fig_threshold = px.bar(
                        threshold_summary,
                        x='Station',
                        y='Percentage (%)',
                        color='Station',
                        title=f'Percentage of Days with PM2.5 > {threshold} μg/m³',
                        labels={'Percentage (%)': 'Percentage of Days'}
                    )
                    
                    st.plotly_chart(fig_threshold, use_container_width=True)
                
                elif analysis_type == "Extreme Value Analysis":
                    st.subheader("Extreme Value Analysis")
                    
                    # Select station
                    extreme_station = st.selectbox("Select Station for Extreme Value Analysis", analysis_data['Station'].unique())
                    
                    # Filter data
                    extreme_data = analysis_data[analysis_data['Station'] == extreme_station].copy()
                    
                    # Select percentile
                    percentile = st.slider("Select Percentile for Analysis", 90, 99, 95)
                    
                    # Calculate threshold
                    threshold = np.percentile(extreme_data['PM2.5'], percentile)
                    
                    # Filter extreme values
                    extreme_values = extreme_data[extreme_data['PM2.5'] >= threshold].copy()
                    
                    st.write(f"**{percentile}th Percentile PM2.5 Value:** {threshold:.2f} μg/m³")
                    st.write(f"**Number of Extreme Days:** {len(extreme_values)} days")
                    
                    # Display extreme days
                    if not extreme_values.empty:
                        st.subheader("Extreme Value Days")
                        st.dataframe(extreme_values[['Date', 'PM2.5', 'Temperature', 'Wind_Speed', 'Relative_Humidity']].sort_values('PM2.5', ascending=False))
                        
                        # Create scatter plot of extreme values
                        fig_extreme = px.scatter(
                            extreme_values,
                            x='Date',
                            y='PM2.5',
                            size='PM2.5',
                            color='PM2.5',
                            title=f'Extreme PM2.5 Values (>{threshold:.2f} μg/m³) for {extreme_station}',
                            labels={'PM2.5': 'PM2.5 (μg/m³)'}
                        )
                        
                        st.plotly_chart(fig_extreme, use_container_width=True)
                        
                        # Analyze weather conditions during extreme days
                        st.subheader("Weather Conditions During Extreme Days")
                        
                        # Calculate average conditions
                        avg_temp = extreme_values['Temperature'].mean()
                        avg_wind = extreme_values['Wind_Speed'].mean()
                        avg_humidity = extreme_values['Relative_Humidity'].mean()
                        
                        # Calculate normal conditions
                        normal_temp = extreme_data['Temperature'].mean()
                        normal_wind = extreme_data['Wind_Speed'].mean()
                        normal_humidity = extreme_data['Relative_Humidity'].mean()
                        
                        # Create comparison dataframe
                        weather_comp = pd.DataFrame({
                            'Condition': ['Temperature (°C)', 'Wind Speed (m/s)', 'Relative Humidity (%)'],
                            'During Extreme Days': [avg_temp, avg_wind, avg_humidity],
                            'Normal Average': [normal_temp, normal_wind, normal_humidity],
                            'Difference': [avg_temp - normal_temp, avg_wind - normal_wind, avg_humidity - normal_humidity]
                        })
                        
                        st.dataframe(weather_comp.round(2))
                        
                        # Create bar chart for comparison
                        fig_weather = go.Figure()
                        
                        # Add bars for extreme days
                        fig_weather.add_trace(go.Bar(
                            x=weather_comp['Condition'],
                            y=weather_comp['During Extreme Days'],
                            name='During Extreme Days',
                            marker_color='red'
                        ))
                        
                        # Add bars for normal days
                        fig_weather.add_trace(go.Bar(
                            x=weather_comp['Condition'],
                            y=weather_comp['Normal Average'],
                            name='Normal Average',
                            marker_color='blue'
                        ))
                        
                        fig_weather.update_layout(
                            title='Weather Conditions Comparison',
                            xaxis_title='Weather Parameter',
                            yaxis_title='Value',
                            barmode='group'
                        )
                        
                        st.plotly_chart(fig_weather, use_container_width=True)
                        
                        # Provide interpretation
                        st.markdown(f"""
                        **Weather Analysis Findings:**
                        
                        During days with extreme PM2.5 levels ({percentile}th percentile and above):
                        
                        - Temperature is **{weather_comp.iloc[0]['Difference']:.2f}°C {"higher" if weather_comp.iloc[0]['Difference'] > 0 else "lower"}** than normal
                        - Wind speed is **{abs(weather_comp.iloc[1]['Difference']):.2f} m/s {"higher" if weather_comp.iloc[1]['Difference'] > 0 else "lower"}** than normal
                        - Relative humidity is **{abs(weather_comp.iloc[2]['Difference']):.2f}% {"higher" if weather_comp.iloc[2]['Difference'] > 0 else "lower"}** than normal
                        
                        This suggests that extreme PM2.5 values in {extreme_station} tend to occur during periods of 
                        {"higher" if weather_comp.iloc[0]['Difference'] > 0 else "lower"} temperature, 
                        {"higher" if weather_comp.iloc[1]['Difference'] > 0 else "lower"} wind speeds, and 
                        {"higher" if weather_comp.iloc[2]['Difference'] > 0 else "lower"} humidity.
                        """)
                
                elif analysis_type == "Rolling Statistics":
                    st.subheader("Rolling Statistics Analysis")
                    
                    # Select station
                    roll_station = st.selectbox("Select Station for Rolling Statistics", analysis_data['Station'].unique())
                    
                    # Filter data
                    roll_data = analysis_data[analysis_data['Station'] == roll_station].copy()
                    
                    # Sort by date
                    roll_data = roll_data.sort_values('Date')
                    
                    # Select window size
                    window_size = st.slider("Select Rolling Window Size (days)", 1, 30, 7)
                    
                    # Calculate rolling statistics
                    roll_data['Rolling_Mean'] = roll_data['PM2.5'].rolling(window=window_size).mean()
                    roll_data['Rolling_Std'] = roll_data['PM2.5'].rolling(window=window_size).std()
                    
                    # Create upper and lower bounds (mean ± 2*std)
                    roll_data['Upper_Bound'] = roll_data['Rolling_Mean'] + 2 * roll_data['Rolling_Std']
                    roll_data['Lower_Bound'] = roll_data['Rolling_Mean'] - 2 * roll_data['Rolling_Std']
                    roll_data['Lower_Bound'] = roll_data['Lower_Bound'].clip(lower=0)  # PM2.5 can't be negative
                    
                    # Create plot
                    fig_roll = go.Figure()
                    
                    # Add actual PM2.5 values
                    fig_roll.add_trace(go.Scatter(
                        x=roll_data['Date'],
                        y=roll_data['PM2.5'],
                        mode='lines',
                        name='Actual PM2.5',
                        line=dict(color='blue')
                    ))
                    
                    # Add rolling mean
                    fig_roll.add_trace(go.Scatter(
                        x=roll_data['Date'],
                        y=roll_data['Rolling_Mean'],
                        mode='lines',
                        name=f'{window_size}-Day Rolling Mean',
                        line=dict(color='red', width=2)
                    ))
                    
                    # Add bounds as a filled area
                    fig_roll.add_trace(go.Scatter(
                        x=roll_data['Date'],
                        y=roll_data['Upper_Bound'],
                        mode='lines',
                        name='Upper Bound (Mean + 2σ)',
                        line=dict(width=0),
                        showlegend=True
                    ))
                    
                    fig_roll.add_trace(go.Scatter(
                        x=roll_data['Date'],
                        y=roll_data['Lower_Bound'],
                        mode='lines',
                        name='Lower Bound (Mean - 2σ)',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor='rgba(255, 0, 0, 0.1)',
                        showlegend=True
                    ))
                    
                    fig_roll.update_layout(
                        title=f'{window_size}-Day Rolling Statistics for {roll_station}',
                        xaxis_title='Date',
                        yaxis_title='PM2.5 (μg/m³)',
                        legend=dict(x=0, y=1, traceorder='normal')
                    )
                    
                    st.plotly_chart(fig_roll, use_container_width=True)
                    
                    # Calculate volatility (average of rolling std)
                    volatility = roll_data['Rolling_Std'].mean()
                    
                    st.markdown(f"""
                    **Rolling Statistics Insights:**
                    
                    - The red line shows the {window_size}-day rolling average PM2.5 level
                    - The shaded area represents the normal range (mean ± 2 standard deviations)
                    - Points outside this range are considered unusual or extreme
                    
                    **PM2.5 Volatility:** {volatility:.2f} μg/m³
                    
                    This represents the average day-to-day variation in PM2.5 levels over the selected period.
                    {"This indicates high volatility in air quality." if volatility > 15 else "This indicates relatively stable air quality." if volatility < 5 else "This indicates moderate volatility in air quality."}
                    """)
                
                elif analysis_type == "Custom Parameter Calculation":
                    st.subheader("Custom Parameter Calculation")
                    
                    st.markdown("""
                    Create a custom air quality index or parameter based on available data.
                    """)
                    
                    # Select station
                    custom_station = st.selectbox("Select Station", analysis_data['Station'].unique())
                    
                    # Filter data
                    custom_data = analysis_data[analysis_data['Station'] == custom_station].copy()
                    
                    # Select parameters to include
                    params = st.multiselect(
                        "Select Parameters to Include",
                        ["PM2.5", "Temperature", "Solar_Radiation", "Wind_Speed", "Relative_Humidity"],
                        default=["PM2.5", "Wind_Speed"]
                    )
                    
                    if len(params) > 0:
                        # Select calculation method
                        calc_method = st.radio(
                            "Select Calculation Method",
                            ["Weighted Average", "Ratio", "Product", "Custom Formula"]
                        )
                        
                        if calc_method == "Weighted Average" and len(params) > 1:
                            # Set weights
                            st.subheader("Set Weights for Parameters")
                            
                            weights = {}
                            cols = st.columns(len(params))
                            
                            for i, param in enumerate(params):
                                with cols[i]:
                                    weights[param] = st.slider(f"Weight for {param}", 0.0, 1.0, 0.5, 0.1)
                            
                            # Normalize weights
                            total_weight = sum(weights.values())
                            norm_weights = {param: weight/total_weight for param, weight in weights.items()}
                            
                            # Calculate weighted average
                            custom_data['Custom_Parameter'] = 0
                            
                            for param in params:
                                # Normalize parameter to 0-100 scale
                                max_val = custom_data[param].max()
                                min_val = custom_data[param].min()
                                if max_val != min_val:
                                    normalized = (custom_data[param] - min_val) / (max_val - min_val) * 100
                                else:
                                    normalized = custom_data[param] * 0
                                
                                custom_data['Custom_Parameter'] += normalized * norm_weights[param]
                            
                            st.write("**Normalized Weights:**")
                            for param, weight in norm_weights.items():
                                st.write(f"- {param}: {weight:.2f}")
                        
                        elif calc_method == "Ratio" and len(params) == 2:
                            # Calculate ratio
                            custom_data['Custom_Parameter'] = custom_data[params[0]] / custom_data[params[1]].replace(0, np.nan)
                            custom_data['Custom_Parameter'].fillna(0, inplace=True)
                            
                            st.write(f"**Calculated Ratio:** {params[0]} / {params[1]}")
                        
                        elif calc_method == "Product":
                            # Calculate product
                            custom_data['Custom_Parameter'] = 1
                            for param in params:
                                custom_data['Custom_Parameter'] *= custom_data[param]
                            
                            st.write(f"**Calculated Product:** {' × '.join(params)}")
                        
                        elif calc_method == "Custom Formula":
                            # Let user enter a formula
                            formula = st.text_input("Enter Custom Formula", value="PM2.5 / (Wind_Speed + 0.1)")
                            
                            try:
                                # Create a copy of the data frame with just the required columns
                                calc_df = custom_data[params].copy()
                                
                                # Evaluate the formula
                                custom_data['Custom_Parameter'] = eval(formula, {"__builtins__": {}}, calc_df)
                                
                                st.write(f"**Custom Formula:** {formula}")
                            except Exception as e:
                                st.error(f"Error evaluating formula: {e}")
                                custom_data['Custom_Parameter'] = np.nan
                        
                        # Plot the custom parameter
                        if not custom_data['Custom_Parameter'].isna().all():
                            st.subheader("Custom Parameter Analysis")
                            
                            # Create time series plot
                            fig_custom = px.line(
                                custom_data,
                                x='Date',
                                y='Custom_Parameter',
                                title=f'Custom Parameter for {custom_station}',
                                labels={'Custom_Parameter': 'Custom Parameter Value'}
                            )
                            
                            st.plotly_chart(fig_custom, use_container_width=True)
                            
                            # Calculate statistics
                            mean_val = custom_data['Custom_Parameter'].mean()
                            max_val = custom_data['Custom_Parameter'].max()
                            min_val = custom_data['Custom_Parameter'].min()
                            std_val = custom_data['Custom_Parameter'].std()
                            
                            st.markdown(f"""
                            **Custom Parameter Statistics:**
                            
                            - **Mean:** {mean_val:.2f}
                            - **Maximum:** {max_val:.2f}
                            - **Minimum:** {min_val:.2f}
                            - **Standard Deviation:** {std_val:.2f}
                            """)
                            
                            # Correlation with PM2.5
                            if 'PM2.5' in custom_data.columns and 'Custom_Parameter' in custom_data.columns:
                                corr = custom_data['PM2.5'].corr(custom_data['Custom_Parameter'])
                                
                                st.markdown(f"""
                                **Correlation with PM2.5:** {corr:.4f}
                                
                                This {"strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"} correlation indicates that your custom parameter is {"closely" if abs(corr) > 0.7 else "somewhat" if abs(corr) > 0.4 else "not very"} related to PM2.5 levels.
                                """)
                                
                    else:
                        st.warning("Please select at least one parameter.")
            
            # Option to save analysis results
            st.subheader("Save Analysis Results")
            
            if st.button("Save Analysis Results"):
                # Prepare data for saving
                save_data = {
                    'analysis_date': datetime.now().strftime("%Y-%m-%d"),
                    'stations': list(analysis_data['Station'].unique()),
                    'date_range': [
                        min(pd.to_datetime(analysis_data['Date'])).strftime("%Y-%m-%d"),
                        max(pd.to_datetime(analysis_data['Date'])).strftime("%Y-%m-%d")
                    ],
                    'summary_statistics': {
                        station: {
                            'mean': float(analysis_data[analysis_data['Station'] == station]['PM2.5'].mean()),
                            'max': float(analysis_data[analysis_data['Station'] == station]['PM2.5'].max()),
                            'min': float(analysis_data[analysis_data['Station'] == station]['PM2.5'].min()),
                            'std': float(analysis_data[analysis_data['Station'] == station]['PM2.5'].std())
                        } for station in analysis_data['Station'].unique()
                    }
                }
                
                # Save the data
                saved_path = save_user_data('analysis', save_data)
                st.success(f"Analysis results saved to {saved_path}")
        
        else:
            st.warning("Please select a valid date range and at least one station.")

# Data Download page
elif page == "Data Download":
    st.header("Download Air Quality Data")
    
    if data.empty:
        st.warning("No data available for download.")
    else:
        # Tabs for different data types
        download_tabs = st.tabs(["Historical Data", "Prediction Data", "User Analysis Data"])
        
        # Historical Data tab
        with download_tabs[0]:
            st.subheader("Download Historical Data")
            
            st.write("Select the date range and stations for data download.")
            
            # Date range selection
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                dl_start_date = st.date_input("Start Date", min(pd.to_datetime(data['Date'])).date())
            with dl_col2:
                dl_end_date = st.date_input("End Date", max(pd.to_datetime(data['Date'])).date())
            
            # Station selection
            dl_station_filter = st.multiselect(
                "Select Stations",
                ["Fort William", "Bidhannagar"],
                default=[selected_station],
                key="dl_stations"
            )
            
            # Data content selection
            data_content = st.radio(
                "Select Data Content",
                ["Raw Data", "Daily Averages", "Monthly Averages"]
            )
            
            # File format selection
            file_format = st.selectbox(
                "Select File Format",
                ["CSV", "Excel", "JSON"]
            )
            
            if dl_start_date <= dl_end_date and dl_station_filter:
                # Filter data based on date range and stations
                dl_filtered_data = data[
                    (pd.to_datetime(data['Date']).dt.date >= dl_start_date) &
                    (pd.to_datetime(data['Date']).dt.date <= dl_end_date) &
                    (data['Station'].isin(dl_station_filter))
                ].copy()
                
                # Process data based on content selection
                if data_content == "Daily Averages":
                    dl_filtered_data = dl_filtered_data.groupby(['Date', 'Station']).mean().reset_index()
                elif data_content == "Monthly Averages":
                    dl_filtered_data['Month'] = pd.to_datetime(dl_filtered_data['Date']).dt.to_period('M')
                    dl_filtered_data = dl_filtered_data.groupby(['Month', 'Station']).mean().reset_index()
                    dl_filtered_data['Date'] = dl_filtered_data['Month'].dt.to_timestamp()
                    dl_filtered_data = dl_filtered_data.drop('Month', axis=1)
                
                # Display a preview of the data
                st.subheader("Data Preview")
                st.dataframe(dl_filtered_data.head(10))
                
                # Generate download link based on file format
                if st.button("Generate Download Link"):
                    # Create a timestamped filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_base_name = f"pm25_data_{timestamp}"
                    
                    if file_format == "CSV":
                        csv_data = dl_filtered_data.to_csv(index=False)
                        # Save a copy to the historical data folder
                        historical_path = os.path.join('data', 'historical', f"{file_base_name}.csv")
                        with open(historical_path, "w") as f:
                            f.write(csv_data)
                        
                        st.success(f"Data saved to {historical_path}")
                        download_link = get_download_link(csv_data, "pm25_data.csv", "text/csv")
                        st.markdown(download_link, unsafe_allow_html=True)
                    
                    elif file_format == "Excel":
                        excel_buffer = BytesIO()
                        dl_filtered_data.to_excel(excel_buffer, index=False)
                        excel_data = excel_buffer.getvalue()
                        
                        # Save a copy to the historical data folder
                        historical_path = os.path.join('data', 'historical', f"{file_base_name}.xlsx")
                        with open(historical_path, "wb") as f:
                            f.write(excel_data)
                        
                        st.success(f"Data saved to {historical_path}")
                        download_link = get_download_link(excel_data, "pm25_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                        st.markdown(download_link, unsafe_allow_html=True)
                    
                    elif file_format == "JSON":
                        json_data = dl_filtered_data.to_json(orient="records")
                        
                        # Save a copy to the historical data folder
                        historical_path = os.path.join('data', 'historical', f"{file_base_name}.json")
                        with open(historical_path, "w") as f:
                            f.write(json_data)
                        
                        st.success(f"Data saved to {historical_path}")
                        download_link = get_download_link(json_data, "pm25_data.json", "application/json")
                        st.markdown(download_link, unsafe_allow_html=True)
            else:
                st.warning("Please select a valid date range and at least one station.")
            
            # List previously downloaded data
            st.subheader("Previously Downloaded Data")
            
            # Check if historical folder exists and has files
            historical_path = os.path.join('data', 'historical')
            if os.path.exists(historical_path) and os.listdir(historical_path):
                # List files
                files = [f for f in os.listdir(historical_path) if os.path.isfile(os.path.join(historical_path, f))]
                
                # Sort by modification time (newest first)
                files.sort(key=lambda x: os.path.getmtime(os.path.join(historical_path, x)), reverse=True)
                
                # Create a dataframe for displaying file information
                file_info = []
                for file in files:
                    file_path = os.path.join(historical_path, file)
                    file_size = os.path.getsize(file_path) / 1024  # Convert to KB
                    file_date = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
                    file_info.append({
                        "Filename": file,
                        "Size (KB)": f"{file_size:.2f}",
                        "Date Created": file_date
                    })
                
                file_df = pd.DataFrame(file_info)
                st.dataframe(file_df, hide_index=True)
            else:
                st.info("No previously downloaded data found.")
        
        # Prediction Data tab
        with download_tabs[1]:
            st.subheader("Download Prediction Data")
            
            # Check if predictions folder exists and has files
            predictions_path = os.path.join('data', 'predictions')
            if os.path.exists(predictions_path) and os.listdir(predictions_path):
                # List files
                files = [f for f in os.listdir(predictions_path) if os.path.isfile(os.path.join(predictions_path, f))]
                
                # Sort by modification time (newest first)
                files.sort(key=lambda x: os.path.getmtime(os.path.join(predictions_path, x)), reverse=True)
                
                # Create a dataframe for displaying file information
                file_info = []
                for file in files:
                    file_path = os.path.join(predictions_path, file)
                    file_size = os.path.getsize(file_path) / 1024  # Convert to KB
                    file_date = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
                    file_info.append({
                        "Filename": file,
                        "Size (KB)": f"{file_size:.2f}",
                        "Date Created": file_date
                    })
                
                file_df = pd.DataFrame(file_info)
                st.dataframe(file_df, hide_index=True)
                
                # Allow user to select a file for download
                selected_file = st.selectbox("Select a file to download", files)
                
                if selected_file:
                    file_path = os.path.join(predictions_path, selected_file)
                    
                    with open(file_path, 'rb') as f:
                        file_data = f.read()
                    
                    # Determine MIME type
                    if selected_file.endswith('.csv'):
                        mime_type = "text/csv"
                    elif selected_file.endswith('.xlsx'):
                        mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    elif selected_file.endswith('.json'):
                        mime_type = "application/json"
                    else:
                        mime_type = "application/octet-stream"
                    
                    # Create download link
                    download_link = get_download_link(file_data, selected_file, mime_type)
                    st.markdown(download_link, unsafe_allow_html=True)
            else:
                st.info("No prediction data found. Make predictions in the 'PM2.5 Predictions' page first.")
        
        # User Analysis Data tab
        with download_tabs[2]:
            st.subheader("Download User Analysis Data")
            
            # List saved user data
            user_data_files = list_user_data()
            
            if user_data_files:
                # Create a dataframe for displaying file information
                file_df = pd.DataFrame(user_data_files)
                st.dataframe(file_df, hide_index=True)
                
                # Allow user to select a file for download
                selected_file = st.selectbox("Select a file to download", [f['filename'] for f in user_data_files])
                
                if selected_file:
                    file_path = os.path.join('data', 'user_data', selected_file)
                    
                    with open(file_path, 'r') as f:
                        file_data = f.read()
                    
                    # Create download link
                    download_link = get_download_link(file_data, selected_file, "application/json")
                    st.markdown(download_link, unsafe_allow_html=True)
            else:
                st.info("No user analysis data found. Save analysis results from the 'Data Analysis and Insights' or 'Historical Analysis' pages first.")