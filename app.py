import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from io import BytesIO
import base64

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

# Set page configuration
st.set_page_config(
    page_title="PM2.5 Air Quality Prediction",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
with open('.streamlit/styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Create necessary folders for storage
if not os.path.exists('data/historical'):
    os.makedirs('data/historical', exist_ok=True)
if not os.path.exists('data/predictions'):
    os.makedirs('data/predictions', exist_ok=True)

# Load the dataset
@st.cache_data
def get_data():
    return load_data()

data = get_data()

# AQI level definitions
def get_aqi_category(pm25):
    if pm25 <= 50:
        return "Good", "good"
    elif pm25 <= 100:
        return "Moderate", "moderate"
    elif pm25 <= 150:
        return "Unhealthy for Sensitive Groups", "unhealthy-for-sensitive-groups"
    elif pm25 <= 200:
        return "Unhealthy", "unhealthy"
    elif pm25 <= 300:
        return "Very Unhealthy", "very-unhealthy"
    else:
        return "Hazardous", "hazardous"

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Station Comparison", "Model Training & Comparison", "PM2.5 Predictions", "Historical Analysis", "Data Download"]
)

# Station selection for predictions
st.sidebar.title("Settings")
selected_station = st.sidebar.selectbox("Select Station", ["Fort William", "Bidhannagar"])

# Display AQI scale in sidebar
st.sidebar.title("AQI Scale")
aqi_levels = [
    {"name": "Good", "range": "0-50", "color": "#00e400"},
    {"name": "Moderate", "range": "51-100", "color": "#ffff00"},
    {"name": "Unhealthy for Sensitive Groups", "range": "101-150", "color": "#ff7e00"},
    {"name": "Unhealthy", "range": "151-200", "color": "#ff0000"},
    {"name": "Very Unhealthy", "range": "201-300", "color": "#8f3f97"},
    {"name": "Hazardous", "range": "301+", "color": "#7e0023"}
]

for level in aqi_levels:
    st.sidebar.markdown(
        f"""<div style="background-color:{level['color']}; color:{'black' if level['name'] in ['Good', 'Moderate'] else 'white'}; 
        padding:5px; border-radius:5px; margin-bottom:5px; text-align:center;">
        <strong>{level['name']} ({level['range']})</strong></div>""", 
        unsafe_allow_html=True
    )

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
        
        # About PM2.5 section
        st.header("About PM2.5 Air Quality Monitoring")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Display PM2.5 icon
            st.image("https://cdn-icons-png.flaticon.com/512/4743/4743525.png", width=150)
        
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
            col1, col2, col3 = st.columns(3)
            
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
                    <div class="weather-card-icon">💨</div>
                    <div class="weather-card-value">{current_weather['Wind_Speed']:.1f}<span class="weather-card-unit">km/h</span></div>
                    <div class="weather-card-label">Wind Speed</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
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
            
            # Train and evaluate models to find best model
            processed_data = preprocess_data(station_data)
            X_train, X_test, y_train, y_test = split_data(processed_data, test_size=0.2)
            models = train_models(X_train, y_train)
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
        
        # Historical trend
        st.subheader("Historical PM2.5 Trend")
        hist_fig = plot_historical_data(station_data)
        st.plotly_chart(hist_fig, use_container_width=True)
        
        # Parameter correlations
        st.subheader("Parameter Correlation with PM2.5")
        
        # Calculate correlation with PM2.5
        corr_with_pm25 = {
            'Temperature': station_data['Temperature'].corr(station_data['PM2.5']),
            'Solar_Radiation': station_data['Solar_Radiation'].corr(station_data['PM2.5']),
            'Wind_Speed': station_data['Wind_Speed'].corr(station_data['PM2.5']),
            'Relative_Humidity': station_data['Relative_Humidity'].corr(station_data['PM2.5']),
        }
        
        # Create bar chart
        corr_df = pd.DataFrame({
            'Parameter': list(corr_with_pm25.keys()),
            'Correlation': list(corr_with_pm25.values())
        })
        
        corr_fig = px.bar(
            corr_df,
            x='Parameter',
            y='Correlation',
            color='Correlation',
            color_continuous_scale='RdBu_r',
            title='Correlation with PM2.5',
            labels={'Correlation': 'Correlation Coefficient'}
        )
        
        st.plotly_chart(corr_fig, use_container_width=True)

# Station Comparison page
elif page == "Station Comparison":
    st.header("Station Comparison")
    
    if data.empty:
        st.warning("No data available for analysis.")
    else:
        # Filter data for both stations
        st.subheader("Compare Fort William and Bidhannagar")
        
        # Date range selection
        st.write("Select date range for comparison:")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min(pd.to_datetime(data['Date'])).date())
        with col2:
            end_date = st.date_input("End Date", max(pd.to_datetime(data['Date'])).date())
        
        if start_date <= end_date:
            # Filter data based on date range
            filtered_data = data[
                (pd.to_datetime(data['Date']).dt.date >= start_date) &
                (pd.to_datetime(data['Date']).dt.date <= end_date)
            ]
            
            # Display PM2.5 comparison
            st.subheader("PM2.5 Comparison")
            
            # Time series comparison
            fig_comparison = plot_historical_data(filtered_data)
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Calculate statistics
            stats = filtered_data.groupby('Station')['PM2.5'].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()
            stats.columns = ['Station', 'Mean', 'Median', 'Minimum', 'Maximum', 'Std Dev']
            stats = stats.round(2)
            
            # Display statistics
            st.subheader("Statistical Comparison")
            st.dataframe(stats, use_container_width=True)
            
            # Create bar chart for mean PM2.5
            fig_stats = px.bar(
                stats,
                x='Station',
                y='Mean',
                color='Station',
                title='Average PM2.5 by Station',
                labels={'Mean': 'Average PM2.5 (μg/m³)'}
            )
            
            st.plotly_chart(fig_stats, use_container_width=True)
            
            # Seasonal comparison
            st.subheader("Seasonal Comparison")
            
            # Add month column for seasonal analysis
            seasonal_data = filtered_data.copy()
            seasonal_data['Month'] = pd.to_datetime(seasonal_data['Date']).dt.month
            
            # Group by month and station
            monthly_avg = seasonal_data.groupby(['Month', 'Station'])['PM2.5'].mean().reset_index()
            
            # Create seasonal plot
            fig_seasonal = px.line(
                monthly_avg,
                x='Month',
                y='PM2.5',
                color='Station',
                markers=True,
                title='Monthly Average PM2.5 by Station',
                labels={'Month': 'Month', 'PM2.5': 'Average PM2.5 (μg/m³)'}
            )
            
            # Update x-axis to show month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            fig_seasonal.update_xaxes(tickvals=list(range(1, 13)), ticktext=month_names)
            
            st.plotly_chart(fig_seasonal, use_container_width=True)
            
            # Parameter comparison
            st.subheader("Parameter Comparison")
            
            # Select parameter
            param = st.selectbox("Select Parameter", ["Temperature", "Solar_Radiation", "Wind_Speed", "Relative_Humidity"])
            
            # Create parameter comparison
            fig_param = px.box(
                filtered_data,
                x='Station',
                y=param,
                color='Station',
                title=f'{param} Comparison',
                labels={param: param}
            )
            
            st.plotly_chart(fig_param, use_container_width=True)
        
        else:
            st.warning("Please select a valid date range.")

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
        
        # Train test split ratio
        split_ratio = st.slider("Test Data Size (%)", 10, 40, 20) / 100
        
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
                    importances = models['Random Forest'].feature_importances_
                    
                    # Create a dataframe for feature importance
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
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
                forecast_df = pd.DataFrame(forecast_data)
                st.dataframe(forecast_df)
                
                # Train models using historical data
                with st.spinner("Training prediction models..."):
                    # Preprocess the data
                    processed_data = preprocess_data(station_data)
                    
                    # Split the data for training
                    X_train, X_test, y_train, y_test = split_data(processed_data, test_size=0.2)
                    
                    # Train the models
                    models = train_models(X_train, y_train)
                
                # Make predictions for the forecast days
                with st.spinner("Making predictions..."):
                    prediction_results = predict_pm25(models, forecast_data)
                
                # Create a dataframe for displaying predictions
                predictions_df = pd.DataFrame({
                    'Date': [item['Date'] for item in forecast_data]
                })
                
                for model_name, predictions in prediction_results.items():
                    predictions_df[model_name] = predictions
                
                # Display predictions
                st.subheader("PM2.5 Predictions by Model")
                st.dataframe(predictions_df)
                
                # Plot predictions
                fig = plot_forecast(predictions_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Find the best model based on historical performance
                with st.spinner("Evaluating model performance..."):
                    evaluation_results = evaluate_models(models, X_test, y_test)
                    best_model_name = min(evaluation_results, key=lambda x: evaluation_results[x]['rmse'])
                
                # Display best model predictions
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
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    forecast_filename = f"pm25_forecast_{selected_station}_{timestamp}.csv"
                    forecast_path = os.path.join('data', 'predictions', forecast_filename)
                    
                    # Save the forecast data and predictions
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
                    st.success(f"Forecast data saved to {forecast_path}")
                
                # Add a download link for the forecast data
                csv_data = predictions_df.to_csv(index=False)
                download_link = get_download_link(csv_data, f"pm25_forecast_{selected_station}.csv", "text/csv")
                st.markdown(download_link, unsafe_allow_html=True)
                
            else:
                st.error("Unable to retrieve weather forecast data. Please try again or use manual parameter input.")
        
        else:  # Manual parameter input
            st.subheader("Enter Weather Parameters")
            
            # Create columns for forecast days
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
                
                # Make predictions for the forecast days
                with st.spinner("Making predictions..."):
                    prediction_results = predict_pm25(models, manual_forecast_data)
                
                # Create a dataframe for displaying predictions
                predictions_df = pd.DataFrame({
                    'Date': [item['Date'] for item in manual_forecast_data]
                })
                
                for model_name, predictions in prediction_results.items():
                    predictions_df[model_name] = predictions
                
                # Display predictions
                st.subheader("PM2.5 Predictions by Model")
                st.dataframe(predictions_df)
                
                # Plot predictions
                fig = plot_forecast(predictions_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Find the best model based on historical performance
                with st.spinner("Evaluating model performance..."):
                    evaluation_results = evaluate_models(models, X_test, y_test)
                    best_model_name = min(evaluation_results, key=lambda x: evaluation_results[x]['rmse'])
                
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
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    forecast_filename = f"pm25_manual_forecast_{selected_station}_{timestamp}.csv"
                    forecast_path = os.path.join('data', 'predictions', forecast_filename)
                    
                    # Save the forecast data and predictions
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
                    st.success(f"Manual forecast data saved to {forecast_path}")
                
                # Add a download link for the forecast data
                csv_data = predictions_df.to_csv(index=False)
                download_link = get_download_link(csv_data, f"pm25_manual_forecast_{selected_station}.csv", "text/csv")
                st.markdown(download_link, unsafe_allow_html=True)

# Historical Analysis page
elif page == "Historical Analysis":
    st.header("Historical Data Analysis")
    
    if data.empty:
        st.warning("No data available for analysis.")
    else:
        # Date range selection
        st.subheader("Select Date Range")
        ha_col1, ha_col2 = st.columns(2)
        with ha_col1:
            ha_start_date = st.date_input("Start Date", min(pd.to_datetime(data['Date'])).date(), key="ha_start")
        with ha_col2:
            ha_end_date = st.date_input("End Date", max(pd.to_datetime(data['Date'])).date(), key="ha_end")
        
        # Station selection
        ha_station_filter = st.multiselect(
            "Select Stations",
            ["Fort William", "Bidhannagar"],
            default=[selected_station],
            key="ha_stations"
        )
        
        if ha_start_date <= ha_end_date and ha_station_filter:
            # Filter data based on date range and stations
            ha_filtered_data = data[
                (pd.to_datetime(data['Date']).dt.date >= ha_start_date) &
                (pd.to_datetime(data['Date']).dt.date <= ha_end_date) &
                (data['Station'].isin(ha_station_filter))
            ].copy()
            
            # Overview of filtered data
            st.subheader("Data Overview")
            
            # Display basic statistics
            stats = ha_filtered_data.groupby('Station')['PM2.5'].agg(['count', 'mean', 'median', 'min', 'max']).reset_index()
            stats.columns = ['Station', 'Count', 'Mean', 'Median', 'Min', 'Max']
            stats = stats.round(2)
            
            st.dataframe(stats)
            
            # Plot historical trends
            st.subheader("Historical PM2.5 Trends")
            fig_hist = plot_historical_data(ha_filtered_data)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Monthly analysis
            st.subheader("Monthly Analysis")
            
            # Add month column for analysis
            ha_filtered_data['Month'] = pd.to_datetime(ha_filtered_data['Date']).dt.month
            
            # Group by month and station
            monthly_avg = ha_filtered_data.groupby(['Month', 'Station'])['PM2.5'].mean().reset_index()
            
            # Plot monthly averages
            fig_monthly = px.line(
                monthly_avg,
                x='Month',
                y='PM2.5',
                color='Station',
                markers=True,
                title='Monthly Average PM2.5',
                labels={'Month': 'Month', 'PM2.5': 'Average PM2.5 (μg/m³)'}
            )
            
            # Update x-axis to show month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            fig_monthly.update_xaxes(tickvals=list(range(1, 13)), ticktext=month_names)
            
            st.plotly_chart(fig_monthly, use_container_width=True)
            
            # Add year column for trend analysis
            ha_filtered_data['Year'] = pd.to_datetime(ha_filtered_data['Date']).dt.year
            
            # Yearly trend analysis if multiple years exist
            years = ha_filtered_data['Year'].unique()
            if len(years) > 1:
                st.subheader("Yearly Trend Analysis")
                
                # Group by year and station
                yearly_avg = ha_filtered_data.groupby(['Year', 'Station'])['PM2.5'].mean().reset_index()
                
                # Plot yearly averages
                fig_yearly = px.line(
                    yearly_avg,
                    x='Year',
                    y='PM2.5',
                    color='Station',
                    markers=True,
                    title='Yearly Average PM2.5',
                    labels={'Year': 'Year', 'PM2.5': 'Average PM2.5 (μg/m³)'}
                )
                
                st.plotly_chart(fig_yearly, use_container_width=True)
            
            # Parameter correlation analysis
            st.subheader("Parameter Correlation Analysis")
            
            # Select station for correlation analysis
            corr_station = st.selectbox("Select Station for Correlation Analysis", ha_station_filter)
            
            # Filter data for selected station
            corr_data = ha_filtered_data[ha_filtered_data['Station'] == corr_station]
            
            # Plot correlation matrix
            fig_corr = plot_correlation_matrix(corr_data)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Parameter comparison
            st.subheader("Parameter Comparison")
            param_x = st.selectbox("Select X Parameter", ["Temperature", "Solar_Radiation", "Wind_Speed", "Relative_Humidity"])
            param_y = st.selectbox("Select Y Parameter", ["PM2.5", "Temperature", "Solar_Radiation", "Wind_Speed", "Relative_Humidity"], index=0)
            
            if param_x != param_y:
                fig_params = plot_parameter_comparison(corr_data, param_x, param_y)
                st.plotly_chart(fig_params, use_container_width=True)
                
                # Calculate and display correlation coefficient
                correlation = corr_data[param_x].corr(corr_data[param_y])
                st.write(f"**Correlation Coefficient between {param_x} and {param_y}:** {correlation:.4f}")
                
                if abs(correlation) > 0.7:
                    st.write("Strong correlation detected!")
                elif abs(correlation) > 0.4:
                    st.write("Moderate correlation detected.")
                else:
                    st.write("Weak correlation detected.")
            else:
                st.warning("Please select different parameters for X and Y axes.")
        
        else:
            st.warning("Please select a valid date range and at least one station.")

# Data Download page
elif page == "Data Download":
    st.header("Download Historical Data")
    
    if data.empty:
        st.warning("No data available for download.")
    else:
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