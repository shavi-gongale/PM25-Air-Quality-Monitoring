import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    page_title="PM2.5 Prediction Application",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title
st.title("Air Quality (PM2.5) Prediction Application")

# Add a short description
st.markdown("""
This application predicts PM2.5 levels for two stations (Fort William and Bidhannagar) using multiple machine learning models. 
It provides predictions for the present day and forecasts for 1, 2, 4, and 7 days ahead based on weather parameters.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Data Analysis", "Model Training & Comparison", "Predictions", "Historical Analysis", "Download Data", "Historical Data Storage"]
)

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

# Station selection for predictions
st.sidebar.title("Prediction Settings")
selected_station = st.sidebar.selectbox("Select Station", ["Fort William", "Bidhannagar"])

# Overview page
if page == "Overview":
    st.header("Air Quality Monitoring Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("https://pixabay.com/get/ga8350ca0ed2512b727809d73188d40ef6145f833bc6ada38864631107c299ffd0d0c09f0a75364002dd99e310d10f85b_1280.jpg", 
                 caption="Air Quality Monitoring Station")
        st.markdown("""
        ### About PM2.5
        PM2.5 refers to atmospheric particulate matter that has a diameter less than 2.5 micrometers. 
        These fine particles are a common air pollutant and can have significant health impacts when inhaled.
        
        ### Health Effects
        - Respiratory issues
        - Cardiovascular problems
        - Decreased lung function
        - Aggravated asthma
        """)
    
    with col2:
        st.image("https://pixabay.com/get/g899f6d89fc835ae5948a5875577f597badf04a1c094ff8ef03133d3891f6a7b3c14a956bcd54c83ef470c24f5ba234bc3bfa65afacaffe07ac5caa9c7f96e0aa_1280.jpg", 
                 caption="Weather Monitoring Equipment")
        st.markdown("""
        ### Prediction Factors
        Our model uses the following parameters to predict PM2.5 levels:
        - Temperature
        - Solar Radiation
        - Wind Speed
        - Relative Humidity
        
        ### Monitoring Stations
        We provide predictions for two stations:
        - Fort William
        - Bidhannagar
        """)
    
    st.subheader("Current Air Quality Status")
    
    # Display some basic stats from the dataset
    if not data.empty:
        last_date = data['Date'].max()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Latest PM2.5 Reading - Fort William", 
                value=f"{data[data['Station'] == 'Fort William'].iloc[-1]['PM2.5']:.2f} μg/m³",
                delta=f"{data[data['Station'] == 'Fort William'].iloc[-1]['PM2.5'] - data[data['Station'] == 'Fort William'].iloc[-2]['PM2.5']:.2f} μg/m³"
            )
        
        with col2:
            st.metric(
                label="Latest PM2.5 Reading - Bidhannagar", 
                value=f"{data[data['Station'] == 'Bidhannagar'].iloc[-1]['PM2.5']:.2f} μg/m³",
                delta=f"{data[data['Station'] == 'Bidhannagar'].iloc[-1]['PM2.5'] - data[data['Station'] == 'Bidhannagar'].iloc[-2]['PM2.5']:.2f} μg/m³"
            )
        
        st.subheader("What this application offers:")
        st.markdown("""
        - **Real-time Predictions**: Get PM2.5 forecasts for the present day using current weather data
        - **Future Forecasts**: Predict PM2.5 levels for 1, 2, 4, and 7 days ahead
        - **Multiple Models**: Utilizes Model Tree, NLR, Random Forest, and Logistic Regression
        - **Model Comparison**: Compare different models' performance with metrics like RMSE and correlation
        - **Data Analysis**: Explore historical trends and relationships between parameters
        - **Data Download**: Export historical and prediction data in various formats
        - **Historical Data Storage**: Store and manage your downloaded historical data
        """)
        
        st.image("https://pixabay.com/get/gd9254fa6c0fd331ab3029d83b28d4e5f966740bd31b6f33cd52e651956326f8a6ff241f438d94b25c947143be03007da94730d81e5ad2094fba1ff6772233819_1280.jpg", 
                 caption="Data Visualization")

# Data Analysis page
elif page == "Data Analysis":
    st.header("Data Analysis")
    
    if data.empty:
        st.warning("No data available for analysis.")
    else:
        st.subheader("Dataset Overview")
        
        # Basic dataset information
        st.write(f"**Dataset Shape:** {data.shape}")
        st.write(f"**Date Range:** {data['Date'].min()} to {data['Date'].max()}")
        
        # Display a sample of the data
        with st.expander("View Data Sample"):
            st.dataframe(data.head(10))
        
        # Summary statistics
        with st.expander("View Summary Statistics"):
            st.dataframe(data.describe())
        
        # Historical PM2.5 levels
        st.subheader("Historical PM2.5 Trends")
        station_filter = st.multiselect("Select Stations", ["Fort William", "Bidhannagar"], default=["Fort William", "Bidhannagar"])
        
        if station_filter:
            filtered_data = data[data['Station'].isin(station_filter)]
            fig = plot_historical_data(filtered_data)
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("Parameter Correlation Analysis")
        station_for_corr = st.selectbox("Select Station for Correlation Analysis", ["Fort William", "Bidhannagar"])
        
        station_data = data[data['Station'] == station_for_corr]
        fig_corr = plot_correlation_matrix(station_data)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Parameter comparison
        st.subheader("Parameter Comparison")
        param_x = st.selectbox("Select X Parameter", ["Temperature", "Solar_Radiation", "Wind_Speed", "Relative_Humidity"])
        param_y = st.selectbox("Select Y Parameter", ["PM2.5", "Temperature", "Solar_Radiation", "Wind_Speed", "Relative_Humidity"], index=0)
        
        if param_x != param_y:
            fig_params = plot_parameter_comparison(station_data, param_x, param_y)
            st.plotly_chart(fig_params, use_container_width=True)
            
            # Calculate and display correlation coefficient
            correlation = station_data[param_x].corr(station_data[param_y])
            st.write(f"**Correlation Coefficient between {param_x} and {param_y}:** {correlation:.4f}")
            
            if abs(correlation) > 0.7:
                st.write("Strong correlation detected!")
            elif abs(correlation) > 0.4:
                st.write("Moderate correlation detected.")
            else:
                st.write("Weak correlation detected.")
        else:
            st.warning("Please select different parameters for X and Y axes.")
        
        # Seasonal analysis
        st.subheader("Seasonal Analysis")
        
        # Add month column for seasonal analysis
        analysis_data = data.copy()
        analysis_data['Month'] = pd.to_datetime(analysis_data['Date']).dt.month
        
        # Group by month and station and calculate average PM2.5
        monthly_avg = analysis_data.groupby(['Month', 'Station'])['PM2.5'].mean().reset_index()
        
        fig_seasonal = px.line(
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
        fig_seasonal.update_xaxes(tickvals=list(range(1, 13)), ticktext=month_names)
        
        st.plotly_chart(fig_seasonal, use_container_width=True)
        
        # Display weather monitoring equipment image
        st.image("https://pixabay.com/get/gf3594d2070170cf15887285cf9260d1892d4c4dc771e4390d229210a34fecda047b588772b967e4e47dcb274c02b24e7e5d17bbe0f3410c323e4f5d63159522e_1280.jpg", 
                 caption="Weather Monitoring Equipment")

# Model Training & Comparison page
elif page == "Model Training & Comparison":
    st.header("Model Training & Comparison")
    
    if data.empty:
        st.warning("No data available for model training.")
    else:
        st.subheader("Train and Evaluate Models")
        
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
        
        # Display data visualization image
        st.image("https://pixabay.com/get/g0ccd2d803467a8cbb9eea38deeb5166dcc31f1d24f5d30c77157a5022ba4065b3612792d9437b2463f9724a1f454701b614bf600bbea160f41303b2ff217c528_1280.png", 
                 caption="Data Visualization Graphics")

# Predictions page
elif page == "Predictions":
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
                st.subheader("PM2.5 Predictions")
                st.dataframe(predictions_df)
                
                # Plot predictions
                fig = plot_forecast(predictions_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Find the best model based on historical performance
                with st.spinner("Evaluating model performance..."):
                    evaluation_results = evaluate_models(models, X_test, y_test)
                    best_model_name = min(evaluation_results, key=lambda x: evaluation_results[x]['rmse'])
                
                st.subheader("Best Model Prediction")
                st.write(f"Based on historical performance, the **{best_model_name}** model is most accurate with an RMSE of {evaluation_results[best_model_name]['rmse']:.2f}.")
                
                # Display best model predictions
                best_pred_df = pd.DataFrame({
                    'Date': predictions_df['Date'],
                    'PM2.5': predictions_df[best_model_name]
                })
                
                st.dataframe(best_pred_df)
                
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
                        save_df[f'PM2.5_{model_name}'] = predictions
                    
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
            
            # Create five columns for the five prediction days
            cols = st.columns(5)
            days = [0, 1, 2, 4, 7]  # Present, and 1, 2, 4, 7 days ahead
            
            manual_forecast_data = []
            
            for i, day in enumerate(days):
                with cols[i]:
                    forecast_date = (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d")
                    st.write(f"**{forecast_date}**")
                    
                    temp = st.number_input(f"Temperature (°C) - Day {day}", min_value=-20.0, max_value=50.0, value=25.0, key=f"temp_{i}")
                    solar = st.number_input(f"Solar Radiation (W/m²) - Day {day}", min_value=0.0, max_value=1500.0, value=250.0, key=f"solar_{i}")
                    wind = st.number_input(f"Wind Speed (m/s) - Day {day}", min_value=0.0, max_value=30.0, value=3.0, key=f"wind_{i}")
                    humidity = st.number_input(f"Relative Humidity (%) - Day {day}", min_value=0.0, max_value=100.0, value=65.0, key=f"humidity_{i}")
                    
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
                st.subheader("PM2.5 Predictions")
                st.dataframe(predictions_df)
                
                # Plot predictions
                fig = plot_forecast(predictions_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Find the best model based on historical performance
                with st.spinner("Evaluating model performance..."):
                    evaluation_results = evaluate_models(models, X_test, y_test)
                    best_model_name = min(evaluation_results, key=lambda x: evaluation_results[x]['rmse'])
                
                st.subheader("Best Model Prediction")
                st.write(f"Based on historical performance, the **{best_model_name}** model is most accurate with an RMSE of {evaluation_results[best_model_name]['rmse']:.2f}.")
                
                # Display best model predictions
                best_pred_df = pd.DataFrame({
                    'Date': predictions_df['Date'],
                    'PM2.5': predictions_df[best_model_name]
                })
                
                st.dataframe(best_pred_df)
                
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
                        save_df[f'PM2.5_{model_name}'] = predictions
                    
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
            default=["Fort William", "Bidhannagar"],
            key="ha_stations"
        )
        
        if ha_start_date <= ha_end_date and ha_station_filter:
            # Filter data based on date range and stations
            ha_filtered_data = data[
                (pd.to_datetime(data['Date']).dt.date >= ha_start_date) &
                (pd.to_datetime(data['Date']).dt.date <= ha_end_date) &
                (data['Station'].isin(ha_station_filter))
            ].copy()
            
            # Display filtered data
            st.subheader("Filtered Data")
            st.dataframe(ha_filtered_data)
            
            # Plot historical trends
            st.subheader("Historical PM2.5 Trends")
            fig_hist = plot_historical_data(ha_filtered_data)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Calculate and display statistics
            st.subheader("Statistical Summary")
            
            # Group by station and calculate statistics
            stats = ha_filtered_data.groupby('Station')['PM2.5'].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()
            stats.columns = ['Station', 'Mean', 'Median', 'Minimum', 'Maximum', 'Std Dev']
            stats = stats.round(2)
            
            st.dataframe(stats)
            
            # Bar chart for comparing statistics
            fig_stats = px.bar(
                stats,
                x='Station',
                y=['Mean', 'Median', 'Minimum', 'Maximum'],
                barmode='group',
                title='PM2.5 Statistics by Station',
                labels={'value': 'PM2.5 (μg/m³)', 'variable': 'Statistic'}
            )
            st.plotly_chart(fig_stats, use_container_width=True)
            
            # Time series decomposition
            st.subheader("Time Series Analysis")
            
            # Select station for time series analysis
            ts_station = st.selectbox("Select Station for Time Series Analysis", ha_station_filter)
            
            # Filter data for the selected station
            ts_data = ha_filtered_data[ha_filtered_data['Station'] == ts_station].copy()
            
            # Ensure data is sorted by date
            ts_data = ts_data.sort_values('Date')
            
            # Plot the time series
            fig_ts = px.line(
                ts_data,
                x='Date',
                y='PM2.5',
                title=f'PM2.5 Time Series for {ts_station}',
                labels={'PM2.5': 'PM2.5 (μg/m³)', 'Date': 'Date'}
            )
            
            # Add a trendline
            ts_data['day_index'] = range(len(ts_data))
            z = np.polyfit(ts_data['day_index'], ts_data['PM2.5'], 1)
            trend = np.poly1d(z)
            
            fig_ts.add_trace(go.Scatter(
                x=ts_data['Date'],
                y=trend(ts_data['day_index']),
                mode='lines',
                name='Trend',
                line=dict(color='red', dash='dash')
            ))
            
            st.plotly_chart(fig_ts, use_container_width=True)
            
            # Display trend information
            if z[0] > 0:
                st.write(f"The PM2.5 levels in {ts_station} show an **increasing trend** of {z[0]:.4f} μg/m³ per day.")
            elif z[0] < 0:
                st.write(f"The PM2.5 levels in {ts_station} show a **decreasing trend** of {abs(z[0]):.4f} μg/m³ per day.")
            else:
                st.write(f"The PM2.5 levels in {ts_station} show **no significant trend**.")
            
            # Monthly average analysis
            st.subheader("Monthly Average Analysis")
            
            # Extract month and year
            ts_data['Month'] = pd.to_datetime(ts_data['Date']).dt.month
            ts_data['Year'] = pd.to_datetime(ts_data['Date']).dt.year
            
            # Group by month and calculate average
            monthly_avg = ts_data.groupby(['Year', 'Month'])['PM2.5'].mean().reset_index()
            monthly_avg['Date'] = pd.to_datetime(monthly_avg[['Year', 'Month']].assign(day=1))
            
            # Plot monthly averages
            fig_monthly = px.line(
                monthly_avg,
                x='Date',
                y='PM2.5',
                markers=True,
                title=f'Monthly Average PM2.5 Levels for {ts_station}',
                labels={'PM2.5': 'Average PM2.5 (μg/m³)', 'Date': 'Month'}
            )
            
            st.plotly_chart(fig_monthly, use_container_width=True)
            
            # Seasonal patterns
            st.subheader("Seasonal Patterns")
            
            # Group by month across all years
            seasonal_avg = ts_data.groupby('Month')['PM2.5'].mean().reset_index()
            
            # Plot seasonal pattern
            fig_seasonal = px.line(
                seasonal_avg,
                x='Month',
                y='PM2.5',
                markers=True,
                title=f'Seasonal PM2.5 Pattern for {ts_station}',
                labels={'PM2.5': 'Average PM2.5 (μg/m³)', 'Month': 'Month'}
            )
            
            # Update x-axis to show month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            fig_seasonal.update_xaxes(tickvals=list(range(1, 13)), ticktext=month_names)
            
            st.plotly_chart(fig_seasonal, use_container_width=True)
                
        else:
            st.warning("Please select a valid date range and at least one station.")

# Download Data page
elif page == "Download Data":
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
            default=["Fort William", "Bidhannagar"],
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

# Historical Data Storage page
elif page == "Historical Data Storage":
    st.header("Historical Data Storage")
    
    st.write("""
    This page helps you manage your stored historical data. You can view, analyze, and manage the data 
    files that you've saved from the application.
    """)
    
    # Tabs for different types of data
    data_tab = st.radio("Select Data Type", ["Historical Data", "Forecast Data"])
    
    if data_tab == "Historical Data":
        # List historical data files
        historical_path = os.path.join('data', 'historical')
        historical_files = [f for f in os.listdir(historical_path) if os.path.isfile(os.path.join(historical_path, f))]
        
        if not historical_files:
            st.info("No historical data files found. You can save historical data from the 'Download Data' page.")
        else:
            st.subheader("Available Historical Data Files")
            
            # Sort files by modification time (newest first)
            historical_files.sort(key=lambda x: os.path.getmtime(os.path.join(historical_path, x)), reverse=True)
            
            # Create a dataframe for displaying file information
            file_info = []
            for file in historical_files:
                file_path = os.path.join(historical_path, file)
                file_size = os.path.getsize(file_path) / 1024  # Convert to KB
                file_date = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
                file_info.append({
                    "Filename": file,
                    "Size (KB)": f"{file_size:.2f}",
                    "Date Modified": file_date
                })
            
            file_df = pd.DataFrame(file_info)
            st.dataframe(file_df)
            
            # Select a file to view
            selected_file = st.selectbox("Select a file to view", historical_files)
            
            if selected_file:
                file_path = os.path.join(historical_path, selected_file)
                
                # Determine file format and load data
                if selected_file.endswith('.csv'):
                    hist_data = pd.read_csv(file_path)
                elif selected_file.endswith('.xlsx'):
                    hist_data = pd.read_excel(file_path)
                elif selected_file.endswith('.json'):
                    hist_data = pd.read_json(file_path)
                else:
                    st.error("Unsupported file format.")
                    hist_data = None
                
                if hist_data is not None:
                    # Display the data
                    st.subheader(f"Data Preview: {selected_file}")
                    st.dataframe(hist_data)
                    
                    # Visualize data if it has the correct format
                    if all(col in hist_data.columns for col in ['Date', 'PM2.5']):
                        st.subheader("Data Visualization")
                        
                        # Check if Station column exists
                        if 'Station' in hist_data.columns:
                            # Plot by station
                            fig = px.line(
                                hist_data,
                                x='Date',
                                y='PM2.5',
                                color='Station',
                                title='PM2.5 Levels by Station',
                                labels={'PM2.5': 'PM2.5 (μg/m³)', 'Date': 'Date'}
                            )
                        else:
                            # Plot without station differentiation
                            fig = px.line(
                                hist_data,
                                x='Date',
                                y='PM2.5',
                                title='PM2.5 Levels',
                                labels={'PM2.5': 'PM2.5 (μg/m³)', 'Date': 'Date'}
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Option to delete the file
                    if st.button(f"Delete {selected_file}"):
                        try:
                            os.remove(file_path)
                            st.success(f"File {selected_file} deleted successfully.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting file: {e}")
    
    else:  # Forecast Data
        # List forecast data files
        forecasts_path = os.path.join('data', 'predictions')
        forecast_files = [f for f in os.listdir(forecasts_path) if os.path.isfile(os.path.join(forecasts_path, f))]
        
        if not forecast_files:
            st.info("No forecast data files found. You can save forecast data from the 'Predictions' page.")
        else:
            st.subheader("Available Forecast Data Files")
            
            # Sort files by modification time (newest first)
            forecast_files.sort(key=lambda x: os.path.getmtime(os.path.join(forecasts_path, x)), reverse=True)
            
            # Create a dataframe for displaying file information
            file_info = []
            for file in forecast_files:
                file_path = os.path.join(forecasts_path, file)
                file_size = os.path.getsize(file_path) / 1024  # Convert to KB
                file_date = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
                file_info.append({
                    "Filename": file,
                    "Size (KB)": f"{file_size:.2f}",
                    "Date Modified": file_date
                })
            
            file_df = pd.DataFrame(file_info)
            st.dataframe(file_df)
            
            # Select a file to view
            selected_file = st.selectbox("Select a file to view", forecast_files)
            
            if selected_file:
                file_path = os.path.join(forecasts_path, selected_file)
                
                # Load data (assuming CSV format for forecast files)
                if selected_file.endswith('.csv'):
                    forecast_data = pd.read_csv(file_path)
                    
                    # Display the data
                    st.subheader(f"Data Preview: {selected_file}")
                    st.dataframe(forecast_data)
                    
                    # Check if we have the necessary columns for visualization
                    pm25_cols = [col for col in forecast_data.columns if 'PM2.5' in col]
                    
                    if 'Date' in forecast_data.columns and pm25_cols:
                        st.subheader("Forecast Visualization")
                        
                        # Create plot
                        fig = go.Figure()
                        
                        for col in pm25_cols:
                            fig.add_trace(go.Scatter(
                                x=forecast_data['Date'],
                                y=forecast_data[col],
                                mode='lines+markers',
                                name=col
                            ))
                        
                        fig.update_layout(
                            title='PM2.5 Forecast Comparison',
                            xaxis_title='Date',
                            yaxis_title='PM2.5 (μg/m³)',
                            legend_title='Model',
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Option to delete the file
                    if st.button(f"Delete {selected_file}"):
                        try:
                            os.remove(file_path)
                            st.success(f"File {selected_file} deleted successfully.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting file: {e}")
                else:
                    st.error("Unsupported file format. Forecast files should be in CSV format.")