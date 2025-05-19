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
            # Get weather forecast data
            with st.spinner("Fetching weather forecast data..."):
                forecast_data = get_weather_forecast(selected_station)
                
                if forecast_data:
                    st.success("Weather forecast data retrieved successfully!")
                    
                    # Display the forecast data
                    st.subheader("Weather Forecast Data")
                    
                    # Create a dataframe for the forecast
                    forecast_df = pd.DataFrame(forecast_data)
                    
                    # Display the forecast dataframe
                    st.dataframe(forecast_df)
                    
                    # Make predictions using the forecast data
                    with st.spinner("Generating PM2.5 predictions..."):
                        # Preprocess the data
                        processed_data = preprocess_data(station_data)
                        
                        # Split the data for training
                        X_train, X_test, y_train, y_test = split_data(processed_data, test_size=0.2)
                        
                        # Train the models
                        models = train_models(X_train, y_train)
                        
                        # Make predictions
                        prediction_results = predict_pm25(models, forecast_data)
                        
                        # Display prediction results
                        st.subheader("PM2.5 Predictions")
                        
                        # Create a dataframe for the predictions
                        predictions_df = pd.DataFrame({
                            'Date': [pd.to_datetime(date).strftime("%Y-%m-%d") for date in forecast_df['Date']],
                            'Model Tree': prediction_results['Model Tree'],
                            'NLR': prediction_results['NLR'],
                            'Random Forest': prediction_results['Random Forest'],
                            'Logistic Regression': prediction_results['Logistic Regression']
                        })
                        
                        # Display the predictions dataframe
                        st.dataframe(predictions_df)
                        
                        # Create a plot for the predictions
                        fig_pred = plot_forecast(predictions_df)
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                        # Calculate and display average predictions
                        predictions_df['Average'] = predictions_df[['Model Tree', 'NLR', 'Random Forest', 'Logistic Regression']].mean(axis=1)
                        
                        st.subheader("Summary of Predictions")
                        
                        # Display cards for each prediction day
                        cols = st.columns(len(predictions_df))
                        
                        for i, (_, row) in enumerate(predictions_df.iterrows()):
                            with cols[i]:
                                day_label = "Today" if i == 0 else f"{i} days ahead"
                                st.metric(
                                    label=f"{day_label} ({row['Date']})",
                                    value=f"{row['Average']:.2f} μg/m³"
                                )
                                
                                # Add a color indicator based on PM2.5 level
                                pm25_level = row['Average']
                                if pm25_level <= 12:
                                    st.success("Good")
                                elif pm25_level <= 35:
                                    st.info("Moderate")
                                elif pm25_level <= 55:
                                    st.warning("Unhealthy for Sensitive Groups")
                                else:
                                    st.error("Unhealthy")
                else:
                    st.error("Failed to retrieve weather forecast data. Please try the manual input method.")
        
        else:  # Manual parameter input
            st.subheader("Enter Weather Parameters")
            
            # Get parameter ranges from the dataset
            temp_min, temp_max = station_data['Temperature'].min(), station_data['Temperature'].max()
            solar_min, solar_max = station_data['Solar_Radiation'].min(), station_data['Solar_Radiation'].max()
            wind_min, wind_max = station_data['Wind_Speed'].min(), station_data['Wind_Speed'].max()
            humidity_min, humidity_max = station_data['Relative_Humidity'].min(), station_data['Relative_Humidity'].max()
            
            # Input for current day
            st.write("**Parameters for Current Day:**")
            col1, col2 = st.columns(2)
            
            with col1:
                temp_today = st.slider("Temperature (°C)", float(temp_min), float(temp_max), float((temp_min + temp_max) / 2))
                solar_today = st.slider("Solar Radiation (W/m²)", float(solar_min), float(solar_max), float((solar_min + solar_max) / 2))
            
            with col2:
                wind_today = st.slider("Wind Speed (m/s)", float(wind_min), float(wind_max), float((wind_min + wind_max) / 2))
                humidity_today = st.slider("Relative Humidity (%)", float(humidity_min), float(humidity_max), float((humidity_min + humidity_max) / 2))
            
            # Create manual forecast data
            manual_forecast = [{
                'Date': (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d"),
                'Temperature': temp_today * (1 + (np.random.random() - 0.5) * 0.1 * i),  # slight random variation for future days
                'Solar_Radiation': solar_today * (1 + (np.random.random() - 0.5) * 0.1 * i),
                'Wind_Speed': wind_today * (1 + (np.random.random() - 0.5) * 0.1 * i),
                'Relative_Humidity': humidity_today * (1 + (np.random.random() - 0.5) * 0.1 * i)
            } for i in range(5)]  # 0 = today, 1 = tomorrow, etc.
            
            # Make predictions button
            if st.button("Generate Predictions"):
                with st.spinner("Generating PM2.5 predictions..."):
                    # Preprocess the data
                    processed_data = preprocess_data(station_data)
                    
                    # Split the data for training
                    X_train, X_test, y_train, y_test = split_data(processed_data, test_size=0.2)
                    
                    # Train the models
                    models = train_models(X_train, y_train)
                    
                    # Make predictions
                    prediction_results = predict_pm25(models, manual_forecast)
                    
                    # Display prediction results
                    st.subheader("PM2.5 Predictions")
                    
                    # Create a dataframe for the manual forecast
                    forecast_df = pd.DataFrame(manual_forecast)
                    
                    # Create a dataframe for the predictions
                    predictions_df = pd.DataFrame({
                        'Date': [pd.to_datetime(date).strftime("%Y-%m-%d") for date in forecast_df['Date']],
                        'Model Tree': prediction_results['Model Tree'],
                        'NLR': prediction_results['NLR'],
                        'Random Forest': prediction_results['Random Forest'],
                        'Logistic Regression': prediction_results['Logistic Regression']
                    })
                    
                    # Display the predictions dataframe
                    st.dataframe(predictions_df)
                    
                    # Create a plot for the predictions
                    fig_pred = plot_forecast(predictions_df)
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Calculate and display average predictions
                    predictions_df['Average'] = predictions_df[['Model Tree', 'NLR', 'Random Forest', 'Logistic Regression']].mean(axis=1)
                    
                    st.subheader("Summary of Predictions")
                    
                    # Display cards for each prediction day
                    cols = st.columns(len(predictions_df))
                    
                    for i, (_, row) in enumerate(predictions_df.iterrows()):
                        with cols[i]:
                            day_label = "Today" if i == 0 else f"{i} days ahead"
                            st.metric(
                                label=f"{day_label} ({row['Date']})",
                                value=f"{row['Average']:.2f} μg/m³"
                            )
                            
                            # Add a color indicator based on PM2.5 level
                            pm25_level = row['Average']
                            if pm25_level <= 12:
                                st.success("Good")
                            elif pm25_level <= 35:
                                st.info("Moderate")
                            elif pm25_level <= 55:
                                st.warning("Unhealthy for Sensitive Groups")
                            else:
                                st.error("Unhealthy")
        
        # Display air quality monitoring station image
        st.image("https://pixabay.com/get/g0f687b6e9401a80e1b2c684c0ab1d8528977f03b31ace081ebd832d704d077bdf1c9442dad533d6708c8c088e3c02fba41ead700a93934d380e4b9465809becf_1280.jpg", 
                 caption="Air Quality Monitoring Station")
        
        # Health implications of PM2.5 levels
        with st.expander("Health Implications of PM2.5 Levels"):
            st.markdown("""
            ### PM2.5 Air Quality Index (AQI)
            
            | PM2.5 Level (μg/m³) | Category | Health Implications |
            | --- | --- | --- |
            | 0-12 | Good | Air quality is satisfactory, and air pollution poses little or no risk. |
            | 12.1-35.4 | Moderate | Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution. |
            | 35.5-55.4 | Unhealthy for Sensitive Groups | Members of sensitive groups may experience health effects. The general public is less likely to be affected. |
            | 55.5-150.4 | Unhealthy | Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects. |
            | 150.5-250.4 | Very Unhealthy | Health alert: The risk of health effects is increased for everyone. |
            | 250.5+ | Hazardous | Health warning of emergency conditions: everyone is more likely to be affected. |
            """)

# Historical Analysis page
elif page == "Historical Analysis":
    st.header("Historical Data Analysis")
    
    if data.empty:
        st.warning("No historical data available for analysis.")
    else:
        st.subheader("Explore Historical PM2.5 Data")
        
        # Date range selection
        min_date = pd.to_datetime(data['Date']).min().date()
        max_date = pd.to_datetime(data['Date']).max().date()
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        
        with col2:
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
        
        # Station selection
        station_filter_hist = st.multiselect(
            "Select Stations",
            ["Fort William", "Bidhannagar"],
            default=["Fort William", "Bidhannagar"]
        )
        
        if start_date <= end_date and station_filter_hist:
            # Filter data based on date range and stations
            filtered_data = data[
                (pd.to_datetime(data['Date']).dt.date >= start_date) &
                (pd.to_datetime(data['Date']).dt.date <= end_date) &
                (data['Station'].isin(station_filter_hist))
            ].copy()
            
            # Display the number of records
            st.write(f"**Number of records:** {len(filtered_data)}")
            
            # Show the filtered data
            with st.expander("View Filtered Data"):
                st.dataframe(filtered_data)
            
            # PM2.5 trend over time
            st.subheader("PM2.5 Trend Over Time")
            
            fig_trend = px.line(
                filtered_data,
                x='Date',
                y='PM2.5',
                color='Station',
                title='PM2.5 Levels Over Time',
                labels={'PM2.5': 'PM2.5 (μg/m³)', 'Date': 'Date'}
            )
            
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # Monthly averages
            st.subheader("Monthly Average PM2.5 Levels")
            
            # Add month and year columns for analysis
            filtered_data['Month'] = pd.to_datetime(filtered_data['Date']).dt.month
            filtered_data['Year'] = pd.to_datetime(filtered_data['Date']).dt.year
            
            # Group by year, month, and station to calculate monthly averages
            monthly_avg = filtered_data.groupby(['Year', 'Month', 'Station'])['PM2.5'].mean().reset_index()
            monthly_avg['Date'] = pd.to_datetime(monthly_avg[['Year', 'Month']].assign(Day=1))
            
            fig_monthly = px.line(
                monthly_avg,
                x='Date',
                y='PM2.5',
                color='Station',
                markers=True,
                title='Monthly Average PM2.5 Levels',
                labels={'PM2.5': 'Average PM2.5 (μg/m³)', 'Date': 'Month-Year'}
            )
            
            st.plotly_chart(fig_monthly, use_container_width=True)
            
            # Seasonal analysis
            st.subheader("Seasonal Analysis")
            
            # Define seasons (meteorological seasons in Northern Hemisphere)
            season_mapping = {
                1: 'Winter', 2: 'Winter', 3: 'Spring',
                4: 'Spring', 5: 'Spring', 6: 'Summer',
                7: 'Summer', 8: 'Summer', 9: 'Autumn',
                10: 'Autumn', 11: 'Autumn', 12: 'Winter'
            }
            
            filtered_data['Season'] = filtered_data['Month'].map(season_mapping)
            
            # Group by season and station to calculate seasonal averages
            seasonal_avg = filtered_data.groupby(['Season', 'Station'])['PM2.5'].mean().reset_index()
            
            # Define season order
            season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
            seasonal_avg['Season'] = pd.Categorical(seasonal_avg['Season'], categories=season_order, ordered=True)
            seasonal_avg = seasonal_avg.sort_values('Season')
            
            fig_seasonal = px.bar(
                seasonal_avg,
                x='Season',
                y='PM2.5',
                color='Station',
                barmode='group',
                title='Seasonal Average PM2.5 Levels',
                labels={'PM2.5': 'Average PM2.5 (μg/m³)', 'Season': 'Season'}
            )
            
            st.plotly_chart(fig_seasonal, use_container_width=True)
            
            # Yearly trend analysis
            st.subheader("Yearly Trend Analysis")
            
            # Group by year and station to calculate yearly averages
            yearly_avg = filtered_data.groupby(['Year', 'Station'])['PM2.5'].mean().reset_index()
            
            fig_yearly = px.line(
                yearly_avg,
                x='Year',
                y='PM2.5',
                color='Station',
                markers=True,
                title='Yearly Average PM2.5 Levels',
                labels={'PM2.5': 'Average PM2.5 (μg/m³)', 'Year': 'Year'}
            )
            
            # Format x-axis to show only years
            fig_yearly.update_xaxes(tickvals=yearly_avg['Year'].unique())
            
            st.plotly_chart(fig_yearly, use_container_width=True)
            
            # Weather parameter correlation with PM2.5
            st.subheader("Weather Parameter Correlation with PM2.5")
            
            # Select station for correlation analysis
            station_for_hist_corr = st.selectbox(
                "Select Station for Correlation Analysis",
                station_filter_hist
            )
            
            # Filter data for the selected station
            station_hist_data = filtered_data[filtered_data['Station'] == station_for_hist_corr]
            
            # Create correlation plot for each parameter
            param_options = ['Temperature', 'Solar_Radiation', 'Wind_Speed', 'Relative_Humidity']
            selected_param = st.selectbox("Select Parameter", param_options)
            
            fig_param_corr = px.scatter(
                station_hist_data,
                x=selected_param,
                y='PM2.5',
                color='Season',
                trendline='ols',
                title=f'Correlation between {selected_param} and PM2.5',
                labels={selected_param: selected_param, 'PM2.5': 'PM2.5 (μg/m³)'}
            )
            
            st.plotly_chart(fig_param_corr, use_container_width=True)
            
            # Calculate and display correlation coefficient
            corr_coef = station_hist_data[selected_param].corr(station_hist_data['PM2.5'])
            st.write(f"**Correlation Coefficient:** {corr_coef:.4f}")
            
            # Interpret the correlation
            if abs(corr_coef) < 0.3:
                st.write("Weak correlation")
            elif abs(corr_coef) < 0.7:
                st.write("Moderate correlation")
            else:
                st.write("Strong correlation")
            
            # Boxplot for PM2.5 distribution
            st.subheader("PM2.5 Distribution by Station")
            
            fig_box = px.box(
                filtered_data,
                x='Station',
                y='PM2.5',
                color='Station',
                title='PM2.5 Distribution by Station',
                labels={'PM2.5': 'PM2.5 (μg/m³)', 'Station': 'Station'}
            )
            
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Display air quality monitoring image
            st.image("https://pixabay.com/get/g376032a20c7986a790f2472e6226aba2d4f8f4b9ecd2e1ddf045c15c61622fb2126687679fc72edd5e96bc46ce6396193f84df658f200321da90abf7cd9d7341_1280.jpg", 
                     caption="Air Quality Monitoring Station")
        
        else:
            if start_date > end_date:
                st.error("Start date must be before or equal to end date.")
            if not station_filter_hist:
                st.error("Please select at least one station.")

# Download Data page
elif page == "Download Data":
    st.header("Download Data")
    
    if data.empty:
        st.warning("No data available for download.")
    else:
        st.subheader("Download Options")
        
        # Date range selection for download
        min_date = pd.to_datetime(data['Date']).min().date()
        max_date = pd.to_datetime(data['Date']).max().date()
        
        col1, col2 = st.columns(2)
        
        with col1:
            dl_start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date, key="dl_start")
        
        with col2:
            dl_end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date, key="dl_end")
        
        # Station selection for download
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
                if file_format == "CSV":
                    csv_data = dl_filtered_data.to_csv(index=False)
                    download_link = get_download_link(csv_data, "pm25_data.csv", "text/csv")
                    st.markdown(download_link, unsafe_allow_html=True)
                
                elif file_format == "Excel":
                    excel_buffer = BytesIO()
                    dl_filtered_data.to_excel(excel_buffer, index=False)
                    excel_data = excel_buffer.getvalue()
                    download_link = get_download_link(excel_data, "pm25_data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    st.markdown(download_link, unsafe_allow_html=True)
                
                elif file_format == "JSON":
                    json_data = dl_filtered_data.to_json(orient="records")
                    download_link = get_download_link(json_data, "pm25_data.json", "application/json")
                    st.markdown(download_link, unsafe_allow_html=True)
            
            # Display data visualization graphics image
            st.image("https://pixabay.com/get/ge17a111ad0bedc0ce521590a579161e835cc223ccf9639284ed96d2cfc9b1760a9fb39a4ab44b7b8a828adc8edfa752f8bbfb749c98690b22a23a6bce9ca9e71_1280.jpg", 
                     caption="Data Visualization Graphics")
        
        else:
            if dl_start_date > dl_end_date:
                st.error("Start date must be before or equal to end date.")
            if not dl_station_filter:
                st.error("Please select at least one station.")

# Footer
st.markdown("---")
st.caption("PM2.5 Prediction Application © 2023 | Powered by Streamlit and OpenWeatherMap")
