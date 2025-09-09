# Overview

This is a comprehensive air quality monitoring and prediction application built with Streamlit that focuses on PM2.5 particulate matter forecasting for two locations in Kolkata, India: Fort William and Bidhannagar. The application uses machine learning models to predict air quality levels based on meteorological parameters and provides real-time monitoring, historical analysis, and multi-day forecasts. It integrates with external weather APIs, satellite imagery services, and air quality monitoring systems to deliver actionable insights for public health and environmental awareness.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
The application uses Streamlit as the primary web framework, providing an interactive dashboard with multiple pages for data analysis, model training, predictions, and historical analysis. The interface includes custom CSS styling for air quality level color coding (good, moderate, unhealthy, etc.) and responsive design elements. The application supports both web-based access and Android mobile app wrapper functionality through WebView integration.

## Backend Architecture
The system follows a modular Python architecture with specialized modules for different functionalities:

- **Data Handler**: Manages data loading, preprocessing, and synthetic dataset generation when real data is unavailable
- **Model Trainer**: Implements multiple machine learning algorithms (Random Forest, Decision Tree, Linear Regression, Logistic Regression) with cross-validation and performance evaluation
- **Predictor**: Handles real-time and forecast predictions using trained models with feature scaling and data preparation
- **Visualizer**: Creates interactive Plotly charts for historical data, correlations, model comparisons, and forecasts
- **Weather API Integration**: Interfaces with OpenWeatherMap API for meteorological data
- **Air Quality APIs**: Integrates with WAQI (World Air Quality Index) for real-time air quality monitoring
- **Satellite Data**: Connects to NASA Earth imagery services for environmental visualization

## Data Storage Solutions
The application uses a file-based storage system with organized directory structure:
- JSON files for user predictions and forecasts stored in `data/user_data/`
- Historical data storage in `data/historical/`
- Model serialization using pickle for trained machine learning models
- CSV data handling for air quality and meteorological datasets
- Support for Excel file imports with automatic column mapping

## Machine Learning Pipeline
The system implements a comprehensive ML pipeline with:
- Feature engineering including seasonal patterns (sine/cosine transformations for temporal data)
- Data preprocessing with StandardScaler normalization and missing value imputation
- Multiple model training with hyperparameter optimization
- Model evaluation using RMSE, MAE, and R² metrics
- Cross-validation for robust performance assessment
- Model persistence and loading for real-time predictions

## Health Recommendation System
The application includes an intelligent health advisory system that generates recommendations based on PM2.5 and AQI levels, providing activity suggestions and health impact assessments for different population groups (general public, sensitive individuals, children, elderly).

# External Dependencies

## Weather and Environmental APIs
- **OpenWeatherMap API**: Primary source for weather forecasts and current conditions, requiring API key authentication
- **WAQI (World Air Quality Index) API**: Real-time air quality data and historical trends
- **NASA Earth Imagery Services**: Satellite imagery and environmental visualization data

## Machine Learning Libraries
- **Scikit-learn**: Core machine learning framework for model training, evaluation, and preprocessing
- **NumPy/Pandas**: Data manipulation and numerical computing
- **Plotly**: Interactive data visualization and charting

## Web Framework and UI
- **Streamlit**: Main web application framework with built-in session state management
- **Custom CSS**: Air quality level styling and responsive design elements

## Mobile Integration
- **Android WebView**: Mobile app wrapper for native Android experience
- **Progressive Web App capabilities**: Offline functionality and caching

## Data Processing
- **JSON/CSV/Excel**: Multiple data format support for imports and exports
- **Base64 encoding**: File download and data transfer functionality
- **Pickle serialization**: Model persistence and loading

## Development and Deployment
- **Python 3.11+**: Runtime environment
- **Replit hosting**: Cloud-based development and deployment platform
- **Environment variable management**: Secure API key and configuration handling

The system is designed to gracefully handle API failures by providing mock data generators and offline functionality, ensuring continuous operation even when external services are unavailable.