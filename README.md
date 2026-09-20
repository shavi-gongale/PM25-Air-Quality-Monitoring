# PM2.5 Air Quality Monitoring & Prediction System

## Overview

PM2.5 Air Quality Monitoring & Prediction System is a Streamlit-based application for monitoring particulate pollution at Fort William and Bidhannagar in Kolkata, India.

The dashboard combines stored observations with current weather and air-quality data, historical analysis, interactive visualizations, machine-learning predictions, satellite imagery, health guidance, and downloadable results. The repository also includes a separate Android WebView application that can connect to a deployed or locally hosted Streamlit instance.

## Key Features

- PM2.5 monitoring and AQI categorization
- Current weather and air-quality conditions for supported stations
- Historical PM2.5 and weather analysis
- Multi-day PM2.5 prediction workflows
- Interactive Plotly and Matplotlib visualizations
- Model training, evaluation, comparison, and feature-importance views
- OpenWeatherMap weather integration
- WAQI air-quality integration
- Open-Meteo fallback for current weather and PM2.5 when authenticated providers are unavailable
- NASA-linked satellite imagery display
- Generated air-quality maps and historical satellite-style estimates
- Downloadable historical, prediction, and user-analysis data
- Health recommendations based on PM2.5 and AQI values
- Optional standalone OpenAI analysis module
- Optional Android WebView application

## Application Pages

The Streamlit navigation contains these pages:

- **Dashboard**: Current PM2.5 status, AQI category, weather context, health guidance, station information, model summaries, and operational actions.
- **Data Analysis and Insights**: Correlations, historical trends, seasonal patterns, prediction accuracy, and forecasting insights.
- **Model Training & Comparison**: Model training, evaluation metrics, RMSE and R2 comparisons, predictions versus actual values, and feature importance.
- **PM2.5 Predictions**: Current conditions, manual or live inputs, multi-day forecasts, model comparisons, confidence information, and downloads.
- **Satellite Imagery**: Satellite imagery, air-quality maps, historical satellite-style data, and comparisons with ground measurements.
- **Historical Analysis**: Date-range analysis, distributions, time series, rolling statistics, correlations, seasonal patterns, thresholds, and custom calculations.
- **Data Download**: Historical data, prediction data, saved user-analysis data, and previews.

## Machine Learning

The application prepares date and weather features, adds seasonal calendar features, handles missing values, scales training data, trains multiple models, evaluates their predictions, and persists trained model dictionaries with pickle.

The model training code currently includes:

- Decision Tree Regressor
- Random Forest Regressor
- Linear Regression
- Logistic Regression wrapped for the project's prediction interface

The main application loads `pm_model.pkl` when available and can train models during application workflows. Retraining can produce `pm_model_retrained.pkl`.

The reproducible training workflow is implemented in `train_and_save_model.py` and uses the Excel source dataset at `attached_assets/edited dataset of kolkata.xlsx`.

No performance numbers are claimed here because this repository README does not define a fixed benchmark result.

## Data Sources & APIs

### Local data

- `data/pm25_dataset.csv` is the primary local dataset loaded by the application.
- `attached_assets/edited dataset of kolkata.xlsx` is used by the training workflow.
- Runtime observations, predictions, and user exports use the `data/daily_observed/`, `data/predictions/`, and `data/user_data/` folders locally. Only `.gitkeep` placeholders for those folders are committed.

### Weather and air quality

- **OpenWeatherMap** provides current weather and forecast requests when `OPENWEATHER_API_KEY` is configured and valid.
- **WAQI** provides current and historical air-quality requests when `WAQI_API_KEY` is configured and valid.
- **Open-Meteo** provides the current-weather and current-PM2.5 fallback when OpenWeatherMap or WAQI is unavailable.

### Satellite and maps

`satellite_data.py` uses NASA Earth imagery URLs for satellite imagery display. It also creates air-quality map images locally and generates historical satellite-style PM2.5 estimates for the analysis view.

### Optional OpenAI module

`openai_analysis.py` contains optional OpenAI-powered trend analysis. The main Streamlit application does not require OpenAI to start, and `OPENAI_API_KEY` is only needed when that optional module is used.

## Tech Stack

- Python 3.11+
- Streamlit
- Pandas
- NumPy
- scikit-learn
- Plotly
- Matplotlib
- Requests
- OpenPyXL
- Statsmodels, retained in the project dependency configuration
- OpenAI Python client for the optional analysis module
- Java, AndroidX, Gradle, and Android WebView for the Android component

## Project Structure

```text
app.py                         Main Streamlit application
run_app.py                     Launcher configured for port 5000
data_handler.py                Dataset loading and preprocessing
model_trainer.py               Model training and evaluation
predictor.py                   Prediction feature preparation and inference
visualizer.py                  Chart construction helpers
weather_api.py                 Weather forecast integration
current_weather_api.py         Current weather and fallback integration
waqi_api.py                    WAQI and air-quality fallback integration
satellite_data.py              Satellite imagery and map functions
train_and_save_model.py        Reproducible training workflow
openai_analysis.py             Optional OpenAI analysis module
pm_model.pkl                   Primary persisted model
pm_model_retrained.pkl         Retraining output model
data/                          Source dataset and local runtime folders
attached_assets/               Training workbook and reference assets
static/                        Web assets
.streamlit/                    Streamlit configuration and styles
android_app/                   Optional Android WebView project
requirements.txt               Python dependencies
pyproject.toml                 Python project metadata and dependencies
uv.lock                        Locked Python dependency resolution
```

## Installation

The verified Windows and VS Code setup is:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run_app.py
```

If PowerShell blocks virtual-environment activation for the current session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

The equivalent direct Streamlit command is:

```powershell
streamlit run app.py --server.port 5000
```

Then open `http://localhost:5000`.

## Environment Variables

Provide your own credentials through environment variables, a local `.env` workflow, or Streamlit secrets managed outside Git. Never commit real values.

```powershell
$env:OPENWEATHER_API_KEY = "your_openweather_api_key"
$env:WAQI_API_KEY = "your_waqi_api_key"
$env:OPENAI_API_KEY = "your_openai_api_key"
```

- `OPENWEATHER_API_KEY`: optional for the OpenWeatherMap provider; Open-Meteo fallback remains available when it is missing or rejected.
- `WAQI_API_KEY`: optional for the WAQI provider; Open-Meteo fallback remains available when it is missing or rejected.
- `OPENAI_API_KEY`: optional; used only by `openai_analysis.py`.

`.env.example` lists the variable names with placeholders. `.env` and Streamlit credential files are excluded by `.gitignore`.

## Model Training

To run the current training workflow:

```powershell
python train_and_save_model.py
```

The script reads:

```text
attached_assets/edited dataset of kolkata.xlsx
```

It cleans and preprocesses the workbook data, trains the configured models, and writes a persisted model output in the project root. The application also supports model training and retraining from its Streamlit workflows.

## Screenshots

Page-specific screenshots are not currently mapped to stable repository paths. Add reviewed screenshots before linking them here:

- Dashboard: `TODO: add a reviewed dashboard screenshot path`
- Data Analysis: `TODO: add a reviewed data-analysis screenshot path`
- Model Training: `TODO: add a reviewed model-training screenshot path`
- PM2.5 Prediction: `TODO: add a reviewed prediction screenshot path`
- Satellite Imagery: `TODO: add a reviewed satellite screenshot path`
- Historical Analysis: `TODO: add a reviewed historical-analysis screenshot path`

The repository currently contains a reference image at `attached_assets/screenshots/youtube_com_shorts_-Eo8G3CYWc.png`, but it is not identified as a page-specific application screenshot.

## Android Application

The `android_app/` directory contains an Android WebView wrapper with Java activities, Android layouts, drawable resources, Gradle configuration, and offline-state handling.

The wrapper is optional and loads a Streamlit deployment or local host through its WebView. The committed Android project does not include `gradlew`, `gradlew.bat`, or `gradle-wrapper.jar`, so one-command Android builds from a fresh clone are not currently guaranteed. Android Studio and a compatible local Gradle/Android SDK setup are required.

## Security

- Never commit API keys, passwords, tokens, or private credentials.
- Provide `OPENWEATHER_API_KEY`, `WAQI_API_KEY`, and optional `OPENAI_API_KEY` through local environment configuration or managed deployment secrets.
- `.env`, Streamlit secrets, and Streamlit credential files are ignored by Git.
- Review generated exports and runtime observations before sharing them publicly.

## Future Scope

Potential future improvements, not current guarantees, include:

- More advanced forecasting models and validation
- Support for more monitoring locations
- Automated scheduled model retraining
- Production deployment and monitoring
- Stronger automated tests for data, APIs, and predictions
- A complete Android Gradle wrapper for reproducible builds
- Curated page-specific screenshots and a project license

## License

This repository currently has no license specified. Add a license file before granting others formal reuse permissions.

## Author

**Shavi Gongale**
