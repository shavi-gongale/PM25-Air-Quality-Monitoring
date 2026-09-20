# PM2.5 Air Quality Monitoring & Prediction System

## 1. Project Overview

This project is a Streamlit application for monitoring and predicting PM2.5 air-quality levels for Fort William and Bidhannagar in Kolkata, India. It combines local historical observations, trained machine-learning models, current weather and air-quality services, forecasts, visual analysis, and health guidance in one dashboard.

The repository also contains an Android WebView wrapper under `android_app/`.

## 2. Features

- Dashboard with current PM2.5 status and AQI classification
- Weather and air-quality data for the supported stations
- Historical air-quality analysis
- Model training and model comparison
- PM2.5 predictions for multiple forecast horizons
- Forecast visualizations and downloadable data
- Satellite and air-quality map integrations
- Health recommendations based on PM2.5 and AQI levels
- Runtime collection of observed station data
- Optional Android WebView client
- Fallback behavior when external APIs are unavailable

## 3. Application Pages

The Streamlit navigation includes:

- Dashboard
- Analytics
- Models
- Predictions
- Satellite
- Historical
- Downloads

## 4. Machine Learning

The application uses scikit-learn models trained from weather and calendar features. The training pipeline includes preprocessing, seasonal features, scaling, missing-value handling, model training, evaluation, and persisted model output.

The primary persisted model is `pm_model.pkl`. Retraining workflows can create `pm_model_retrained.pkl`.

## 5. Data Sources

- `data/pm25_dataset.csv`: local historical PM2.5 and weather dataset
- `attached_assets/edited dataset of kolkata.xlsx`: source dataset used by the training workflow
- OpenWeatherMap: current weather and forecast data when configured
- WAQI: current air-quality data when configured
- Open-Meteo: public fallback for current weather and PM2.5 data
- Satellite and map services used by `satellite_data.py`
- Runtime observations saved under `data/daily_observed/`

## 6. Technology Stack

- Python 3.11+
- Streamlit
- Pandas and NumPy
- scikit-learn
- Plotly and Matplotlib
- Requests
- OpenPyXL
- OpenAI Python client for the optional analysis module
- Gradle, Java, AndroidX, and WebView for the Android wrapper

## 7. Project Structure

```text
app.py                         Main Streamlit application
run_app.py                     Streamlit launcher for port 5000
data_handler.py                Dataset loading and preprocessing
model_trainer.py               Model training and evaluation
predictor.py                   Prediction preparation and inference
visualizer.py                  Plotly visualization helpers
weather_api.py                 Weather forecast integration
current_weather_api.py         Current weather integration
waqi_api.py                    WAQI air-quality integration
satellite_data.py              Satellite and map integrations
train_and_save_model.py        Reproducible model-training workflow
pm_model.pkl                   Primary persisted model
pm_model_retrained.pkl         Retraining output model
data/                         Datasets and runtime data
static/                       Static web assets
.streamlit/                   Streamlit configuration and styles
attached_assets/              Training source data and project assets
android_app/                  Optional Android WebView application
```

## 8. Requirements

- Windows 10 or later
- Python 3.11 or later
- Internet access for external APIs and satellite services
- API keys for the authenticated providers if live OpenWeatherMap or WAQI data is required
- Android Studio and an Android SDK only when building the Android wrapper

Python dependencies are listed in `requirements.txt`. `pyproject.toml` and `uv.lock` are also retained for Python project and lock-file workflows.

## 9. Windows + VS Code Installation

Open the project folder in VS Code, open a PowerShell terminal, and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks activation for the current session, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

## 10. Virtual Environment Setup

The virtual environment is local to the developer machine and is intentionally excluded from Git. Activate it before installing dependencies or running the application:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 11. API Configuration

Set API keys in the PowerShell session before starting the app:

```powershell
$env:OPENWEATHER_API_KEY = "your_openweather_api_key"
$env:WAQI_API_KEY = "your_waqi_api_key"
$env:OPENAI_API_KEY = "your_openai_api_key"
```

`OPENWEATHER_API_KEY` and `WAQI_API_KEY` are used for their respective live providers. `OPENAI_API_KEY` is optional and is used only by the optional `openai_analysis.py` module. The main Streamlit app does not require OpenAI to start.

See `.env.example` for the variable names. Do not commit real values or local secret files.

## 12. How to Run

Recommended VS Code command:

```powershell
python run_app.py
```

Equivalent command:

```powershell
streamlit run app.py --server.port 5000
```

Open the application at [http://localhost:5000](http://localhost:5000).

## 13. Model Training

To reproduce the persisted model from the Excel training source:

```powershell
python train_and_save_model.py
```

The training workflow reads `attached_assets/edited dataset of kolkata.xlsx` and writes the trained model output in the project root. Keep the training dataset available when reproducing or retraining the model.

## 14. Dataset Information

The application expects PM2.5 observations and weather fields including date, station, temperature, solar radiation, wind speed, relative humidity, and PM2.5. The normal application path loads `data/pm25_dataset.csv`. Missing local data can trigger the built-in synthetic-data fallback.

Runtime observation, prediction, and user-export files are written under:

- `data/daily_observed/`
- `data/predictions/`
- `data/user_data/`

## 15. Usage

1. Start the Streamlit application.
2. Select a station from the navigation controls.
3. Review the dashboard and current PM2.5 status.
4. Use Analytics and Historical pages to inspect stored data.
5. Use Models and Predictions to train, compare, and generate forecasts.
6. Use Satellite and Downloads for environmental imagery and exported results.
7. Configure API keys when authenticated live-provider data is needed.

## 16. Screenshots

The current repository contains project reference images under `attached_assets/`. No separate curated GitHub screenshot gallery is currently configured. Screenshots can be added here later without changing application behavior.

## 17. Future Scope

- Add automated tests for data processing, API adapters, and prediction workflows
- Add a complete root-level screenshot gallery
- Improve reproducible release packaging for the Android wrapper
- Add monitoring and structured logging for external API availability
- Add a clearer separation between source datasets and generated runtime data

## 18. License

No license file is currently included. Choose and add a license before publishing the repository if you want to grant reuse permissions. Until then, the project does not declare an open-source license.
