import requests
import os
import json
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_weather_forecast(location):
    """
    Fetch weather forecast data from OpenWeatherMap API.
    
    Args:
        location (str): Location name (e.g., 'Fort William', 'Bidhannagar')
        
    Returns:
        list: List of dictionaries containing forecast data for the next 7 days
    """
    # Get API key from environment variable
    api_key = os.getenv("OPENWEATHER_API_KEY", "")
    
    if not api_key:
        logging.warning("OpenWeather API key not found. Using mock data for demonstration.")
        return generate_mock_forecast(location)
    
    # Map location to city and country for API request
    location_mapping = {
        'Kolkata': {'city': 'Kolkata', 'country': 'IN', 'lat': 22.5726, 'lon': 88.3639},  # Main Kolkata location
        'Fort William': {'city': 'Kolkata', 'country': 'IN', 'lat': 22.5726, 'lon': 88.3639},  # Fort William is in Kolkata
        'Bidhannagar': {'city': 'Bidhannagar', 'country': 'IN', 'lat': 22.5839, 'lon': 88.4403}
    }
    
    if location not in location_mapping:
        logging.error(f"Location '{location}' not found in location mapping.")
        return None
    
    location_info = location_mapping[location]
    
    # Construct API URL for 5-day forecast (OpenWeatherMap free tier limitation)
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={location_info['lat']}&lon={location_info['lon']}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        forecast_data = response.json()
        
        # Process the forecast data into the required format
        processed_forecast = process_forecast_data(forecast_data)
        
        return processed_forecast
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching weather forecast: {e}")
        return generate_mock_forecast(location)

def process_forecast_data(forecast_data):
    """
    Process the raw forecast data from OpenWeatherMap API.
    
    Args:
        forecast_data (dict): Raw forecast data from OpenWeatherMap API
        
    Returns:
        list: Processed forecast data for the next 7 days
    """
    processed_data = []
    today = datetime.now().date()
    
    # For each forecast day (up to 5 days from OpenWeatherMap free tier)
    for i in range(5):
        forecast_date = today + timedelta(days=i)
        forecast_date_str = forecast_date.strftime("%Y-%m-%d")
        
        # Filter forecast entries for this day
        day_entries = [entry for entry in forecast_data['list'] 
                       if datetime.fromtimestamp(entry['dt']).date() == forecast_date]
        
        if day_entries:
            # Calculate daily averages
            temp_sum = sum(entry['main']['temp'] for entry in day_entries)
            humidity_sum = sum(entry['main']['humidity'] for entry in day_entries)
            wind_sum = sum(entry['wind']['speed'] for entry in day_entries)
            
            # Estimate solar radiation from cloud cover and time of day
            solar_values = []
            for entry in day_entries:
                hour = datetime.fromtimestamp(entry['dt']).hour
                cloud_cover = entry['clouds']['all']  # percentage
                
                # Simple model: max radiation at noon, scaled by cloud cover
                hour_factor = 1 - abs(hour - 12) / 12  # 1 at noon, 0 at midnight
                cloud_factor = 1 - cloud_cover / 100  # 1 for clear sky, 0 for fully cloudy
                
                # Base value around 1000 W/m² for clear noon
                solar_estimate = 1000 * hour_factor * cloud_factor
                solar_values.append(solar_estimate)
            
            solar_avg = sum(solar_values) / len(solar_values) if solar_values else 0
            
            # Create entry for this day
            processed_data.append({
                'Date': forecast_date_str,
                'Temperature': temp_sum / len(day_entries),
                'Solar_Radiation': solar_avg,
                'Wind_Speed': wind_sum / len(day_entries),
                'Relative_Humidity': humidity_sum / len(day_entries)
            })
    
    # For days beyond the 5-day forecast, use extrapolation
    for i in range(5, 8):  # Days 6 and 7
        forecast_date = today + timedelta(days=i)
        forecast_date_str = forecast_date.strftime("%Y-%m-%d")
        
        # Simple extrapolation: use the average of days 3-5 with some random variation
        if len(processed_data) >= 3:
            base_temp = sum(entry['Temperature'] for entry in processed_data[-3:]) / 3
            base_solar = sum(entry['Solar_Radiation'] for entry in processed_data[-3:]) / 3
            base_wind = sum(entry['Wind_Speed'] for entry in processed_data[-3:]) / 3
            base_humidity = sum(entry['Relative_Humidity'] for entry in processed_data[-3:]) / 3
            
            # Add some random variation (±10%)
            import random
            temp_var = base_temp * (0.9 + 0.2 * random.random())
            solar_var = base_solar * (0.9 + 0.2 * random.random())
            wind_var = base_wind * (0.9 + 0.2 * random.random())
            humidity_var = base_humidity * (0.9 + 0.2 * random.random())
            
            # Ensure humidity stays in valid range
            humidity_var = min(100, max(0, humidity_var))
            
            processed_data.append({
                'Date': forecast_date_str,
                'Temperature': temp_var,
                'Solar_Radiation': solar_var,
                'Wind_Speed': wind_var,
                'Relative_Humidity': humidity_var
            })
    
    return processed_data

def generate_mock_forecast(location):
    """
    Generate mock weather forecast data when API is unavailable.
    
    Args:
        location (str): Location name
        
    Returns:
        list: List of dictionaries containing mock forecast data
    """
    mock_data = []
    today = datetime.now().date()
    
    # Base values for different locations
    if location == 'Fort William':
        temp_base = 28
        solar_base = 220
        wind_base = 3.0
        humidity_base = 65
    else:  # Bidhannagar
        temp_base = 29
        solar_base = 230
        wind_base = 2.5
        humidity_base = 70
    
    # Generate forecast for 7 days
    for i in range(7):
        forecast_date = today + timedelta(days=i)
        forecast_date_str = forecast_date.strftime("%Y-%m-%d")
        
        # Add some random variation and seasonal pattern
        day_of_year = forecast_date.timetuple().tm_yday
        season_factor = 0.5 * (1 + 0.5 * (forecast_date.month % 12 / 6))
        
        import random
        temp_var = temp_base * season_factor + random.normalvariate(0, 2)
        solar_var = solar_base * season_factor + random.normalvariate(0, 30)
        wind_var = wind_base + random.normalvariate(0, 0.5)
        humidity_var = humidity_base + random.normalvariate(0, 5)
        
        # Ensure values are in reasonable ranges
        solar_var = max(0, solar_var)
        wind_var = max(0.5, wind_var)
        humidity_var = min(100, max(30, humidity_var))
        
        mock_data.append({
            'Date': forecast_date_str,
            'Temperature': temp_var,
            'Solar_Radiation': solar_var,
            'Wind_Speed': wind_var,
            'Relative_Humidity': humidity_var
        })
    
    return mock_data
