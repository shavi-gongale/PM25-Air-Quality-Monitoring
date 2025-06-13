import requests
import os
from datetime import datetime
import json

def get_current_weather_data(location):
    """
    Get current observed weather data from OpenWeatherMap API
    
    Args:
        location (str): Location name (e.g., 'Fort William', 'Bidhannagar')
        
    Returns:
        dict: Current weather data or None if unavailable
    """
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not api_key:
        return None
    
    try:
        # Map location names to coordinates
        location_coords = {
            'Fort William': {'lat': 22.5675, 'lon': 88.3476},
            'Bidhannagar': {'lat': 22.5775, 'lon': 88.4328}
        }
        
        if location not in location_coords:
            return None
        
        coords = location_coords[location]
        
        # Get current weather data
        current_url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': coords['lat'],
            'lon': coords['lon'],
            'appid': api_key,
            'units': 'metric'
        }
        
        response = requests.get(current_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Calculate solar radiation estimate from cloud cover and time
            current_hour = datetime.now().hour
            cloud_cover = data['clouds']['all']  # percentage
            
            # Simple solar radiation model
            if 6 <= current_hour <= 18:  # Daylight hours
                base_solar = 800 * (1 - abs(current_hour - 12) / 6)  # Peak at noon
                solar_radiation = base_solar * (1 - cloud_cover / 100)
            else:
                solar_radiation = 0
            
            return {
                'temp': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'wind_speed': data['wind']['speed'],
                'solar_radiation': max(0, solar_radiation),
                'timestamp': datetime.now().isoformat()
            }
        else:
            return None
            
    except Exception as e:
        print(f"Error fetching current weather: {e}")
        return None

def get_current_uv_index(location):
    """
    Get current UV index data from OpenWeatherMap API
    
    Args:
        location (str): Location name
        
    Returns:
        float: UV index or None if unavailable
    """
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not api_key:
        return None
    
    try:
        location_coords = {
            'Fort William': {'lat': 22.5675, 'lon': 88.3476},
            'Bidhannagar': {'lat': 22.5775, 'lon': 88.4328}
        }
        
        if location not in location_coords:
            return None
        
        coords = location_coords[location]
        
        uv_url = f"http://api.openweathermap.org/data/2.5/uvi"
        params = {
            'lat': coords['lat'],
            'lon': coords['lon'],
            'appid': api_key
        }
        
        response = requests.get(uv_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('value', 0)
        else:
            return None
            
    except Exception as e:
        print(f"Error fetching UV index: {e}")
        return None