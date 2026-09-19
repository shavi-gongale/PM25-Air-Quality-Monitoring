import requests
import os
from datetime import datetime
import json
import logging


_LOCATION_COORDS = {
    'Fort William': {'lat': 22.5675, 'lon': 88.3476},
    'Bidhannagar': {'lat': 22.5775, 'lon': 88.4328},
}


def _fallback_current_weather(location):
    """Return stable demo conditions when OpenWeather is not configured."""
    defaults = {
        'Fort William': {'temp': 28.0, 'humidity': 65.0, 'wind_speed': 3.0},
        'Bidhannagar': {'temp': 29.0, 'humidity': 70.0, 'wind_speed': 2.5},
    }
    values = defaults.get(location, defaults['Fort William']).copy()
    current_hour = datetime.now().hour
    hour_factor = max(0, 1 - abs(current_hour - 12) / 12)
    values.update({
        'solar_radiation': round(220 * hour_factor, 1),
        'timestamp': datetime.now().isoformat(),
        'location': location,
        'is_fallback': True,
        'source': 'demo',
    })
    return values

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
        logging.warning(
            "OPENWEATHER_API_KEY is not configured; using demo current weather data."
        )
        return _fallback_current_weather(location)
    
    try:
        # Map location names to coordinates
        if location not in _LOCATION_COORDS:
            logging.warning("Unknown weather location '%s'.", location)
            return None

        coords = _LOCATION_COORDS[location]
        
        # Get current weather data
        current_url = "https://api.openweathermap.org/data/2.5/weather"
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
                'timestamp': datetime.now().isoformat(),
                'location': location,
                'is_fallback': False,
                'source': 'openweathermap',
            }
        else:
            logging.warning(
                "OpenWeather returned HTTP %s; using demo current weather data.",
                response.status_code,
            )
            return _fallback_current_weather(location)
            
    except Exception as e:
        logging.warning(
            "Could not fetch OpenWeather current weather (%s); "
            "using demo current weather data.",
            e,
        )
        return _fallback_current_weather(location)

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