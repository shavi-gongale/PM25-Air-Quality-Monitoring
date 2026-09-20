import requests
import os
from datetime import datetime
import json
import logging


_LOCATION_COORDS = {
    'Fort William': {'lat': 22.5675, 'lon': 88.3476},
    'Bidhannagar': {'lat': 22.5775, 'lon': 88.4328},
}


def _get_open_meteo_weather(location):
    """Get current weather from Open-Meteo when OpenWeather is unavailable."""
    coords = _LOCATION_COORDS.get(location)
    if not coords:
        return None

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            'latitude': coords['lat'],
            'longitude': coords['lon'],
            'current': 'temperature_2m,relative_humidity_2m,wind_speed_10m',
            'wind_speed_unit': 'ms',
            'timezone': 'auto',
        },
        timeout=10,
    )
    response.raise_for_status()
    current = response.json()['current']
    current_hour = datetime.now().hour
    solar_radiation = 0 if current_hour < 6 or current_hour > 18 else 400
    return {
        'temp': current['temperature_2m'],
        'humidity': current['relative_humidity_2m'],
        'wind_speed': current['wind_speed_10m'],
        'solar_radiation': solar_radiation,
        'timestamp': datetime.now().isoformat(),
        'location': location,
        'source': 'Open-Meteo',
    }

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
        logging.error(
            "OpenWeather current weather unavailable: "
            "OPENWEATHER_API_KEY is not configured."
        )
        try:
            return _get_open_meteo_weather(location)
        except (requests.exceptions.RequestException, KeyError, TypeError, ValueError) as e:
            logging.error("Open-Meteo current weather request failed: %s", e)
            return None
    
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
            }
        else:
            try:
                error_message = response.json().get('message', response.text[:200])
            except (ValueError, AttributeError):
                error_message = response.text[:200]
            logging.error(
                "OpenWeather current weather request failed: HTTP %s - %s",
                response.status_code, error_message
            )
            try:
                return _get_open_meteo_weather(location)
            except (requests.exceptions.RequestException, KeyError, TypeError, ValueError) as e:
                logging.error("Open-Meteo current weather request failed: %s", e)
                return None
            
    except requests.exceptions.RequestException as e:
        logging.error("OpenWeather current weather request failed: %s", e)
        try:
            return _get_open_meteo_weather(location)
        except (requests.exceptions.RequestException, KeyError, TypeError, ValueError) as fallback_error:
            logging.error("Open-Meteo current weather request failed: %s", fallback_error)
            return None
    except (KeyError, TypeError, ValueError) as e:
        logging.error("OpenWeather returned an unexpected response: %s", e)
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