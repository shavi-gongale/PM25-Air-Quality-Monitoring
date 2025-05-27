import json
import os
from openai import OpenAI

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai = OpenAI(api_key=OPENAI_API_KEY)

def analyze_air_quality_trends(data, station):
    """
    Use OpenAI to analyze air quality trends and provide insights
    
    Args:
        data (dict): Air quality data for analysis
        station (str): Station name
        
    Returns:
        dict: AI-generated insights and recommendations
    """
    if not OPENAI_API_KEY:
        raise Exception("OpenAI API key not found. Please provide OPENAI_API_KEY.")
    
    prompt = f"""
    Analyze the following air quality data for {station} and provide insights:
    
    Air Quality Data:
    - PM2.5: {data.get('pm25', 'N/A')} μg/m³
    - AQI: {data.get('aqi', 'N/A')}
    - Temperature: {data.get('temperature', 'N/A')}°C
    - Humidity: {data.get('humidity', 'N/A')}%
    - Wind Speed: {data.get('wind_speed', 'N/A')} km/h
    
    Please provide:
    1. Health impact assessment
    2. Recommendations for residents
    3. Trend analysis
    4. Weather correlation insights
    
    Respond in JSON format with keys: health_impact, recommendations, trend_analysis, weather_correlation
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an air quality expert providing analysis and recommendations. Respond only in valid JSON format."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=800
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        raise Exception(f"Error generating AI analysis: {e}")

def generate_health_recommendations(pm25_level, aqi_level):
    """
    Generate health recommendations based on PM2.5 and AQI levels
    
    Args:
        pm25_level (float): PM2.5 concentration
        aqi_level (float): Air Quality Index
        
    Returns:
        dict: Health recommendations and activity suggestions
    """
    if not OPENAI_API_KEY:
        raise Exception("OpenAI API key not found. Please provide OPENAI_API_KEY.")
    
    prompt = f"""
    Based on PM2.5 level of {pm25_level} μg/m³ and AQI of {aqi_level}, provide specific health recommendations.
    
    Include:
    1. Activity recommendations (outdoor activities, exercise, etc.)
    2. Vulnerable groups advice (children, elderly, asthma patients)
    3. Protective measures (masks, air purifiers, etc.)
    4. Timing suggestions (best/worst times for outdoor activities)
    
    Respond in JSON format with keys: activities, vulnerable_groups, protective_measures, timing
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a health expert providing air quality-based recommendations. Respond only in valid JSON format."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=600
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        raise Exception(f"Error generating health recommendations: {e}")

def predict_air_quality_trends(historical_data, station):
    """
    Use AI to analyze historical data and predict trends
    
    Args:
        historical_data (list): List of historical air quality measurements
        station (str): Station name
        
    Returns:
        dict: AI predictions and trend analysis
    """
    if not OPENAI_API_KEY:
        raise Exception("OpenAI API key not found. Please provide OPENAI_API_KEY.")
    
    # Summarize the data for the prompt
    data_summary = f"Historical PM2.5 data for {station}: "
    if historical_data:
        avg_pm25 = sum([d.get('PM2.5', 0) for d in historical_data]) / len(historical_data)
        max_pm25 = max([d.get('PM2.5', 0) for d in historical_data])
        min_pm25 = min([d.get('PM2.5', 0) for d in historical_data])
        data_summary += f"Average: {avg_pm25:.1f}, Max: {max_pm25:.1f}, Min: {min_pm25:.1f}"
    
    prompt = f"""
    Analyze the following air quality trends for {station}:
    
    {data_summary}
    
    Based on this data, provide:
    1. Short-term prediction (next 3 days)
    2. Weekly trend forecast
    3. Seasonal patterns identified
    4. Factors likely influencing the trends
    
    Respond in JSON format with keys: short_term, weekly_trend, seasonal_patterns, influencing_factors
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an environmental data analyst. Provide air quality trend predictions based on historical data. Respond only in valid JSON format."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=700
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        raise Exception(f"Error generating trend predictions: {e}")

def compare_stations(station1_data, station2_data):
    """
    Use AI to compare air quality between two stations
    
    Args:
        station1_data (dict): Air quality data for first station
        station2_data (dict): Air quality data for second station
        
    Returns:
        dict: Comparative analysis
    """
    if not OPENAI_API_KEY:
        raise Exception("OpenAI API key not found. Please provide OPENAI_API_KEY.")
    
    prompt = f"""
    Compare air quality between these two stations:
    
    Station 1 - {station1_data.get('station', 'Fort William')}:
    PM2.5: {station1_data.get('pm25', 'N/A')}, AQI: {station1_data.get('aqi', 'N/A')}
    
    Station 2 - {station2_data.get('station', 'Bidhannagar')}:
    PM2.5: {station2_data.get('pm25', 'N/A')}, AQI: {station2_data.get('aqi', 'N/A')}
    
    Provide:
    1. Which station has better air quality
    2. Significant differences and possible reasons
    3. Recommendations for each location
    
    Respond in JSON format with keys: better_station, differences, recommendations
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an environmental analyst comparing air quality data. Respond only in valid JSON format."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=500
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        raise Exception(f"Error generating station comparison: {e}")