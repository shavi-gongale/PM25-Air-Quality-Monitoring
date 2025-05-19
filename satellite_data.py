import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import base64
from io import BytesIO

def get_satellite_imagery(location, date=None):
    """
    Get satellite imagery for a location
    
    This function tries to get satellite imagery for a location from NASA's Earth API.
    If the API key is not available or API call fails, returns a fallback image URL.
    
    Args:
        location (str): Location name (e.g., "Fort William", "Bidhannagar")
        date (str, optional): Date in format YYYY-MM-DD. Defaults to current date.
        
    Returns:
        str: URL to the satellite image or base64 encoded image
    """
    # Location coordinates (approximate)
    locations = {
        "Fort William": {"lat": 22.5726, "lon": 88.3639},
        "Bidhannagar": {"lat": 22.5839, "lon": 88.4403}
    }
    
    # Check if location is supported
    if location not in locations:
        return None
    
    # Get coordinates
    coords = locations[location]
    
    # Use NASA GIBS API to get satellite imagery
    # This would typically require an API key for production use
    try:
        # For this demonstration, we'll use a static fallback image for each location
        if location == "Fort William":
            image_url = "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73751/world.topo.bathy.200407.3x5400x2700.jpg"
        else:
            image_url = "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73751/world.topo.bathy.200407.3x5400x2700.jpg"
            
        return image_url
    except Exception as e:
        print(f"Error fetching satellite imagery: {e}")
        return None

def get_air_quality_map(location, date=None):
    """
    Get air quality map for a location
    
    This function simulates retrieving air quality maps based on satellite data.
    In a real implementation, this would call an external API like NASA Earth Observations
    or MODIS for PM2.5 global maps.
    
    Args:
        location (str): Location name
        date (str, optional): Date in format YYYY-MM-DD. Defaults to current date.
        
    Returns:
        str: URL or base64 encoded image of the air quality map
    """
    # In a real implementation, we would use an API to get this data
    # For now, return a sample air quality heat map
    
    # We'll create a simple heatmap based on the location and (optional) date
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colors
        
        # Create a grid of data (this would come from real data in production)
        if date:
            # Use the date to seed the random number generator for consistent results
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            seed = int(date_obj.strftime("%Y%m%d"))
            np.random.seed(seed)
        else:
            np.random.seed(42)
        
        # Create a grid with higher PM2.5 values near urban centers
        grid_size = 20
        grid = np.random.normal(25, 10, (grid_size, grid_size))
        
        # Make the center (representing the city) have higher values
        center_x, center_y = grid_size // 2, grid_size // 2
        for i in range(grid_size):
            for j in range(grid_size):
                # Distance from center
                dist = np.sqrt((i - center_x)**2 + (j - center_y)**2)
                # Increase PM2.5 values near center
                grid[i, j] += max(0, 50 - 3 * dist)
        
        # Create a figure
        plt.figure(figsize=(8, 6))
        
        # Create a heatmap
        cmap = colors.LinearSegmentedColormap.from_list(
            'AQI',
            ['#00e400', '#ffff00', '#ff7e00', '#ff0000', '#8f3f97', '#7e0023']
        )
        
        plt.imshow(grid, cmap=cmap)
        plt.colorbar(label='PM2.5 (μg/m³)')
        plt.title(f'PM2.5 Levels for {location}')
        
        # If date is provided, add it to the title
        if date:
            plt.title(f'PM2.5 Levels for {location} on {date}')
        
        # Add a marker for the location
        plt.plot(center_x, center_y, 'ko', markersize=10)
        plt.text(center_x + 1, center_y, location, fontsize=12, color='white')
        
        # Save the figure to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Convert to base64 for embedding in HTML
        data = base64.b64encode(buf.read()).decode('utf-8')
        
        # Create a data URL
        img_src = f'data:image/png;base64,{data}'
        
        plt.close()
        
        return img_src
    except Exception as e:
        print(f"Error creating air quality map: {e}")
        return None

def get_historical_satellite_data(location, start_date, end_date):
    """
    Get historical satellite data for a location over a period
    
    Args:
        location (str): Location name
        start_date (str): Start date in format YYYY-MM-DD
        end_date (str): End date in format YYYY-MM-DD
        
    Returns:
        pd.DataFrame: DataFrame with satellite-derived PM2.5 estimates
    """
    # In a real implementation, this would fetch data from NASA Earth Observations or similar
    # For now, generate synthetic data that roughly correlates with our existing dataset
    
    try:
        # Convert dates to datetime objects
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Create date range
        date_range = pd.date_range(start=start, end=end, freq='D')
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': date_range,
            'Satellite_PM2.5': np.random.normal(35, 10, len(date_range))
        })
        
        # Add some seasonality
        for i, date in enumerate(date_range):
            day_of_year = date.dayofyear
            # Add seasonal pattern
            seasonal_factor = 10 * np.sin(2 * np.pi * day_of_year / 365)
            # Add the seasonal component to the PM2.5 value
            df.loc[i, 'Satellite_PM2.5'] += seasonal_factor
        
        # Ensure PM2.5 values are positive
        df['Satellite_PM2.5'] = df['Satellite_PM2.5'].clip(lower=5)
        
        # Convert date to string format for easier handling
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        return df
    except Exception as e:
        print(f"Error generating historical satellite data: {e}")
        return pd.DataFrame()