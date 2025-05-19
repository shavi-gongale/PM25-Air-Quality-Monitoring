import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_historical_data(data):
    """
    Plot historical PM2.5 data.
    
    Args:
        data (pandas.DataFrame): Dataset containing historical PM2.5 data
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    fig = px.line(
        data,
        x='Date',
        y='PM2.5',
        color='Station',
        title='Historical PM2.5 Levels',
        labels={'PM2.5': 'PM2.5 (μg/m³)', 'Date': 'Date'}
    )
    
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='PM2.5 (μg/m³)',
        legend_title='Station',
        hovermode='x unified'
    )
    
    return fig

def plot_correlation_matrix(data):
    """
    Plot correlation matrix for the dataset.
    
    Args:
        data (pandas.DataFrame): Dataset containing PM2.5 and meteorological data
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Select numerical columns for correlation
    numerical_cols = ['Temperature', 'Solar_Radiation', 'Wind_Speed', 'Relative_Humidity', 'PM2.5']
    corr_matrix = data[numerical_cols].corr().round(2)
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title='Correlation Matrix',
        aspect='auto'
    )
    
    fig.update_layout(
        xaxis_title='Parameter',
        yaxis_title='Parameter',
        coloraxis_colorbar=dict(title='Correlation')
    )
    
    return fig

def plot_parameter_comparison(data, param_x, param_y):
    """
    Plot scatter plot comparing two parameters.
    
    Args:
        data (pandas.DataFrame): Dataset containing PM2.5 and meteorological data
        param_x (str): Name of the parameter for x-axis
        param_y (str): Name of the parameter for y-axis
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    fig = px.scatter(
        data,
        x=param_x,
        y=param_y,
        trendline='ols',
        title=f'{param_y} vs {param_x}',
        labels={param_x: param_x, param_y: param_y}
    )
    
    fig.update_layout(
        xaxis_title=param_x,
        yaxis_title=param_y,
        hovermode='closest'
    )
    
    return fig

def plot_model_comparison(evaluation_results):
    """
    Plot model comparison based on evaluation metrics.
    
    Args:
        evaluation_results (dict): Dictionary containing evaluation metrics for each model
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Extract model names and RMSE values
    models = list(evaluation_results.keys())
    rmse_values = [evaluation_results[model]['rmse'] for model in models]
    r2_values = [evaluation_results[model]['r2'] for model in models]
    
    # Create subplots
    fig = go.Figure()
    
    # Add RMSE bars
    fig.add_trace(go.Bar(
        x=models,
        y=rmse_values,
        name='RMSE',
        marker_color='indianred'
    ))
    
    # Add R² values on secondary axis
    fig.add_trace(go.Scatter(
        x=models,
        y=r2_values,
        name='R²',
        mode='markers+lines',
        marker=dict(size=10, color='royalblue'),
        yaxis='y2'
    ))
    
    # Update layout
    fig.update_layout(
        title='Model Comparison: RMSE and R²',
        xaxis_title='Model',
        yaxis=dict(
            title='RMSE (lower is better)',
            side='left'
        ),
        yaxis2=dict(
            title='R² (higher is better)',
            side='right',
            overlaying='y',
            range=[0, 1]
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    return fig

def plot_forecast(predictions_df):
    """
    Plot PM2.5 forecasts from different models.
    
    Args:
        predictions_df (pandas.DataFrame): DataFrame containing predictions from different models
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    # Add a trace for each model
    models = [col for col in predictions_df.columns if col != 'Date']
    
    for model in models:
        fig.add_trace(go.Scatter(
            x=predictions_df['Date'],
            y=predictions_df[model],
            mode='lines+markers',
            name=model
        ))
    
    # Update layout
    fig.update_layout(
        title='PM2.5 Forecast Comparison',
        xaxis_title='Date',
        yaxis_title='PM2.5 (μg/m³)',
        legend_title='Model',
        hovermode='x unified'
    )
    
    return fig
