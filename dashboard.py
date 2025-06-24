import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Set page config
st.set_page_config(
    page_title="Cotton Price Forecasting Dashboard",
    page_icon="🧶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Load data functions
def load_forecast_data(days):
    """Load forecast data for specified days"""
    forecast_path = f"ensemble_forecast_{days}days.csv"
    if os.path.exists(forecast_path):
        df = pd.read_csv(forecast_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

def load_historical_data():
    """Load historical data"""
    data_path = "combined_dataset.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

def load_model_performance():
    """Load model performance metrics"""
    metrics_path = "subset_results.csv"
    if os.path.exists(metrics_path):
        return pd.read_csv(metrics_path)
    return None

# Dashboard Header
st.title("🧶 Cotton Price Forecasting Dashboard")
st.markdown("""
This dashboard displays forecasts from the LSTM-based cotton price prediction model.
Explore historical trends, model performance, and future projections.
""")

# Load data
with st.spinner('Loading data...'):
    historical_df = load_historical_data()
    performance_df = load_model_performance()
    forecast_7d = load_forecast_data(7)
    forecast_30d = load_forecast_data(30)
    forecast_90d = load_forecast_data(90)

missing = []
if historical_df is None: missing.append("combined_dataset.csv")
if performance_df is None: missing.append("subset_results.csv")
if forecast_7d is None: missing.append("ensemble_forecast_7days.csv")

if missing:
    st.error(f"Missing required files: {', '.join(missing)}")
    st.stop()


# Sidebar controls
st.sidebar.header("Dashboard Controls")
forecast_horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    ["7 days", "30 days", "90 days"],
    index=0
)

show_confidence = st.sidebar.checkbox("Show Confidence Intervals", value=True)
show_historical = st.sidebar.checkbox("Show Historical Data", value=True)
model_to_highlight = st.sidebar.selectbox(
    "Highlight Model Performance",
    performance_df['Subset'].unique(),
    index=0
)

# Main dashboard layout
tab1, tab2, tab3 = st.tabs(["Forecast Visualization", "Model Performance", "Data Exploration"])

with tab1:
    st.header("Price Forecasts")
    
    # Select forecast data based on horizon
    if forecast_horizon == "7 days":
        forecast_df = forecast_7d
    elif forecast_horizon == "30 days":
        forecast_df = forecast_30d
    else:
        forecast_df = forecast_90d
    
    # Create plot
    fig = go.Figure()
    
    if show_historical and historical_df is not None:
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_df['Date'],
            y=historical_df['Cotton_Price'],
            name="Historical Prices",
            line=dict(color='#636EFA'),
            mode='lines'
        ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Forecasted_Cotton_Price'],
        name="Forecast",
        line=dict(color='#FFA15A', width=3),
        mode='lines'
    ))
    
    if show_confidence:
        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['CI_Upper_95'],
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['CI_Lower_95'],
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(255, 161, 90, 0.2)',
            name="95% Confidence"
        ))
    
    # Update layout
    fig.update_layout(
        title=f"Cotton Price Forecast - {forecast_horizon}",
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        hovermode="x unified",
        height=600,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display forecast table
    st.subheader("Forecast Data")
    st.dataframe(
        forecast_df.style.format({
            'Forecasted_Cotton_Price': '{:.2f}',
            'CI_Lower_95': '{:.2f}',
            'CI_Upper_95': '{:.2f}'
        }),
        use_container_width=True
    )

with tab2:
    st.header("Model Performance Metrics")
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Best R-squared",
            value=f"{performance_df['R2'].max():.4f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Lowest MAE",
            value=f"{performance_df['MAE'].min():.2f}",
            delta=None
        )
    
    with col3:
        st.metric(
            "Lowest RMSE",
            value=f"{performance_df['RMSE'].min():.2f}",
            delta=None
        )
    
    with col4:
        st.metric(
            "Selected Model",
            value=model_to_highlight,
            delta=None
        )
    
    # Highlight selected model
    highlight_style = [
        {
            'if': {
                'filter_query': f'{{Subset}} = "{model_to_highlight}"',
                'column_id': list(performance_df.columns)
            },
            'backgroundColor': '#FFF2CC',
            'fontWeight': 'bold'
        }
    ]
    
    st.dataframe(
        performance_df.style.format({
            'MAE': '{:.2f}',
            'MSE': '{:.2f}',
            'RMSE': '{:.2f}',
            'R2': '{:.4f}'
        }).apply(
            lambda x: ['background: #FFF2CC' if x.name == model_to_highlight else '' for i in x],
            axis=1
        ),
        use_container_width=True
    )
    
    # Performance comparison chart
    fig = px.bar(
        performance_df,
        x='Subset',
        y=['MAE', 'RMSE'],
        barmode='group',
        title="Model Performance Comparison",
        labels={'value': 'Error', 'variable': 'Metric'},
        color_discrete_sequence=['#636EFA', '#EF553B']
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Historical Data Exploration")
    
    if historical_df is not None:
        # Date range selector
        min_date = historical_df['Date'].min()
        max_date = historical_df['Date'].max()
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start date",
                min_date,
                min_value=min_date,
                max_value=max_date
            )
        
        with col2:
            end_date = st.date_input(
                "End date",
                max_date,
                min_value=min_date,
                max_value=max_date
            )
        
        # Filter data
        filtered_df = historical_df[
            (historical_df['Date'] >= pd.to_datetime(start_date)) &
            (historical_df['Date'] <= pd.to_datetime(end_date))
        ]
        
        # Display statistics
        st.subheader("Summary Statistics")
        st.dataframe(
            filtered_df['Cotton_Price'].describe().to_frame().T.style.format('{:.2f}'),
            use_container_width=True
        )
        
        # Time series plot
        fig = px.line(
            filtered_df,
            x='Date',
            y='Cotton_Price',
            title="Historical Cotton Prices",
            labels={'Cotton_Price': 'Price (USD)'}
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution plot
        st.subheader("Price Distribution")
        fig = px.histogram(
            filtered_df,
            x='Cotton_Price',
            nbins=50,
            title="Price Distribution",
            labels={'Cotton_Price': 'Price (USD)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Historical data not available")

# Footer
st.markdown("---")
st.markdown("""
**About this dashboard:**
- Forecasts generated using LSTM neural networks
- Data updated after each model run
- Confidence intervals represent 95% prediction intervals
""")
