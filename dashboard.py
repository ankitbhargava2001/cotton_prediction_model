import streamlit as st
import pandas as pd
import numpy as np
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
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning("style.css not found. Using default styling.")

local_css("style.css")

# Load data functions
def load_forecast_data(days):
    """Load forecast data for specified days."""
    forecast_path = f"ensemble_forecast_{days}days.csv"
    if os.path.exists(forecast_path):
        df = pd.read_csv(forecast_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

def load_historical_data():
    """Load historical data."""
    data_path = "combined_dataset.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

def load_model_performance():
    """Load model performance metrics."""
    metrics_path = "subset_results_improved.csv"
    if os.path.exists(metrics_path):
        return pd.read_csv(metrics_path)
    return None

# Dashboard Header
st.title("🧶 Cotton Price Forecasting Dashboard")
st.markdown("""
This dashboard displays forecasts from an LSTM-based cotton price prediction model with an ensemble approach. 
Explore historical trends, model performance, and future projections with interactive timeline filters and graph options.
""")

# Load data
with st.spinner('Loading data...'):
    historical_df = load_historical_data()
    performance_df = load_model_performance()
    forecast_7d = load_forecast_data(7)
    forecast_30d = load_forecast_data(30)
    forecast_90d = load_forecast_data(90)

if historical_df is None or performance_df is None or forecast_7d is None:
    st.error("Data files not found. Please ensure the model pipeline has been run.")
    st.stop()

# Sidebar controls
st.sidebar.header("Dashboard Controls")

# Timeline Filters
st.sidebar.subheader("Timeline Filters")
timeline_filter = st.sidebar.selectbox(
    "Historical Data Timeline",
    ["1 Day", "7 Days", "30 Days", "1 Year", "All Time"],
    index=4,
    help="Filter the historical data displayed in graphs."
)

# Graph Options
st.sidebar.subheader("Graph Options")
show_confidence = st.sidebar.checkbox(
    "Show Confidence Intervals",
    value=True,
    help="Display 95% confidence intervals for ensemble forecasts."
)
show_historical = st.sidebar.checkbox(
    "Show Historical Data",
    value=True,
    help="Include historical cotton prices in forecast plots."
)
show_smoothing = st.sidebar.checkbox(
    "Apply 7-day Moving Average",
    value=False,
    help="Smooth historical data with a 7-day moving average."
)
available_features = [col for col in historical_df.columns if col not in ['Date']]
selected_features = st.sidebar.multiselect(
    "Additional Features to Plot",
    available_features,
    default=[],
    help="Select additional features to plot in the Data Exploration tab."
)

# Forecast Controls
st.sidebar.subheader("Forecast Controls")
forecast_horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    ["7 days", "30 days", "90 days"],
    index=0,
    help="Select the forecast period to display."
)

# Reset View Button
if st.sidebar.button("Reset View", help="Restore default graph view (All Time historical data)."):
    timeline_filter = "All Time"
    st.rerun()

# Filter historical data based on timeline
def filter_historical_data(df, timeline):
    max_date = df['Date'].max()
    if timeline == "1 Day":
        start_date = max_date - timedelta(days=1)
    elif timeline == "7 Days":
        start_date = max_date - timedelta(days=7)
    elif timeline == "30 Days":
        start_date = max_date - timedelta(days=30)
    elif timeline == "1 Year":
        start_date = max_date - timedelta(days=365)
    else:  # All Time
        start_date = df['Date'].min()
    return df[df['Date'] >= start_date]

filtered_historical_df = filter_historical_data(historical_df, timeline_filter)

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
    
    if show_historical and filtered_historical_df is not None:
        y_data = filtered_historical_df['Cotton_Price']
        if show_smoothing:
            y_data = filtered_historical_df['Cotton_Price'].rolling(window=7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=filtered_historical_df['Date'],
            y=y_data,
            name="Historical Prices",
            line=dict(color='#636EFA'),
            mode='lines'
        ))
    
    # Add ensemble forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Ensemble_Forecast'],
        name="Ensemble Forecast",
        line=dict(color='#FFA15A', width=3),
        mode='lines'
    ))
    
    if show_confidence:
        # Add confidence intervals for ensemble forecast
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Upper_CI'],
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Lower_CI'],
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(255, 161, 90, 0.2)',
            name="95% Confidence"
        ))
    
    # Update layout with range slider and buttons
    fig.update_layout(
        title=f"Cotton Price Forecast - {forecast_horizon}",
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        hovermode="x unified",
        height=600,
        template="plotly_white",
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date",
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            )
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Download forecast data
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"Download {forecast_horizon} Forecast Data",
        data=csv,
        file_name=f"ensemble_forecast_{forecast_horizon.replace(' ', '_')}.csv",
        mime="text/csv",
        help="Download the forecast data as a CSV file."
    )
    
    # Display forecast table
    st.subheader("Forecast Data")
    st.dataframe(
        forecast_df.style.format({
            'Ensemble_Forecast': '{:.2f}',
            'Lower_CI': '{:.2f}',
            'Upper_CI': '{:.2f}'
        }),
        use_container_width=True
    )

with tab2:
    st.header("Model Performance Metrics")
    
    # Create metrics columns
    col1, col2, col3 = st.columns(3)
    
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
    
    # Display performance table without highlighting
    st.dataframe(
        performance_df.style.format({
            'MAE': '{:.2f}',
            'MSE': '{:.2f}',
            'RMSE': '{:.2f}',
            'R2': '{:.4f}'
        }),
        use_container_width=True
    )
    
    # Performance comparison chart
    fig = px.bar(
        performance_df,
        x='subset',
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
    
    if filtered_historical_df is not None:
        # Date range selector
        min_date = historical_df['Date'].min()
        max_date = historical_df['Date'].max()
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start date",
                min_date,
                min_value=min_date,
                max_value=max_date,
                help="Select the start date for historical data."
            )
        
        with col2:
            end_date = st.date_input(
                "End date",
                max_date,
                min_value=min_date,
                max_value=max_date,
                help="Select the end date for historical data."
            )
        
        # Filter data by date range
        date_filtered_df = filtered_historical_df[
            (filtered_historical_df['Date'] >= pd.to_datetime(start_date)) &
            (filtered_historical_df['Date'] <= pd.to_datetime(end_date))
        ]
        
        # Download filtered historical data
        csv = date_filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Historical Data",
            data=csv,
            file_name="filtered_historical_data.csv",
            mime="text/csv",
            help="Download the filtered historical data as a CSV file."
        )
        
        # Display statistics
        st.subheader("Summary Statistics")
        st.dataframe(
            date_filtered_df['Cotton_Price'].describe().to_frame().T.style.format('{:.2f}'),
            use_container_width=True
        )
        
        # Time series plot with selected features
        fig = go.Figure()
        for feature in ['Cotton_Price'] + selected_features:
            y_data = date_filtered_df[feature]
            if show_smoothing and feature == 'Cotton_Price':
                y_data = date_filtered_df[feature].rolling(window=7, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=date_filtered_df['Date'],
                y=y_data,
                name=feature,
                mode='lines'
            ))
        
        fig.update_layout(
            title="Historical Data",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode="x unified",
            height=500,
            template="plotly_white",
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date",
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ])
                )
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution plot
        st.subheader("Price Distribution")
        fig = px.histogram(
            date_filtered_df,
            x='Cotton_Price',
            nbins=50,
            title="Cotton Price Distribution",
            labels={'Cotton_Price': 'Price (INR)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Historical data not available")

# Footer
st.markdown("---")
st.markdown("""
**About this dashboard:**
- Forecasts generated using LSTM neural networks with an ensemble approach
- Interactive timeline filters: 1 Day, 7 Days, 30 Days, 1 Year, All Time
- Advanced graph options: zoom, pan, feature selection, smoothing
- Data updated after each model run
- Confidence intervals represent approximate 95% prediction intervals
""")
