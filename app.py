"""
Smart City Traffic Congestion Prediction & Optimization
Streamlit Cloud Deployment
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Smart City Traffic",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Smart City Traffic Prediction System")
st.markdown("*AI-Powered Traffic Congestion Prediction & Signal Optimization*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    city_zone = st.selectbox("Select Zone", ["Downtown", "Industrial", "Residential", "Highway"])
    time_period = st.selectbox("Time Period", ["Morning Rush", "Afternoon", "Evening Rush", "Night"])
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.metric("Active Sensors", "120")
    st.metric("Predictions/Hour", "7,200")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Predictions", "⚡ Optimization", "📈 Analytics"])

with tab1:
    st.header("Real-Time Traffic Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Congestion", "Medium", "↗️ +5%")
    with col2:
        st.metric("Avg Speed", "35 km/h", "↘️ -3%")
    with col3:
        st.metric("Active Incidents", "3")
    with col4:
        st.metric("Optimization Score", "78%")
    
    # Traffic heatmap simulation
    st.subheader("🗺️ Traffic Density Map")
    
    hours = list(range(24))
    segments = [f"Segment {i}" for i in range(1, 11)]
    traffic_data = np.random.randint(20, 100, size=(10, 24))
    
    fig = px.imshow(traffic_data, x=hours, y=segments, 
                    color_continuous_scale='RdYlGn_r',
                    labels=dict(x="Hour", y="Road Segment", color="Congestion %"))
    fig.update_layout(title="24-Hour Traffic Pattern", height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("🔮 Traffic Predictions")
    
    prediction_hours = st.slider("Prediction Horizon (hours)", 1, 24, 6)
    
    if st.button("🎯 Generate Predictions"):
        # Generate sample predictions
        times = pd.date_range(start=datetime.now(), periods=prediction_hours, freq='H')
        predictions = 40 + np.cumsum(np.random.randn(prediction_hours) * 5)
        predictions = np.clip(predictions, 10, 100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=predictions, mode='lines+markers',
                                  name='Predicted Congestion', fill='tozeroy'))
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                      annotation_text="High Congestion Threshold")
        fig.update_layout(title=f"Congestion Forecast - Next {prediction_hours} Hours",
                         yaxis_title="Congestion Level (%)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Peak Congestion", f"{predictions.max():.1f}%")
        with col2:
            st.metric("Model Accuracy", "87.2%")
        with col3:
            st.metric("Alerts Generated", int(np.sum(predictions > 70)))

with tab3:
    st.header("⚡ Signal Optimization")
    
    st.info("Optimize traffic signal timings using AI algorithms")
    
    col1, col2 = st.columns(2)
    with col1:
        optimization_goal = st.selectbox("Optimization Goal", 
            ["Minimize Wait Time", "Maximize Throughput", "Reduce Emissions"])
    with col2:
        num_intersections = st.number_input("Intersections to Optimize", 5, 50, 20)
    
    if st.button("🚀 Run Optimization"):
        with st.spinner("Optimizing signal timings..."):
            import time
            time.sleep(1)
        
        st.success("✅ Optimization Complete!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Wait Time Reduction", "-25.3%")
        with col2:
            st.metric("Throughput Increase", "+18.7%")
        with col3:
            st.metric("CO₂ Reduction", "-18.5%")
        
        # Show sample signal timings
        st.subheader("Optimized Signal Timings")
        timing_data = pd.DataFrame({
            'Intersection': [f'INT-{i:03d}' for i in range(1, 6)],
            'Green (N-S)': np.random.randint(30, 60, 5),
            'Green (E-W)': np.random.randint(30, 60, 5),
            'Cycle Time': np.random.randint(90, 150, 5)
        })
        st.dataframe(timing_data, use_container_width=True)

with tab4:
    st.header("📈 Performance Analytics")
    
    # Model performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Classification Models")
        models = ['Random Forest', 'XGBoost', 'Neural Network']
        accuracy = [85.3, 87.2, 86.1]
        
        fig = px.bar(x=models, y=accuracy, title="Model Accuracy Comparison",
                     labels={'x': 'Model', 'y': 'Accuracy (%)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Optimization Impact")
        metrics = ['Wait Time', 'Throughput', 'Emissions', 'Queue Length']
        improvements = [-25.3, 18.7, -18.5, -22.1]
        colors = ['green' if x < 0 else 'blue' for x in improvements]
        
        fig = px.bar(x=metrics, y=improvements, title="Optimization Results (%)",
                     color=colors, color_discrete_map={'green': 'green', 'blue': 'blue'})
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>🚦 Smart City Traffic System | Educational Demo</p>", 
            unsafe_allow_html=True)
