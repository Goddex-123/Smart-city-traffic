"""
Streamlit Dashboard for Smart City Traffic System.
Interactive dashboard for traffic visualization and analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
from pathlib import Path
import pickle
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import get_config, get_logger

logger = get_logger(__name__)

# Page config
st.set_page_config(
    page_title="Smart City Traffic System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
    }
    h1, h2, h3 {
        color: #00d4ff;
    }
    </style>
""", unsafe_allow_html=True)


class TrafficDashboard:
    """Main dashboard class."""
    
    def __init__(self):
        """Initialize dashboard."""
        self.config = get_config()
        self.load_data()
    
    def load_data(self):
        """Load all necessary data."""
        try:
            # Paths
            data_raw = self.config.get_path('data_raw')
            data_processed = self.config.get_path('data_processed')
            models_path = self.config.get_path('models')
            
            # Load raw data
            self.road_network = pd.read_pickle(data_raw / 'road_network.pkl')
            self.traffic_signals = pd.read_pickle(data_raw / 'traffic_signals.pkl')
            
            # Load processed data
            self.traffic_data = pd.read_pickle(data_processed / 'processed_traffic_data.pkl')
            
            # Load optimization results
            try:
                self.optimized_signals = pd.read_pickle(data_processed / 'optimized_signals.pkl')
            except:
                self.optimized_signals = None
            
            # Load simulation results
            try:
                with open(data_processed / 'simulation_results.pkl', 'rb') as f:
                    sim_results = pickle.load(f)
                self.simulation_comparison = sim_results['comparison']
                self.benefits = sim_results['benefits']
            except:
                self.simulation_comparison = None
                self.benefits = None
            
            # Load model results
            try:
                with open(models_path / 'classification_results.pkl', 'rb') as f:
                    self.classification_results = pickle.load(f)
            except:
                self.classification_results = None
            
            logger.success("Data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            st.error(f"Error loading data: {e}")
    
    def render_sidebar(self):
        """Render sidebar navigation."""
        st.sidebar.title("🚦 Smart City Traffic")
        st.sidebar.markdown("---")
        
        page = st.sidebar.radio(
            "Navigation",
            ["📊 Overview", "📍 Real-Time Map", "📈 Traffic Analysis", 
             "🤖 ML Performance", "⚡ Optimization Results"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.info(
            "**Smart City Traffic System**\n\n"
            "AI-powered traffic congestion prediction and optimization.\n\n"
            f"**Segments:** {len(self.road_network)}\n\n"
            f"**Signals:** {len(self.traffic_signals)}\n\n"
            f"**Data Points:** {len(self.traffic_data):,}"
        )
        
        return page
    
    def page_overview(self):
        """Overview page with key metrics."""
        st.title("🚦 Smart City Traffic System - Overview")
        st.markdown("### AI-Powered Congestion Prediction & Signal Optimization")
        st.markdown("---")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Road Segments",
                f"{len(self.road_network)}",
                "Monitored"
            )
        
        with col2:
            st.metric(
                "Traffic Signals",
                f"{len(self.traffic_signals)}",
                "Optimized"
            )
        
        with col3:
            if self.benefits:
                st.metric(
                    "CO₂ Reduction",
                    f"{self.benefits['co2_reduction_pct']:.1f}%",
                    f"{self.benefits['co2_reduction_kg']:.0f} kg"
                )
            else:
                st.metric("CO₂ Reduction", "N/A", "Run optimization")
        
        with col4:
            if self.benefits:
                st.metric(
                    "Economic Benefit",
                    f"${self.benefits['total_economic_benefit_usd']:.0f}",
                    f"Per simulation"
                )
            else:
                st.metric("Economic Benefit", "N/A", "Run optimization")
        
        st.markdown("---")
        
        # Congestion distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Congestion Level Distribution")
            congestion_counts = self.traffic_data['congestion_level'].value_counts()
            
            fig = px.pie(
                values=congestion_counts.values,
                names=congestion_counts.index,
                color=congestion_counts.index,
                color_discrete_map={
                    'Free Flow': '#2ecc71',
                    'Moderate': '#f39c12',
                    'Heavy': '#e74c3c',
                    'Severe': '#8e44ad'
                }
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🚗 Traffic Volume by Road Type")
            road_type_volume = self.traffic_data.merge(
                self.road_network[['segment_id', 'road_type']], 
                on='segment_id'
            ).groupby('road_type')['volume_vehicles'].mean()
            
            fig = px.bar(
                x=road_type_volume.index,
                y=road_type_volume.values,
                labels={'x': 'Road Type', 'y': 'Avg Volume (vehicles/hour)'},
                color=road_type_volume.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Temporal patterns
        st.subheader("⏰ Traffic Patterns Over Time")
        
        hourly_avg = self.traffic_data.groupby('hour').agg({
            'speed_kmh': 'mean',
            'volume_vehicles': 'mean'
        }).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=hourly_avg['hour'], y=hourly_avg['speed_kmh'],
                      name='Avg Speed', line=dict(color='#3498db', width=3)),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=hourly_avg['hour'], y=hourly_avg['volume_vehicles'],
                      name='Avg Volume', line=dict(color='#e74c3c', width=3)),
            secondary_y=True
        )
        
        fig.update_xaxes(title_text="Hour of Day")
        fig.update_yaxes(title_text="Speed (km/h)", secondary_y=False)
        fig.update_yaxes(title_text="Volume (vehicles/hour)", secondary_y=True)
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def page_realtime_map(self):
        """Real-time traffic map."""
        st.title("📍 Real-Time Traffic Map")
        st.markdown("---")
        
        # Create folium map
        center_lat = self.road_network['latitude'].mean()
        center_lon = self.road_network['longitude'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Get latest data for each segment
        latest_data = self.traffic_data.sort_values('timestamp').groupby('segment_id').last()
        
        # Add road segments
        for _, road in self.road_network.iterrows():
            segment_id = road['segment_id']
            
            if segment_id in latest_data.index:
                data = latest_data.loc[segment_id]
                congestion = data['congestion_level']
                speed = data['speed_kmh']
                volume = data['volume_vehicles']
                
                # Color by congestion
                color_map = {
                    'Free Flow': 'green',
                    'Moderate': 'orange',
                    'Heavy': 'red',
                    'Severe': 'darkred'
                }
                color = color_map.get(congestion, 'gray')
                
                # Add marker
                folium.CircleMarker(
                    location=[road['latitude'], road['longitude']],
                    radius=8,
                    popup=f"<b>{segment_id}</b><br>"
                          f"Speed: {speed:.1f} km/h<br>"
                          f"Volume: {volume:.0f} veh/h<br>"
                          f"Status: {congestion}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)
        
        # Add traffic signals
        for _, signal in self.traffic_signals.iterrows():
            folium.Marker(
                location=[signal['latitude'], signal['longitude']],
                popup=f"<b>{signal['signal_id']}</b><br>"
                      f"Cycle: {signal['cycle_length']}s",
                icon=folium.Icon(color='blue', icon='traffic-light', prefix='fa')
            ).add_to(m)
        
        # Display map
        folium_static(m, width=1400, height=600)
        
        # Legend
        st.markdown("""
        **Legend:**
        - 🟢 Green: Free Flow
        - 🟠 Orange: Moderate Congestion
        - 🔴 Red: Heavy Congestion
        - 🟣 Dark Red: Severe Congestion
        - 🔵 Blue Markers: Traffic Signals
        """)
    
    def page_traffic_analysis(self):
        """Traffic analysis page."""
        st.title("📈 Traffic Analysis")
        st.markdown("---")
        
        # Segment selector
        segments = self.traffic_data['segment_id'].unique()
        selected_segment = st.selectbox("Select Road Segment", segments)
        
        # Filter data
        segment_data = self.traffic_data[
            self.traffic_data['segment_id'] == selected_segment
        ].sort_values('timestamp')
        
        # Time series plot
        st.subheader(f"📊 Time Series - {selected_segment}")
        
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=('Speed Over Time', 'Volume Over Time'))
        
        fig.add_trace(
            go.Scatter(x=segment_data['timestamp'], y=segment_data['speed_kmh'],
                      name='Speed', line=dict(color='#3498db')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=segment_data['timestamp'], y=segment_data['volume_vehicles'],
                      name='Volume', line=dict(color='#e74c3c')),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Speed (km/h)", row=1, col=1)
        fig.update_yaxes(title_text="Volume (vehicles/hour)", row=2, col=1)
        fig.update_layout(height=600, showlegend=False)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Speed", f"{segment_data['speed_kmh'].mean():.1f} km/h")
        
        with col2:
            st.metric("Avg Volume", f"{segment_data['volume_vehicles'].mean():.0f} veh/h")
        
        with col3:
            st.metric("Peak Volume", f"{segment_data['volume_vehicles'].max():.0f} veh/h")
        
        with col4:
            congestion_pct = (segment_data['congestion_code'] >= 2).mean() * 100
            st.metric("Congestion %", f"{congestion_pct:.1f}%")
    
    def page_ml_performance(self):
        """ML model performance page."""
        st.title("🤖 Machine Learning Performance")
        st.markdown("---")
        
        if self.classification_results is None:
            st.warning("No model results available. Please train models first.")
            return
        
        # Model comparison
        st.subheader("📊 Classification Model Comparison")
        
        metrics_data = []
        for model_name, results in self.classification_results.items():
            metrics_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        fig = px.bar(metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score'),
                    x='Metric', y='Score', color='Model', barmode='group',
                    color_discrete_sequence=['#3498db', '#e74c3c', '#2ecc71'])
        fig.update_layout(height=400, yaxis_range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        st.subheader("📋 Detailed Metrics")
        st.dataframe(metrics_df.set_index('Model'), use_container_width=True)
        
        # Best model
        best_model = max(self.classification_results.items(), 
                        key=lambda x: x[1]['accuracy'])
        st.success(f"🏆 Best Model: **{best_model[0].replace('_', ' ').title()}** "
                  f"with {best_model[1]['accuracy']:.4f} accuracy")
    
    def page_optimization_results(self):
        """Optimization results page."""
        st.title("⚡ Traffic Signal Optimization Results")
        st.markdown("---")
        
        if self.simulation_comparison is None or self.benefits is None:
            st.warning("No optimization results available. Please run optimization first.")
            return
        
        # Key improvements
        improvements = self.simulation_comparison['improvements']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Wait Time Reduction",
                f"{improvements['wait_time_reduction']:.1f}%",
                f"{improvements['baseline_avg_wait']:.1f}s → {improvements['optimized_avg_wait']:.1f}s"
            )
        
        with col2:
            st.metric(
                "Throughput Increase",
                f"{improvements['throughput_increase']:.1f}%",
                f"{improvements['baseline_throughput']:.0f} → {improvements['optimized_throughput']:.0f} veh/h"
            )
        
        with col3:
            st.metric(
                "Queue Reduction",
                f"{improvements['queue_reduction']:.1f}%",
                "Vehicles"
            )
        
        st.markdown("---")
        
        # Environmental & Economic Benefits
        st.subheader("🌍 Environmental & Economic Benefits")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌱 Environmental Impact")
            st.metric("CO₂ Reduction", 
                     f"{self.benefits['co2_reduction_kg']:.1f} kg",
                     f"{self.benefits['co2_reduction_pct']:.1f}% decrease")
            st.metric("Fuel Saved",
                     f"{self.benefits['fuel_savings_liters']:.1f} liters",
                     f"${self.benefits['fuel_savings_value_usd']:.2f}")
        
        with col2:
            st.markdown("### 💰 Economic Impact")
            st.metric("Time Savings Value",
                     f"${self.benefits['time_savings_value_usd']:.2f}",
                     f"{self.benefits['wait_time_saved_hours']:.1f} hours")
            st.metric("Total Economic Benefit",
                     f"${self.benefits['total_economic_benefit_usd']:.2f}",
                     "Per simulation period")
        
        st.markdown("---")
        
        # Signal changes
        if self.optimized_signals is not None:
            st.subheader("🚦 Signal Timing Changes")
            
            # Prepare comparison data
            comparison_data = []
            for _, signal in self.optimized_signals.iterrows():
                comparison_data.append({
                    'Signal ID': signal['signal_id'],
                    'NS Original (s)': signal['green_time_ns_original'],
                    'NS Optimized (s)': signal['green_time_ns'],
                    'EW Original (s)': signal['green_time_ew_original'],
                    'EW Optimized (s)': signal['green_time_ew'],
                    'Cycle Original (s)': signal['cycle_length_original'],
                    'Cycle Optimized (s)': signal['cycle_length']
                })
            
            comparison_df = pd.DataFrame(comparison_data).head(10)
            st.dataframe(comparison_df, use_container_width=True)
            
            st.info("📝 Showing first 10 signals. Signal timings optimized based on traffic volume and flow patterns.")
    
    def run(self):
        """Run the dashboard."""
        page = self.render_sidebar()
        
        if page == "📊 Overview":
            self.page_overview()
        elif page == "📍 Real-Time Map":
            self.page_realtime_map()
        elif page == "📈 Traffic Analysis":
            self.page_traffic_analysis()
        elif page == "🤖 ML Performance":
            self.page_ml_performance()
        elif page == "⚡ Optimization Results":
            self.page_optimization_results()


def main():
    """Main function to run dashboard."""
    dashboard = TrafficDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
