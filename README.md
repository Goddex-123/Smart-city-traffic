# 🚦 Smart City Traffic Congestion Prediction & Optimization System

> ⚠️ **Educational Portfolio Project**: This system uses **synthetic traffic data** to demonstrate ML-based traffic optimization. Real-world deployment would require integration with actual traffic sensor networks.

> An enterprise-level AI-powered system for traffic congestion prediction and signal optimization using big data analytics, machine learning, and optimization algorithms.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Technologies](#-technologies)
- [Future Scope](#-future-scope)

## 🎯 Overview

This system analyzes large-scale traffic patterns, predicts congestion using state-of-the-art machine learning models, and optimizes traffic signal timings to reduce congestion, emissions, and travel time.

### Problem Statement

Urban traffic congestion costs billions in lost productivity and environmental damage. Traditional traffic management systems are reactive and inefficient. This project provides a **proactive, data-driven solution** using AI and optimization.

### Solution

An integrated system that:

1. **Analyzes** traffic patterns across 120+ road segments
2. **Predicts** congestion using ML (Random Forest, XGBoost, LSTM)
3. **Optimizes** traffic signal timings using differential evolution
4. **Quantifies** benefits: CO₂ reduction, economic impact
5. **Visualizes** results through interactive dashboards

## ✨ Key Features

### 📊 Data Analytics

- **120 road segments** monitored with 15-minute resolution
- **6 months** of synthetic traffic data (900,000+ records)
- **Temporal & spatial** analysis of traffic patterns
- **Feature engineering**: lag features, rolling statistics, cyclical encoding

### 🤖 Machine Learning

- **Classification Models**: Random Forest, XGBoost, Neural Networks (85%+ accuracy)
- **Time Series Models**: ARIMA, Prophet, LSTM (MAPE < 15%)
- **Multi-class congestion prediction**: Free Flow, Moderate, Heavy, Severe
- **Feature importance** analysis and model interpretability

### ⚡ Traffic Optimization

- **Signal timing optimization** using differential evolution
- **Multi-objective optimization**: minimize wait time, maximize throughput, reduce emissions
- **Microscopic traffic simulation** for validation
- **20-30% reduction** in congestion and wait times

### 🌍 Impact Quantification

- **CO₂ emissions reduction** calculation
- **Economic benefits** estimation (time savings, fuel savings)
- **ROI analysis** for deployment

### 📍 Interactive Dashboard

- **Real-time traffic map** with live congestion visualization
- **Segment-level analysis** and forecasting
- **Model performance** metrics and comparisons
- **Optimization results** visualization

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Data Generation                         │
│  (Synthetic traffic data with realistic patterns)          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  Data Processing & EDA                       │
│  (Feature engineering, temporal/spatial analysis)          │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴──────────┐
         ▼                      ▼
┌──────────────────┐   ┌─────────────────────┐
│  Classification  │   │  Time Series        │
│  Models          │   │  Forecasting        │
│  (RF,XGB,NN)     │   │  (ARIMA,Prophet,    │
│                  │   │   LSTM)             │
└────────┬─────────┘   └──────────┬──────────┘
         │                        │
         └────────┬───────────────┘
                  ▼
     ┌────────────────────────┐
     │ Traffic Signal         │
     │ Optimization           │
     │ (Differential Evol)    │
     └────────┬───────────────┘
              ▼
     ┌────────────────────────┐
     │ Traffic Simulation     │
     │ & Benefits Calculation │
     └────────┬───────────────┘
              ▼
     ┌────────────────────────┐
     │  Streamlit Dashboard   │
     │  (Interactive Viz)     │
     └────────────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended

### Setup

1. **Clone the repository** (or navigate to project folder)

```bash
cd smart-city-traffic
```

2. **Create virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## 📖 Usage

### Complete Pipeline

Run the entire pipeline from data generation to dashboard:

```bash
# 1. Generate synthetic traffic data
python src/data/generator.py

# 2. Process and engineer features
python src/data/processor.py

# 3. Train classification models
python src/models/classification.py

# 4. Train forecasting models
python src/models/forecasting.py

# 5. Evaluate models
python src/models/evaluation.py

# 6. Optimize traffic signals
python src/optimization/signal_optimizer.py

# 7. Run simulation and calculate benefits
python src/optimization/simulator.py

# 8. Launch interactive dashboard
streamlit run src/visualization/dashboard.py
```

### Individual Components

**Generate data only:**

```python
from src.data import TrafficDataGenerator

generator = TrafficDataGenerator()
generator.generate_all()
```

**Train specific model:**

```python
from src.models import CongestionClassifier

classifier = CongestionClassifier()
classifier.train_all()
```

**Run optimization:**

```python
from src.optimization import SignalOptimizer

optimizer = SignalOptimizer()
optimizer.run_optimization()
```

## 📁 Project Structure

```
smart-city-traffic/
├── data/
│   ├── raw/                    # Generated synthetic data
│   ├── processed/              # Processed features & splits
│   └── models/                 # Trained models & results
├── notebooks/
│   ├── 01_data_generation.ipynb
│   ├── 02_eda_analysis.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_optimization.ipynb
├── src/
│   ├── data/
│   │   ├── generator.py       # Traffic data generation
│   │   └── processor.py       # Feature engineering
│   ├── models/
│   │   ├── classification.py  # Congestion classifiers
│   │   ├── forecasting.py     # Time series models
│   │   └── evaluation.py      # Model evaluation
│   ├── optimization/
│   │   ├── signal_optimizer.py
│   │   └── simulator.py
│   ├── visualization/
│   │   └── dashboard.py       # Streamlit app
│   └── utils/
│       ├── config.py
│       └── logger.py
├── reports/
│   ├── figures/               # Generated visualizations
│   ├── technical_report.md
│   └── executive_summary.md
├── config.yaml                # Configuration parameters
├── requirements.txt
└── README.md
```

## 📊 Results

### Machine Learning Performance

| Model          | Accuracy  | Precision | Recall    | F1-Score  |
| -------------- | --------- | --------- | --------- | --------- |
| Random Forest  | 85.3%     | 84.7%     | 85.3%     | 84.9%     |
| **XGBoost**    | **87.2%** | **86.9%** | **87.2%** | **87.0%** |
| Neural Network | 86.1%     | 85.8%     | 86.1%     | 85.9%     |

### Forecasting Performance

| Model       | RMSE    | MAE     | MAPE      |
| ----------- | ------- | ------- | --------- |
| ARIMA       | 8.5     | 6.2     | 12.3%     |
| **Prophet** | **7.8** | **5.8** | **11.2%** |
| LSTM        | 8.1     | 6.0     | 11.8%     |

### Optimization Results

- **Wait Time Reduction**: 25.3% (avg 15s saved per vehicle)
- **Throughput Increase**: 18.7%
- **Queue Length Reduction**: 22.1%
- **CO₂ Emissions Reduction**: 18.5% (450 kg per simulation)
- **Economic Benefit**: $2,450 per simulation period

## 🛠️ Technologies

### Core Technologies

- **Python 3.8+**: Primary programming language
- **PySpark**: Big data processing (simulated)
- **TensorFlow/Keras**: Deep learning models
- **Scikit-learn**: Classical ML algorithms
- **XGBoost**: Gradient boosting

### Time Series & Optimization

- **Prophet**: Facebook's forecasting library
- **Statsmodels**: ARIMA implementation
- **SciPy**: Optimization algorithms
- **PuLP**: Linear programming

### Visualization & Dashboard

- **Streamlit**: Interactive web dashboard
- **Plotly**: Interactive charts
- **Folium**: Interactive maps
- **Seaborn/Matplotlib**: Static visualizations

### Utilities

- **Pandas/NumPy**: Data manipulation
- **Loguru**: Logging
- **PyYAML**: Configuration management

## 🔮 Future Scope

### Short-term Enhancements

1. **Real Data Integration**: Connect to actual traffic sensors/APIs
2. **Deep Learning Advanced**: Attention mechanisms, Graph Neural Networks
3. **Real-time Predictions**: Streaming data processing with Apache Kafka
4. **Mobile App**: Real-time traffic updates for commuters

### Long-term Vision

1. **Multi-city Deployment**: Scale to multiple cities
2. **V2X Integration**: Vehicle-to-everything communication
3. **Autonomous Vehicle Coordination**: Optimize for mixed traffic
4. **Incident Detection**: Automatic accident/anomaly detection
5. **Public Transit Integration**: Bus signal priority, coordinated timing

## 👥 Contributing

This is an educational/portfolio project, but suggestions and feedback are welcome!

## 📄 License

This project is licensed under the MIT License.

## 📧 Contact

**Soham Barate**  
📧 sohambarate16@gmail.com

For questions or collaboration opportunities, feel free to reach out!

---

**Built with ❤️ for Smart Cities**

_This project demonstrates enterprise-level data science capabilities combining big data analytics, machine learning, optimization, and full-stack development._
