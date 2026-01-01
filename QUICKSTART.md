# Smart City Traffic System - Quick Start Guide

## Installation

1. **Create and activate virtual environment:**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Running the System

### Option 1: Complete Pipeline (Recommended for first time)

Run everything at once:
```bash
python run_pipeline.py
```

This will:
1. Generate synthetic traffic data
2. Process and engineer features
3. Train all ML models (classification + forecasting)
4. Optimize traffic signals
5. Run traffic simulation
6. Evaluate model performance

**Time**: ~10-15 minutes on average hardware

### Option 2: Individual Components

Run each step separately:

```bash
# 1. Generate data
python src/data/generator.py

# 2. Process data
python src/data/processor.py

# 3. Train classification models
python src/models/classification.py

# 4. Train forecasting models
python src/models/forecasting.py

# 5. Optimize signals
python src/optimization/signal_optimizer.py

# 6. Run simulation
python src/optimization/simulator.py

# 7. Evaluate models
python src/models/evaluation.py
```

### Option 3: Interactive Dashboard

Launch the Streamlit dashboard:
```bash
streamlit run src/visualization/dashboard.py
```

Then open your browser to: `http://localhost:8501`

## Dashboard Features

- **📊 Overview**: Key metrics, congestion distribution, traffic patterns
- **📍 Real-Time Map**: Interactive map with live traffic conditions
- **📈 Traffic Analysis**: Segment-level time series analysis
- **🤖 ML Performance**: Model comparison and metrics
- **⚡ Optimization Results**: Signal changes, benefits, ROI

## Configuration

Edit `config.yaml` to customize:
- Number of road segments
- Simulation duration
- Model hyperparameters
- Optimization parameters
- Dashboard settings

## Output Files

After running the pipeline, check:

- `data/raw/` - Generated traffic data
- `data/processed/` - Processed features and splits
- `data/models/` - Trained models and results
- `reports/` - Executive summary and documentation
- `reports/figures/` - Generated visualizations

## Troubleshooting

### Memory Issues
If you encounter memory errors, reduce in `config.yaml`:
- `data_generation.num_road_segments` (default: 120)
- `preprocessing.lag_features` (fewer lag features)

### Performance Issues
For faster execution:
- Reduce `models.xgboost.n_estimators`
- Reduce `models.lstm.epochs`
- Use smaller dataset (fewer months)

### Import Errors
Make sure you're in the virtual environment:
```bash
# Check if venv is active (should see (venv) in prompt)
pip list  # Should show all dependencies
```

## Next Steps

1. ✅ Run the complete pipeline
2. ✅ Explore the dashboard
3. ✅ Review model performance reports
4. ✅ Check executive summary for ROI analysis
5. ✅ Customize configuration for your use case

## Support

For issues or questions:
1. Check the main README.md
2. Review reports/executive_summary.md
3. Examine configuration in config.yaml

---

**Happy Traffic Optimization! 🚦**
