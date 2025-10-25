import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide"
)

# -------------------- Load Models Function --------------------
@st.cache_resource
def load_models():
    """
    Loads trained models and feature names from the notebooks folder.
    This works whether you run Streamlit from root or from within webapp.
    """
    # Resolve the models directory path
    models_dir = Path(__file__).resolve().parent.parent / "Notebooks"

    # Load models
    rf_model = joblib.load(models_dir / "rf_model.joblib")
    xgb_model = joblib.load(models_dir / "xgb_model.joblib")

    # Load feature names
    feature_file = models_dir / "feature_names.json"
    if feature_file.exists():
        with open(feature_file, "r") as f:
            feature_names = json.load(f)
    else:
        feature_names = []

    return rf_model, xgb_model, feature_names


#  # -------------------- Initialize Models --------------------
try:
    rf_model, xgb_model, feature_names = load_models()
    st.success("✅ Models loaded successfully!")
    models_loaded = True
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    models_loaded = False

# Main app header
st.title("🏢 Building Energy Consumption Predictor")
st.markdown("""
This app predicts building energy consumption based on various features. 
Choose a model and input the building characteristics to get a prediction.
""")

# Sidebar for model selection
model_name = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "XGBoost"]
)

# Create input form
st.subheader("Building Characteristics")

# Create columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    occupants = st.number_input("Number of Occupants", min_value=1, max_value=500, value=100)
    floor_area = st.number_input("Floor Area (sq ft)", min_value=100, max_value=10000, value=2000)

with col2:
    building_type = st.selectbox("Building Type", ["Industrial", "Residential","Commercial"])
    energy_per_occupant = st.number_input("Energy per Occupant", min_value=0.0, max_value=100.0, value=2.0)

with col3:
    energy_per_floor_area = st.number_input("Energy per Floor Area", min_value=0.0, max_value=1.0, value=0.1)
    consumption_ur = st.number_input("Energy UR Consumption", min_value=0.0, max_value=200.0, value=50.0)

# Create feature vector when predict button is clicked
if st.button("Predict Energy Consumption"):
    if not models_loaded:
        st.error("Models not loaded. Cannot make predictions.")
    else:
        # Create a dictionary of features
        features = {
            'Consumption_Energy_Ur_Consumption': consumption_ur,
            'Occupants': occupants,
            'Floor_Area': floor_area,
            'Energy_per_Occupant': energy_per_occupant,
            'Energy_per_FloorArea': energy_per_floor_area,
            'Building_Type_Industrial': 1 if building_type == "Industrial" else 0,
            'Building_Type_Residential': 1 if building_type == "Residential" else 0,
            'Building_Type_Commercial': 1 if building_type == "Commercial" else 0
        }
        
        # Create feature vector
        X = pd.DataFrame([features])
        
        # Reorder columns to match training data
        X = X.reindex(columns=feature_names, fill_value=0)
        
        # Make prediction
        if model_name == "Random Forest":
            prediction = rf_model.predict(X)[0]
        else:  # XGBoost
            prediction = xgb_model.predict(X)[0]
        
        # Display prediction
        st.success(f"Predicted Energy Consumption: {prediction:.2f} units")
        
        # Additional insights
        st.subheader("Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"Energy Efficiency Ratio: {(prediction / floor_area):.3f} units/sq ft")
        
        with col2:
            st.info(f"Per Occupant Consumption: {(prediction / occupants):.3f} units/person")
    
# Add sidebar information
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Instructions
1. Select a model from the dropdown
2. Input building characteristics
3. Click 'Predict' to get results
""")