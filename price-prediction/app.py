from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.joblib"

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("🏠 House Price Prediction")
st.write("Enter property details to estimate a price using the trained model.")

if not MODEL_PATH.exists():
    st.warning("Model file not found. Train the model first.")
    st.code("python src/train.py")
    st.stop()


@st.cache_resource
def load_model():
    # Only load model files that you created or otherwise trust.
    return joblib.load(MODEL_PATH)


model = load_model()

col1, col2 = st.columns(2)
with col1:
    area = st.number_input("Area (sq. ft.)", min_value=100, max_value=100000,
                           value=1500, step=50)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=20,
                               value=3, step=1)
with col2:
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=20,
                                value=2, step=1)
    age_years = st.number_input("Property age (years)", min_value=0,
                                max_value=300, value=5, step=1)

if st.button("Predict price", type="primary"):
    features = pd.DataFrame([{
        "area_sqft": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age_years,
    }])
    prediction = float(model.predict(features)[0])
    if prediction <= 0:
        st.warning("The model returned a non-positive estimate. Review your dataset.")
    else:
        st.success(f"Estimated price: ₹{prediction:,.0f}")
    st.caption(
        "Educational estimate only. The sample dataset is synthetic and does not "
        "represent real property values."
    )

st.divider()
st.caption("Model quality depends on the data used for training.")
