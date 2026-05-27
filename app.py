import json
import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model


def cached_load(model_path: str, scaler_path: str):
    """Load model and scaler with simple caching and friendly errors."""
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load model '{model_path}': {e}")
        st.stop()

    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        st.error(f"Failed to load scaler '{scaler_path}': {e}")
        st.stop()

    return model, scaler


# Page config
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="centered")

# Header
st.markdown("""
<div style='display:flex;align-items:center;gap:12px'>
  <div style='font-size:34px'>🎓</div>
  <div>
    <h1 style='margin:0'>Student Performance Predictor</h1>
    <div style='color:gray'>AI-powered pass/fail probability with practical tips</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.write("---")

# Sidebar
with st.sidebar:
    st.header("About")
    st.write("This demo predicts whether a student will pass (G3 >= 10) using a small ANN trained on student data.")
    st.write("Model: ANN • Inputs: studytime, absences, failures, G1, G2")
    st.write("Use the presets to try common student profiles.")


# Paths (adjust if needed)
MODEL_PATH = "student_ann_model.h5"
SCALER_PATH = "scaler.pkl"

# Load resources
model, scaler = cached_load(MODEL_PATH, SCALER_PATH)


# Presets
presets = {
    "Average Student": {"studytime": 2, "absences": 5, "failures": 0, "G1": 10, "G2": 11},
    "At-risk Student": {"studytime": 1, "absences": 20, "failures": 2, "G1": 6, "G2": 5},
    "High Performer": {"studytime": 4, "absences": 0, "failures": 0, "G1": 15, "G2": 16},
}

col1, col2 = st.columns([2, 1])

with col1:
    preset = st.selectbox("Choose a preset", ["Custom"] + list(presets.keys()), key="preset_select")
    if preset != "Custom":
        values = presets[preset]
        studytime = st.slider(
            "Study Time",
            1,
            5,
            value=values["studytime"],
            key="studytime_slider_preset",
        )
        absences = st.number_input(
            "Absences",
            min_value=0,
            max_value=100,
            value=values["absences"],
            key="absences_input_preset",
        )
        failures = st.slider(
            "Past Failures",
            0,
            5,
            value=values["failures"],
            key="failures_slider_preset",
        )
        G1 = st.slider(
            "Grade G1",
            0,
            20,
            value=values["G1"],
            key="g1_slider_preset",
        )
        G2 = st.slider(
            "Grade G2",
            0,
            20,
            value=values["G2"],
            key="g2_slider_preset",
        )
    else:
        studytime = st.slider("Study Time", 1, 5, 2, key="studytime_slider_custom")
        absences = st.number_input(
            "Absences",
            min_value=0,
            max_value=100,
            value=5,
            key="absences_input_custom",
        )
        failures = st.slider("Past Failures", 0, 5, 0, key="failures_slider_custom")
        G1 = st.slider("Grade G1", 0, 20, 10, key="g1_slider_custom")
        G2 = st.slider("Grade G2", 0, 20, 10, key="g2_slider_custom")

with col2:
    st.subheader("Quick Metrics")
    st.metric(label="Study Time (1-5)", value=studytime)
    st.metric(label="Absences", value=absences)


def predict_pass_prob(model, scaler, features: np.ndarray) -> float:
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled, verbose=0)
    return float(pred[0][0])


if st.button("Predict Result"):
    input_data = np.array([[studytime, absences, G1, G2, failures]])
    probability = predict_pass_prob(model, scaler, input_data)
    percent = int(probability * 100)

    # Result card
    if probability >= 0.5:
        st.success(f"✅ Pass likely — {percent}% confidence")
    else:
        st.error(f"❌ Fail likely — {percent}% confidence")

    st.progress(percent)

    # Detailed view
    with st.expander("View details & guidance"):
        st.write("**Prediction details**")
        st.write({
            "studytime": studytime,
            "absences": absences,
            "failures": failures,
            "G1": G1,
            "G2": G2,
            "probability": probability,
        })

        st.write("---")
        if probability < 0.6:
            st.write("**Actionable suggestions**")
            st.write("- Increase study time and structured study sessions.")
            st.write("- Reduce unexcused absences and improve class participation.")
            st.write("- Remedial tutoring for topics with low scores.")
        else:
            st.write("**Keep it up!**")
            st.write("- Maintain consistent study habits and peer study groups.")

    # Offer download of the prediction
    payload = {
        "studytime": int(studytime),
        "absences": int(absences),
        "failures": int(failures),
        "G1": int(G1),
        "G2": int(G2),
        "probability": probability,
    }

    st.download_button("Download prediction (JSON)", data=json.dumps(payload, indent=2), file_name="prediction.json", mime="application/json")