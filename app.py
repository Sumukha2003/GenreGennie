import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

from src.utils import get_transfer_input, load_audio


st.set_page_config(page_title="Genre Genie", page_icon="🎵", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
.main {background: linear-gradient(-45deg, #1e3c72, #2a5298, #4facfe, #00f2fe); background-size: 400% 400%; animation: gradientShift 15s ease infinite;}
@keyframes gradientShift {0% {background-position: 0% 50%;} 50% {background-position: 100% 50%;} 100% {background-position: 0% 50%;}}
.stButton > button {background: linear-gradient(45deg, #FF6B6B, #4ECDC4); color: white; border-radius: 25px; padding: 12px 32px; font-weight: bold;}
.card {background: rgba(255,255,255,0.15); backdrop-filter: blur(20px); border-radius: 20px; padding: 2rem; margin: 1rem 0; text-align: center;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    family_model = load_model("models/family/best_model.keras")
    western_model = load_model("models/western/best_model.keras")
    indian_model = load_model("models/indian/best_model.keras")

    family_encoder = joblib.load("models/family/encoder.pkl")
    western_encoder = joblib.load("models/western/encoder.pkl")
    indian_encoder = joblib.load("models/indian/encoder.pkl")
    return {
        "family_model": family_model,
        "western_model": western_model,
        "indian_model": indian_model,
        "family_encoder": family_encoder,
        "western_encoder": western_encoder,
        "indian_encoder": indian_encoder,
    }


def prepare_input(audio_source):
    y, sr = load_audio(audio_source)
    features = get_transfer_input(y, sr)[np.newaxis, ...]
    return preprocess_input(features * 255.0)


def predict_audio(audio_source):
    features = prepare_input(audio_source)

    family_probs = models["family_model"].predict(features, verbose=0)[0]
    family_idx = int(np.argmax(family_probs))
    family_label = models["family_encoder"].classes_[family_idx]

    if family_label == "western":
        detail_probs = models["western_model"].predict(features, verbose=0)[0]
        detail_classes = models["western_encoder"].classes_
    else:
        detail_probs = models["indian_model"].predict(features, verbose=0)[0]
        detail_classes = models["indian_encoder"].classes_

    return family_label, family_probs, detail_classes, detail_probs


def render_prediction(family_label, family_probs, detail_classes, detail_probs):
    best_idx = int(np.argmax(detail_probs))
    best_label = detail_classes[best_idx]
    best_conf = detail_probs[best_idx] * 100
    family_conf = np.max(family_probs) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div class="card">
                <h1 style='color: gold'>🎯 <b>BEST MATCH</b></h1>
                <h2>{best_label.upper()}</h2>
                <h3>{best_conf:.1f}%</h3>
                <p><b>Family:</b> {family_label.upper()} ({family_conf:.1f}%)</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        df_top3 = pd.DataFrame(
            {"Match": detail_classes, "Confidence %": detail_probs * 100}
        ).nlargest(3, "Confidence %")
        st.markdown("**📈 Top 3 Near Matches:**")
        st.bar_chart(df_top3.set_index("Match"), height=400)


models = load_models()

st.sidebar.title("🎵 Genre Genie")
st.sidebar.success("✅ Two-stage classifier: family + specialist")
st.sidebar.markdown("**Western classes:** " + ", ".join(models["western_encoder"].classes_))
st.sidebar.markdown("**Indian classes:** " + ", ".join(models["indian_encoder"].classes_))

st.markdown("# 🎵 **Genre Genie** - Upload & Classify")
st.markdown("**Upload audio and get a family prediction first, then a specialist genre/instrument prediction.**")

uploaded_file = st.file_uploader(
    "📁 Upload Song (MP3/WAV/M4A)", type=["mp3", "wav", "m4a", "ogg", "flac"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("🔮 Analyze Uploaded Song", use_container_width=True):
        with st.spinner("🎵 Extracting features and running the two-stage classifier..."):
            family_label, family_probs, detail_classes, detail_probs = predict_audio(uploaded_file)

        render_prediction(family_label, family_probs, detail_classes, detail_probs)
        st.balloons()
        st.success("✅ Analysis complete!")
else:
    st.info("👆 Upload a song to see the best genre/instrument match.")

st.markdown("---")
