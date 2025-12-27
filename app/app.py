import streamlit as st
import tensorflow as tf
import os
import sys
import numpy as np

# ==========================================
# 1. SYSTEM SETUP
# ==========================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import MODEL_PATH
from src.utils import preprocess_image

# ==========================================
# 2. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="DeepDetect AI",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .title-text {
        font-weight: 800; font-size: 3.5rem;
        background: -webkit-linear-gradient(45deg, #4285F4, #34A853);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .result-card {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 30px; border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-top: 20px;
    }
    
    .confidence-bar {
        background-color: #333; 
        height: 12px; 
        border-radius: 6px; 
        width: 100%;
        margin-top: 15px;
        margin-bottom: 15px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. LOAD THE BRAIN
# ==========================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ==========================================
# 4. SIDEBAR
# ==========================================
with st.sidebar:
    st.title("🛡️ DeepDetect")
    st.markdown("### System Status")
    
    if model:
        st.success("🟢 Model Online")
        st.caption("Engine: MobileNetV2")
        st.caption("Resolution: 128x128px")
    else:
        st.error("🔴 Model Offline")
        st.warning(f"Missing: {MODEL_PATH}")

    st.markdown("---")
    st.info("AI forensics system for synthetic artifact detection.")

# ==========================================
# 5. MAIN INTERFACE
# ==========================================
col_main, _ = st.columns([2, 1])
with col_main:
    st.markdown('<p class="title-text">DeepDetect AI</p>', unsafe_allow_html=True)
    st.caption("Next-Generation Synthetic Image Detection System")

if model is None:
    st.warning("⚠️ Model file missing. Please check your setup.")
    st.stop()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.markdown("---")
    col1, col2 = st.columns([1, 1], gap="large")
    
    # Preprocess
    display_image, model_input = preprocess_image(uploaded_file)
    
    if display_image:
        with col1:
            st.image(display_image, caption="Source Image", use_container_width=True)
            
        with col2:
            with st.spinner("Scanning for GAN/Diffusion artifacts..."):
                # PREDICTION
                raw_prediction = model.predict(model_input)
                confidence_score = raw_prediction[0][0]
                
                # --- FINAL CALIBRATED LOGIC (INVERTED) ---
                # We are permanently applying the "Invert" logic.
                # 0.0 = Real
                # 1.0 = Fake (AI)
                
                is_real = confidence_score < 0.5
                
                if is_real:
                    label = "REAL IMAGE"
                    final_score = (1 - confidence_score) * 100
                    color = "#00C853" # Green
                    emoji = "✅"
                    desc = "Natural sensor noise detected."
                else:
                    label = "AI GENERATED"
                    final_score = confidence_score * 100
                    color = "#FF2B2B" # Red
                    emoji = "🤖"
                    desc = "Synthetic artifacts detected."

                # RESULT CARD
                html_code = f"""
<div class="result-card">
    <h3 style="color: {color}; margin:0; letter-spacing: 1px;">{emoji} {label}</h3>
    <h1 style="font-size: 5rem; margin:10px 0; font-weight:800; color:white;">{final_score:.1f}%</h1>
    <p style="color: #A0A0A0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px;">Confidence Score</p>
    <div class="confidence-bar">
        <div style="background-color: {color}; width: {final_score}%; height: 100%;"></div>
    </div>
    <p style="margin-top:20px; color:#ddd; line-height: 1.6;">{desc}</p>
</div>
"""
                st.markdown(html_code, unsafe_allow_html=True)