import streamlit as st
import tensorflow as tf
import os
import sys

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

# --- CSS (Defined flush left to prevent errors) ---
custom_css = """
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
    
    .subtitle {
        font-size: 1.2rem;
        color: #A0A0A0;
        margin-bottom: 40px;
    }

    .result-card {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 30px; 
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin-top: 20px;
        text-align: center;
    }
    
    .confidence-bar {
        background-color: #333; 
        height: 12px; 
        border-radius: 6px; 
        width: 100%;
        margin-top: 15px;
        margin-bottom: 15px;
        overflow: hidden;
        position: relative;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s ease-in-out;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================
# 3. LOAD THE BRAIN
# ==========================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

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
        st.warning("Please place 'deepdetect_mobilenet_128.h5' in the models folder.")

    st.markdown("---")
    st.info("AI forensics system for synthetic artifact detection.")

# ==========================================
# 5. MAIN INTERFACE
# ==========================================
col_main, _ = st.columns([2, 1])
with col_main:
    st.markdown('<p class="title-text">DeepDetect AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Next-Generation Synthetic Image Detection System</p>', unsafe_allow_html=True)

if model is None:
    st.warning("⚠️ Model file missing. Please check your setup.")
    st.stop()

uploaded_file = st.file_uploader("Upload an image to analyze", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.markdown("---")
    col1, col2 = st.columns([1, 1], gap="large")
    
    # --- PRE-PROCESSING ---
    display_image, model_input = preprocess_image(uploaded_file)
    
    if display_image:
        with col1:
            st.image(display_image, caption="Source Image", use_container_width=True)
            
        with col2:
            with st.spinner("Scanning for GAN/Diffusion artifacts..."):
                # --- PREDICTION ---
                prediction = model.predict(model_input)
                confidence = prediction[0][0]
                
                # LOGIC
                if confidence > 0.5:
                    label = "REAL IMAGE"
                    score = confidence * 100
                    color = "#00C853"
                    emoji = "✅"
                    desc = "Natural sensor noise detected."
                else:
                    label = "AI GENERATED"
                    score = (1 - confidence) * 100
                    color = "#FF2B2B"
                    emoji = "🤖"
                    desc = "Synthetic artifacts detected."

                # --- RESULT CARD (NO INDENTATION HERE) ---
                # We define the HTML variable flush left to force it to work
                result_html = f"""
<div class="result-card">
    <h3 style="color: {color}; margin:0; letter-spacing: 1px;">{emoji} {label}</h3>
    <h1 style="font-size: 5rem; margin:10px 0; font-weight:800; color:white;">{score:.1f}%</h1>
    <p style="color: #A0A0A0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 2px;">Confidence Score</p>
    <div class="confidence-bar">
        <div class="confidence-fill" style="background-color: {color}; width: {score}%;"></div>
    </div>
    <p style="margin-top:20px; color:#ddd; line-height: 1.6;">{desc}</p>
</div>
"""
                st.markdown(result_html, unsafe_allow_html=True)