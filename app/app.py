import os
import sys
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import torch
import joblib

# Add project root to sys.path to enable src imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page layout to wide and title
st.set_page_config(
    page_title="MARS | Audio Deepfake Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for premium Dark Theme, glassmorphism, and styling
st.markdown("""
<style>
    /* Main container background */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1c24 100%);
        color: #f0f2f6;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #12141c;
        border-right: 1px solid rgba(1, 1, 1, 0.05);
    }
    
    /* Custom container/card with glassmorphism */
    .premium-card {
        background: rgba(26, 28, 36, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }
    
    /* Header gradient */
    .gradient-header {
        background: linear-gradient(90deg, #ff4b4b 0%, #ff8a00 50%, #e52e71 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 5px;
    }
    
    /* Subtitle styling */
    .subheader-text {
        color: #8892b0;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    
    /* Genuine alert card styling */
    .genuine-result {
        border-left: 6px solid #10b981;
        background: rgba(16, 185, 129, 0.08);
        border-radius: 8px;
        padding: 16px;
        margin-top: 15px;
    }
    
    /* Deepfake alert card styling */
    .deepfake-result {
        border-left: 6px solid #ef4444;
        background: rgba(239, 68, 68, 0.08);
        border-radius: 8px;
        padding: 16px;
        margin-top: 15px;
    }
    
    /* Custom badge */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-right: 8px;
    }
    .badge-genuine {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-deepfake {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    
    /* Center text */
    .center-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Import backend modules
from src.features import process_file, load_audio
from src.models import DeepfakeAudioCNN

base_dir = os.path.join("D:\\", "MARS_ML")
rf_path = os.path.join(base_dir, "models", "rf_detector.pkl")
cnn_path = os.path.join(base_dir, "models", "deepfake_detector.pth")

# Load models safely
@st.cache_resource
def load_models():
    rf_model = None
    cnn_model = None
    
    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
    if os.path.exists(cnn_path):
        cnn_model = DeepfakeAudioCNN()
        cnn_model.load_state_dict(torch.load(cnn_path, map_location=torch.device('cpu')))
        cnn_model.eval()
        
    return rf_model, cnn_model

rf_model, cnn_model = load_models()

# Sidebar Setup
st.sidebar.markdown("<div class='center-content'><h1 style='font-size: 1.8rem; color: #ff4b4b; margin-top: 0;'>🎙️ MARS ML</h1><p style='color:#8892b0; font-size:0.85rem;'>Audio Forensic Suite</p></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.header("⚙️ Model Configuration")
model_choice = st.sidebar.selectbox(
    "Select Classification Model:",
    ["Random Forest (ML Baseline)", "PyTorch CNN (Deep Learning)"]
)

st.sidebar.markdown("---")

st.sidebar.header("📊 Model Metrics Reference")
st.sidebar.markdown("""
Our models are trained and verified on the *Fake-or-Real* benchmark dataset:
- **Target Accuracy**: &ge; 80%
- **Target EER**: &le; 12%
- **Target Class Acc.**: &ge; 75%
""")

# Main Content
st.markdown("<h1 class='gradient-header'>🎙️ Deepfake Audio Detection Suite</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader-text'>Analyze speech recordings to verify authenticity using machine learning and convolutional neural networks.</p>", unsafe_allow_html=True)

# Check if model files exist
models_ready = True
if model_choice.startswith("Random Forest") and rf_model is None:
    models_ready = False
    st.warning(f"⚠️ Random Forest model weights not found at `{rf_path}`. Please run model training first.")
elif model_choice.startswith("PyTorch CNN") and cnn_model is None:
    models_ready = False
    st.warning(f"⚠️ PyTorch CNN model weights not found at `{cnn_path}`. Please run model training first.")

# Main app UI structure
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
    st.subheader("📤 Upload Audio Sample")
    uploaded_file = st.file_uploader(
        "Upload a speech sample in WAV or MP3 format:",
        type=["wav", "mp3"],
        help="WAV files are recommended for higher precision analysis."
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file is not None and models_ready:
        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        st.subheader("🔊 Play Sample")
        st.audio(uploaded_file, format='audio/wav')
        
        # Save uploaded file temporarily for extraction
        temp_path = os.path.join(base_dir, "temp_upload.wav")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.spinner("Extracting acoustic features and analyzing..."):
            try:
                # Load audio data for visualization
                y, sr = load_audio(temp_path)
                
                # Predict
                if model_choice.startswith("Random Forest"):
                    # Extract 1D features
                    feat_1d, _ = process_file(temp_path)
                    prob = rf_model.predict_proba([feat_1d])[0, 1]
                else:
                    # Extract 2D features
                    _, feat_2d = process_file(temp_path)
                    feat_tensor = torch.tensor(feat_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        logit = cnn_model(feat_tensor).squeeze(0)
                        prob = torch.sigmoid(logit).item()
                
                # Cleanup temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
                prediction = "Deepfake (AI-Generated)" if prob >= 0.5 else "Genuine (Human)"
                confidence = prob if prob >= 0.5 else (1.0 - prob)
                
                # Display Prediction result
                st.markdown("---")
                st.subheader("🔍 Analysis Output")
                
                if prediction.startswith("Genuine"):
                    st.markdown(f"""
                    <div class='genuine-result'>
                        <span class='badge badge-genuine'>Genuine</span>
                        <h3 style='color: #10b981; margin: 5px 0;'>Human Voice Verified</h3>
                        <p style='margin: 0;'>The voice pattern aligns with authentic biological acoustic properties with <b>{confidence * 100:.2f}%</b> confidence.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='deepfake-result'>
                        <span class='badge badge-deepfake'>Deepfake</span>
                        <h3 style='color: #ef4444; margin: 5px 0;'>AI-Generated Voice Detected</h3>
                        <p style='margin: 0;'>Acoustic analysis detected robotic pitch flattening, phase vocoder anomalies, or synthetic high-frequency patterns with <b>{confidence * 100:.2f}%</b> confidence.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show confidence meter
                st.markdown("<br>", unsafe_allow_html=True)
                st.write("**Confidence Score:**")
                st.progress(float(confidence))
                
            except Exception as e:
                st.error(f"Error analyzing audio: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    if uploaded_file is not None and 'y' in locals():
        st.markdown("<div class='premium-card'>", unsafe_allow_html=True)
        st.subheader("📈 Visual Signal Inspection")
        
        # Plot Waveform
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
        fig.patch.set_facecolor('#1a1c24')
        
        # Waveform
        time_axis = np.linspace(0, len(y) / sr, len(y))
        ax1.plot(time_axis, y, color='#ff4b4b', alpha=0.8, linewidth=0.8)
        ax1.set_title("Acoustic Waveform (Amplitude vs Time)", color='#f0f2f6', fontsize=10)
        ax1.set_facecolor('#12141c')
        ax1.tick_params(colors='#8892b0')
        ax1.set_xlabel("Time (s)", color='#8892b0', fontsize=8)
        ax1.set_ylabel("Amplitude", color='#8892b0', fontsize=8)
        ax1.grid(color=(1, 1, 1, 0.05), linestyle='--')
        
        # Spectrogram
        # We will use simple scipy spectrogram calculation for robustness
        from scipy.signal import spectrogram
        f, t_spec, Sxx = spectrogram(y, sr, nperseg=256, noverlap=128)
        
        # Display logarithmic scale
        Sxx_log = 10 * np.log10(Sxx + 1e-10)
        
        im = ax2.pcolormesh(t_spec, f, Sxx_log, cmap='inferno', shading='gouraud')
        ax2.set_title("Spectral Analysis (Frequency vs Time)", color='#f0f2f6', fontsize=10)
        ax2.set_facecolor('#12141c')
        ax2.tick_params(colors='#8892b0')
        ax2.set_xlabel("Time (s)", color='#8892b0', fontsize=8)
        ax2.set_ylabel("Frequency (Hz)", color='#8892b0', fontsize=8)
        ax2.set_ylim(0, 8000) # Limit human speech frequency range
        
        fig.tight_layout()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='premium-card' style='height: 100%; display: flex; align-items: center; justify-content: center;'>", unsafe_allow_html=True)
        st.markdown("""
        <div class='center-content'>
            <span style='font-size: 4rem;'>📊</span>
            <h3 style='color: #8892b0; margin-top: 15px;'>Signal Inspection Desk</h3>
            <p style='color: #5d6780; max-width: 300px; font-size: 0.9rem;'>Upload an audio file on the left to display its waveform and acoustic spectrogram here.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
