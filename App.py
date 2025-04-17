import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Custom CSS for styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #cc8d6b;
        background-image: linear-gradient(135deg, #6b6b6b 0%, #181818 100%);
    }
    
    /* Navigation bar */
    .nav {
        display: flex;
        justify-content: space-between;
        padding: 1rem;
        background: rgba(14, 17, 23, 0.9);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid #2D3746;
        position: sticky;
        top: 0;
        z-index: 1000;
        border-radius: 2%;
    }
    
    /* Particle animation */
    @keyframes float {
        0% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
        100% { transform: translateY(0px) rotate(360deg); }
    }
    
    .particles {
        position: fixed;
        width: 100%;
        height: 100%;
        pointer-events: none;
    }
    
    .particle {
        position: absolute;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        width: 4px;
        height: 4px;
        border-radius: 50%;
        animation: float 8s infinite linear;
    }
    
    /* Custom button styling */
    .stButton button {
        background: linear-gradient(45deg, #1A73E8, #0F3460);
        color: white !important;
        border: none;
        transition: all 0.3s ease;
        border-radius: 8px;
    }
    
    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(26, 115, 232, 0.4);
    }
    
    /* Text input styling */
    .stTextArea textarea {
        background: rgba(25, 28, 36, 0.8) !important;
        color: white !important;
        border: 1px solid #2D3746 !important;
        border-radius: 8px;
    }
    
    /* IMDb button styling */
    .imdb-btn {
        background-image: url('https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg');
        background-size: 20px;
        background-repeat: no-repeat;
        background-position: 10px center;
        padding-left: 40px !important;
    }
</style>
""", unsafe_allow_html=True)

# Add floating particles
st.markdown("""
<div class="particles">
    <div class="particle" style="left: 10%; top: 20%"></div>
    <div class="particle" style="left: 30%; top: 50%"></div>
    <div class="particle" style="left: 70%; top: 80%"></div>
    <div class="particle" style="left: 85%; top: 40%"></div>
</div>
""", unsafe_allow_html=True)

# Navigation bar
pages = {
    "Classify": "classify",
    "Profile": "profile",
    "Documents": "documents",
    "Analytics": "analytics"
}

st.markdown("""
<nav class="nav">
    <div style="display: flex; gap: 2rem;">
        <a href="#classify" style="color: white; text-decoration: none; font-weight: 500;">Classify</a>
        <a href="#profile" style="color: white; text-decoration: none; font-weight: 500;">Profile</a>
        <a href="#documents" style="color: white; text-decoration: none; font-weight: 500;">Documents</a>
    </div>
    <div style="color: white; font-weight: 500;">v1.0.0</div>
</nav>
""", unsafe_allow_html=True)

# Page Content
st.title("🎬 Constructive Criticism or Hate?")
st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

# Model loading (keep this outside of main content to load only once)
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("/Users/rafaelzieganpalg/Projects/SRP_Lab/Main_Proj/deberta_model")
    tokenizer = AutoTokenizer.from_pretrained("/Users/rafaelzieganpalg/Projects/SRP_Lab/Main_Proj/deberta_model")
    return model, tokenizer

model, tokenizer = load_model()

# Main content
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        input_review = st.text_area("", placeholder="Enter or paste your movie review here...", height=150)
    with col2:
        st.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)
        if st.button("🎤 Voice Input", use_container_width=True):
            pass  # Add voice input functionality here

    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
    
    # Action buttons
    col3, col4 = st.columns([1, 3])
    with col3:
        if st.button("🔍 Fetch IMDb Reviews", key="fetch", use_container_width=True):
            pass  # Add fetch functionality here
    with col4:
        if st.button("✨ Analyze Review", type="primary", use_container_width=True):
            if input_review:
                with st.spinner("Analyzing..."):
                    inputs = tokenizer(input_review, return_tensors="pt", truncation=True, padding=True)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        prediction = torch.argmax(outputs.logits).item()
                
                label = "🌟 Constructive Criticism" if prediction == 0 else "🔥 Hate Speech"
                color = "#4ECDC4" if prediction == 0 else "#FF6B6B"
                
                st.markdown(f"""
                <div style="padding: 1.5rem; background: rgba(25, 28, 36, 0.8); border-radius: 12px; border-left: 4px solid {color}; margin: 1rem 0;">
                    <h3 style="color: {color}; margin: 0;">{label}</h3>
                    <p style="color: #8792A2; margin: 0.5rem 0 0 0;">Analysis Result</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Please enter a review to analyze")

# Footer
st.markdown("""
<div style="position: fixed; bottom: 0; right: 0; padding: 1rem; color: #8792A2;">
    Powered by DeBERTa v3 | SRP Labs
</div>
""", unsafe_allow_html=True)