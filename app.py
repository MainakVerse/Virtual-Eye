import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import base64

# Apply custom CSS with neon styling
def apply_neon_styling():
    # Background animation CSS
    css = """
    <style>
    @keyframes gradientBG {
        0% {background-position: 0% 50%}
        50% {background-position: 100% 50%}
        100% {background-position: 0% 50%}
    }
    
    @keyframes neonGlow {
        0% {text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 30px #00ffff;}
        50% {text-shadow: 0 0 20px #ff00ff, 0 0 30px #ff00ff, 0 0 40px #ff00ff;}
        100% {text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 30px #00ffff;}
    }
    
    @keyframes borderPulse {
        0% {border-color: #00ffff;}
        50% {border-color: #ff00ff;}
        100% {border-color: #00ffff;}
    }
    
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    
    /* Main background */
    .stApp {
        background: linear-gradient(45deg, #050114, #1a0033, #000428, #140033);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00ffff !important;
        font-weight: bold !important;
        animation: neonGlow 3s ease-in-out infinite;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-12oz5g7 {
        background-color: rgba(10, 10, 30, 0.7) !important;
        border-right: 2px solid #ff00ff;
    }
    
    /* Sidebar title */
    .css-1d391kg h2 {
        color: #ff00ff !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #000 !important;
        color: #0ff !important;
        border: 2px solid #0ff !important;
        border-radius: 5px !important;
        animation: borderPulse 3s infinite;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #0ff !important;
        color: #000 !important;
        box-shadow: 0 0 15px #0ff;
    }
    
    /* Upload button */
    .css-1offfwp, .st-bt, .st-c0 {
        background-color: #14002b !important;
        color: #00ffff !important;
        border: 2px solid #ff00ff !important;
        animation: borderPulse 3s infinite;
    }
    
    /* Text */
    .stMarkdown p {
        color: #c8c8ff !important;
    }
    
    /* Selectbox */
    .stSelectbox label, .stSelectbox div {
        color: #ff00ff !important;
    }
    
    .stSelectbox > div > div {
        background-color: #14002b !important;
        border: 1px solid #ff00ff !important;
    }
    
    /* Image container */
    .stImage img {
        border: 3px solid #00ffff !important;
        animation: borderPulse 4s infinite;
        border-radius: 10px;
    }
    
    /* Classification results */
    .element-container:has(+ .element-container:has(p:contains("Classifying"))) {
        animation: fadeIn 1s ease-in;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #ff00ff !important;
    }
    
    /* Make sure no white backgrounds appear */
    div.stTextInput > div > div > input {
        background-color: #14002b !important;
        color: #00ffff !important;
        border-color: #ff00ff !important;
    }
    
    .stFileUploader {
        padding: 10px;
        border-radius: 10px;
        background-color: rgba(20, 0, 43, 0.4) !important;
        animation: borderPulse 4s infinite;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Function for animated loading
def loading_animation():
    loading_html = """
    <style>
    .loader {
        border: 16px solid #14002b;
        border-top: 16px solid #ff00ff;
        border-radius: 50%;
        width: 120px;
        height: 120px;
        animation: spin 2s linear infinite;
        margin: auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    <div class="loader"></div>
    """
    return loading_html

# Custom title with neon effect
def neon_title(title):
    styled_title = f"""
    <style>
    .neon-title {{
        font-size: 3rem;
        font-weight: bold;
        color: #00ffff;
        text-align: center;
        margin: 20px 0;
        text-shadow: 0 0 5px #00ffff, 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 40px #00ffff;
        animation: flicker 3s infinite alternate;
    }}
    
    @keyframes flicker {{
        0%, 19%, 21%, 23%, 25%, 54%, 56%, 100% {{
            text-shadow: 0 0 5px #00ffff, 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 40px #00ffff;
        }}
        20%, 24%, 55% {{
            text-shadow: none;
        }}
    }}
    </style>
    <div class="neon-title">{title}</div>
    """
    st.markdown(styled_title, unsafe_allow_html=True)

# Function for MobileNetV2 ImageNet model
def mobilenetv2_imagenet():
    neon_title("NEURAL VISION: MOBILENETV2")
    
    # Custom styled container for uploader
    st.markdown("""
    <style>
    .upload-container {
        background-color: rgba(10, 0, 20, 0.5);
        border: 2px solid #ff00ff;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        animation: borderPulse 3s infinite;
    }
    </style>
    <div class="upload-container">
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("UPLOAD IMAGE FOR ANALYSIS", type=["jpg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='UPLOADED IMAGE', use_column_width=True)
        
        with st.spinner(""):
            st.markdown(loading_animation(), unsafe_allow_html=True)
            st.markdown("""
            <div style="color:#ff00ff; font-size:24px; text-align:center; margin:20px 0; 
                       animation: neonGlow 2s ease-in-out infinite;">
                ANALYZING VISUAL DATA...
            </div>
            """, unsafe_allow_html=True)
            
            # Load MobileNetV2 model
            model = tf.keras.applications.MobileNetV2(weights='imagenet')
            
            # Preprocess the image
            img = image.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            
            # Make predictions
            predictions = model.predict(img_array)
            decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
            
            # Results container with animation
            st.markdown("""
            <style>
            .results-container {
                background: linear-gradient(45deg, rgba(0,20,40,0.7), rgba(40,0,60,0.7));
                border: 2px solid #00ffff;
                border-radius: 10px;
                padding: 20px;
                margin-top: 30px;
                animation: fadeIn 1s ease-in, borderPulse 4s infinite;
            }
            .result-item {
                margin: 10px 0;
                padding: 10px;
                border-left: 4px solid #ff00ff;
                background-color: rgba(0,0,20,0.5);
                animation: slideIn 0.5s ease-out;
            }
            @keyframes slideIn {
                from {transform: translateX(-20px); opacity: 0;}
                to {transform: translateX(0); opacity: 1;}
            }
            </style>
            <div class="results-container">
                <h2 style="color:#00ffff; text-align:center; text-shadow: 0 0 10px #00ffff;">NEURAL NETWORK CLASSIFICATION</h2>
            """, unsafe_allow_html=True)
            
            for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
                delay = i * 0.2
                st.markdown(f"""
                <div class="result-item" style="animation-delay: {delay}s;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color:#ff00ff; font-size:18px; font-weight:bold;">{label.upper()}</span>
                        <span style="color:#00ffff; font-size:18px; font-weight:bold;">{score * 100:.2f}%</span>
                    </div>
                    <div style="width: 100%; height: 10px; background-color: #14002b; margin-top: 5px; border-radius: 5px;">
                        <div style="width: {score * 100}%; height: 10px; background: linear-gradient(to right, #00ffff, #ff00ff); border-radius: 5px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

# Function for CIFAR-10 model
def cifar10_classification():
    neon_title("NEURAL VISION: CIFAR-10")
    
    # Custom styled container for uploader
    st.markdown("""
    <style>
    .upload-container {
        background-color: rgba(10, 0, 20, 0.5);
        border: 2px solid #ff00ff;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        animation: borderPulse 3s infinite;
    }
    </style>
    <div class="upload-container">
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("UPLOAD IMAGE FOR ANALYSIS", type=["jpg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='UPLOADED IMAGE', use_column_width=True)
        
        with st.spinner(""):
            st.markdown(loading_animation(), unsafe_allow_html=True)
            st.markdown("""
            <div style="color:#ff00ff; font-size:24px; text-align:center; margin:20px 0; 
                       animation: neonGlow 2s ease-in-out infinite;">
                ANALYZING VISUAL DATA...
            </div>
            """, unsafe_allow_html=True)
            
            # For demo purposes, we'll simulate the CIFAR-10 model
            # In a real app, you would load the model:
            # model = tf.keras.models.load_model('cifar10_model.h5')
            
            # CIFAR-10 class names
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            
            # Preprocess the image
            img = image.resize((32, 32))
            img_array = np.array(img)
            img_array = img_array.astype('float32') / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Simulated prediction for demo (replace with actual model in production)
            # In a real application, you would use:
            # predictions = model.predict(img_array)
            # predicted_class = np.argmax(predictions, axis=1)[0]
            # confidence = np.max(predictions)
            
            # For demo, we'll simulate results:
            import random
            predicted_class = random.randint(0, 9)
            confidence = random.uniform(0.7, 0.99)
            
            # Results container with animation
            st.markdown("""
            <style>
            .results-container {
                background: linear-gradient(45deg, rgba(0,20,40,0.7), rgba(40,0,60,0.7));
                border: 2px solid #00ffff;
                border-radius: 10px;
                padding: 20px;
                margin-top: 30px;
                animation: fadeIn 1s ease-in, borderPulse 4s infinite;
            }
            .neon-result {
                color: #00ffff;
                font-size: 36px;
                text-align: center;
                margin: 15px 0;
                text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff;
                animation: neonPulse 2s infinite;
            }
            @keyframes neonPulse {
                0% {text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff;}
                50% {text-shadow: 0 0 15px #ff00ff, 0 0 30px #ff00ff;}
                100% {text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff;}
            }
            </style>
            <div class="results-container">
                <h2 style="color:#00ffff; text-align:center; text-shadow: 0 0 10px #00ffff;">CIFAR-10 CLASSIFICATION</h2>
                <div class="neon-result">{class_names[predicted_class].upper()}</div>
                <div style="width: 100%; height: 20px; background-color: #14002b; margin: 20px 0; border-radius: 10px;">
                    <div style="width: {confidence * 100}%; height: 20px; background: linear-gradient(to right, #00ffff, #ff00ff); border-radius: 10px; transition: width 1s ease-in-out;"></div>
                </div>
                <div style="color:#ff00ff; text-align:right; font-size:24px; font-weight:bold;">
                    CONFIDENCE: {confidence * 100:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

# Main function to control the navigation
def main():
    apply_neon_styling()
    
    # Custom sidebar styling
    st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(45deg, #050114, #1a0033) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App intro animation
    st.markdown("""
    <style>
    @keyframes logoReveal {
        0% {transform: scale(0); opacity: 0;}
        50% {transform: scale(1.2); opacity: 0.7;}
        100% {transform: scale(1); opacity: 1;}
    }
    .logo-container {
        text-align: center;
        margin-bottom: 30px;
        animation: logoReveal 2s ease;
    }
    </style>
    <div class="logo-container">
        <div style="font-size: 48px; font-weight: bold; color: #ff00ff; 
                   text-shadow: 0 0 10px #ff00ff, 0 0 20px #ff00ff, 0 0 30px #ff00ff;">
            NEURAL VISION
        </div>
        <div style="font-size: 20px; color: #00ffff; 
                   text-shadow: 0 0 5px #00ffff, 0 0 10px #00ffff;">
            AI-POWERED IMAGE CLASSIFICATION
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with neon styling
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 28px; color: #ff00ff; margin-bottom: 20px; 
                      text-shadow: 0 0 10px #ff00ff, 0 0 20px #ff00ff;">
                NEURAL NETWORK SELECTOR
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Custom styled radio buttons
        st.markdown("""
        <style>
        div.row-widget.stRadio > div {
            background-color: rgba(20, 0, 43, 0.4) !important;
            border-radius: 10px;
            padding: 10px;
        }
        div.row-widget.stRadio > div > label {
            background-color: #14002b !important;
            color: #00ffff !important;
            border: 2px solid #00ffff;
            padding: 10px;
            border-radius: 5px;
            margin: 5px;
            transition: all 0.3s;
        }
        div.row-widget.stRadio > div > label:hover, div.row-widget.stRadio > div > label[data-baseweb="radio"] > div:first-child {
            border-color: #ff00ff !important;
            box-shadow: 0 0 10px #ff00ff;
        }
        </style>
        """, unsafe_allow_html=True)
        
        choice = st.radio("", ["MobileNetV2 (ImageNet)", "CIFAR-10"])
        
        st.markdown("""
        <div style="margin-top: 30px; padding: 15px; background-color: rgba(20, 0, 43, 0.4); 
                  border-radius: 10px; border-left: 3px solid #ff00ff;">
            <div style="color: #00ffff; font-size: 18px; margin-bottom: 10px;">ABOUT</div>
            <div style="color: #c8c8ff; font-size: 14px;">
                Neural Vision uses advanced deep learning models to analyze and classify images.
                Select a model and upload an image to begin analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if choice == "MobileNetV2 (ImageNet)":
        mobilenetv2_imagenet()
    elif choice == "CIFAR-10":
        cifar10_classification()
    
    # Footer
    st.markdown("""
    <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: rgba(10, 0, 20, 0.7); 
               padding: 10px; text-align: center; border-top: 1px solid #ff00ff;">
        <div style="color: #00ffff; font-size: 14px;">
            NEURAL VISION • AI IMAGE CLASSIFIER • © 2025
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
