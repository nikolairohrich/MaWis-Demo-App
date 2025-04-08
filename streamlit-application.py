import streamlit as st
import torch
import os
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

led_types = ['cree', 'dominant', 'lumileds', 'nichia131', 'nichia170', 'osrambf', 'osramcomp', 'samsung', 'seoul']

# Load PyTorch models (Replace with your own models)
# TODO: Use correct paths to the specified .pth files 
def load_models():
    model1 = torch.load(os.path.join('models', 'mae_classification.pth'), map_location='cpu', weights_only=False)['model']
    model1.eval()

    model2 = torch.load(os.path.join('models', 'mae_regression.pth'), map_location='cpu', weights_only=False)['model']
    model2.eval()

    model3 = torch.load(os.path.join('models', 'mae_quality_classification.pth'), map_location='cpu', weights_only=False)['model']
    model3.eval()

    return model1, model2, model3


# Preprocess the image before feeding it to the model
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


# Run inference on the image
def get_model_outputs(image, model):
    with torch.no_grad():
        outputs = model(image)
    return outputs.numpy().flatten()

# Streamlit App
st.title("🔍 MaWIs-KI")

# Sidebar Navigation
page = st.sidebar.radio(
    "Navigate:",
    ["Home", "Image-Based Deep Learning", "LED Type & Solder Paste Prediction", "About"],
    format_func=lambda name: {
        "Home": "🏠 Home",
        "Image-Based Deep Learning": "🖼️ Image-Based Deep Learning ",
        "LED Type & Solder Paste Prediction": "🔬 Numerical Prediction",
        "About": "ℹ️ About"
    }.get(name, name)
)

# Display content based on selected page
if page == "Home":
    st.header("Home Page")
    st.write("Welcome to the MaWiS-KI App!")
    st.write("""Here, we showcase various statistical and deep learning models used for the lifetime prediction of solder joints.
             Just use the sidebar to navigate between models.""")

elif page == "Image-Based Deep Learning":
    st.header("Image-Based Deep Learning")
    st.write("""Here, we showcase Vision Transformer (ViT) models for solder joint predictive maintenance. 
                Our models were self-pretrained using the Masked Autoencoder (MAE) framework, where large portions of 
                inputs are masked and reconstructed by the model.""")
    # Load models
    model1, model2, model3 = load_models()

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    st.markdown("""
    <style>
        .highlight {
            background-color: #0c3b39;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2, col3 = st.columns(3)
        with col2: 
            st.image(image, caption="Uploaded Image")

        # Preprocess image
        processed_image = preprocess_image(image)

        # Get outputs from models
        classification_output = get_model_outputs(processed_image, model1)
        led_type = led_types[np.argmax(classification_output)]
        led_type = led_type[0].upper() + led_type[1:]

        regression_output = get_model_outputs(processed_image, model2)

        quality_classification = get_model_outputs(processed_image, model3)
     
        # Display outputs
        with col2: 
            st.write("**Model Predictions:**")
            st.markdown(f"LED Type: <span class='highlight'>{led_type}</span>", unsafe_allow_html=True)
            st.markdown(f"LED still functional: <span class='highlight'>{min(100, float(quality_classification*100)):.2f}%</span>", unsafe_allow_html=True)
            st.markdown(f"BMAX at 1250 TSC: <span class='highlight'>{regression_output[0]:.2f}</span>", unsafe_allow_html=True)



elif page == "LED Type & Solder Paste Prediction":
    st.header("LED & Solder Paste Selection")

    st.write("""Here, we showcase a statistical model.""")
    
    # Select LED types
    led_type = st.selectbox("Choose LED Type:", [led_type[0].upper() + led_type[1:] for led_type in led_types])
    
    # Select Solder Pastes
    solder_paste = st.selectbox("Choose Solder Paste:", ["SAC105", 
                                                         "SAC+BiIn", 
                                                         "SAC305",
                                                         "SAC+Sb",
                                                         "SAC+SbBiNi"])
    
    # Select TTA values
    tsc_cycle = st.slider("Select TSC Value:", min_value=0, max_value=1500, step=1)
    
    # Define mathematical formula (Example Formula: LED Type Weight + Solder Paste Effect + TTA Factor)
    led_factor = {  'cree': 0.56,
                    'dominant': 0.99,
                    'lumileds': 1.67,
                    'nichia131': 1.06,
                    'nichia170': 1.25,
                    'osrambf': 0.98,
                    'osramcomp': 1.42,
                    'samsung': 0.83,
                    'seoul': 1.2}[led_type.lower()]
    paste_factor = {'SAC105': 0.86,
                    'SAC+BiIn': 1.55,
                    'SAC305': 1.06,
                    'SAC+Sb': 0.52,
                    'SAC+SbBiNi': 1.66}[solder_paste]
    
    result = max(0, (led_factor + paste_factor) * 500 - tsc_cycle)
    
    # Display Output
    st.subheader("Calculated Output Value")
    st.write(f"For **{led_type}**, **{solder_paste}**, and current TSC **{tsc_cycle}**, the calculated remaining useful lifetime is:")
    col1, col2, col3 = st.columns(3)
    with col2: 
        st.write(f"**{result:.2f}** TSC")


elif page == "About":
    st.header("About")
    st.write("This app demonstrates the use of PyTorch models in Streamlit for image processing.")
