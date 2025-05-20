import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import time
from model_utils import LifetimePredictionModel, parse_size, parse_ratio, normalize_feature, denormalize_feature
import torchvision.transforms as transforms
from PIL import Image

led_types = ['cree', 'dominant', 'lumileds', 'nichia131', 'nichia170', 'osrambf', 'osramcomp', 'samsung', 'seoul']

def _load_models():
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


# === PAGE CONFIGURATION ===
st.set_page_config(page_title="MAWIS-KI", layout="centered")

# === CSS STYLING ===
st.markdown("""
    <style>
    html, body {
        margin: 0;
        padding: 0;
        font-family: "Segoe UI", sans-serif;
    }
    .main-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 2rem 2rem;
    }
    .title-center {
        text-align: center;
        font-size: 3.5rem;       /* Make it bigger */
        font-weight: 900;        /* Make it bolder */
        color: #4A90E2;
        margin-top: -2rem;       /* Pull closer to the top */
        margin-bottom: 1.5rem;   /* Keep space below */
        line-height: 1;
    }
    .section-title {
        font-size: 1.6em;
        color: #333;
        font-weight: bold;
        margin-top: 30px;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1em;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

# === APP HEADING ===
st.markdown('<div class="title-center">MAWIS-KI</div>', unsafe_allow_html=True)

# === APP LAYOUT ===
tab1, tab2, tab3 = st.tabs(["THI Prediction Model", "mts Prediction Model", "XITASO Prediction Model"])

# === Constants ===
min_max_values = {
    'Creep_Strain': (0.007, 0.576),
    'Pad_Size': (1.45, 2.79),
    'Pad_gap': (0.25, 0.45),
    'ceramic_size': (1.6, 2.5),
    'elec_pad_size': (0.49, 1.6),
    'therm_pad_size': (0.0, 1.7),
    'Pad_ratio': (1.0, 4.44),
    'Lifetime': (383, 1817)
}

# === TAB 1: THI with Subtabs ===
with tab1:
    subtab1, subtab2 = st.tabs(["LED Lifetime Prediction", "AI Based Solder reliability Prediction using 3D FEA data"])

    with subtab1:
        st.write('This model predicts the lifetime of LEDs under thermal cycling using a deep learning-based regression model. ' \
        'Simply adjust input paramters like solder type or LED type to generate predictions.')
        model = LifetimePredictionModel(4, 7, 2, 2, 7, 8)
        model.load_state_dict(torch.load("saved_models/final_model.pth", map_location=torch.device("cpu")))
        model.eval()

        encoder_solder = LabelEncoder()
        encoder_led = LabelEncoder()
        encoder_submount = LabelEncoder()
        encoder_pad_count = LabelEncoder()

        encoder_solder.classes_ = np.array(['SAC305', 'SAC105', 'SAC387+SbBiNi', 'SAC107+BiIn'])
        encoder_led.classes_ = np.array(['FC-SP2', 'FC-SP3', 'FC-GB1', 'FC-GB2', 'FC-GB3', 'VTF2', 'FC-SP4'])
        encoder_submount.classes_ = np.array(['thick film AlN ceramic', 'thin film AlN ceramic'])
        encoder_pad_count.classes_ = np.array(['two', 'three'])

        col1, col2 = st.columns(2)
        with col1:
            solder_type = st.selectbox("Solder Type", encoder_solder.classes_)
        with col2:
            led_name = st.selectbox("LED Name", encoder_led.classes_)

        col3, col4 = st.columns(2)
        with col3:
            submount = st.selectbox("Submount", encoder_submount.classes_)
        with col4:
            pad_count = st.selectbox("Pad Count", encoder_pad_count.classes_)

        creep_strain = st.slider("Creep Strain", 0.0, 1.0, 0.01)
        pad_size = st.slider("Pad Size (mm²)", 0.0, 5.0, 1.8)
        pad_gap = st.slider("Pad Gap (mm)", 0.0, 1.0, 0.3)

        with st.expander("Advanced Inputs"):
            ceramic_size_str = st.text_input("Ceramic Size", "1.6 x 1.6")
            elec_pad_size_str = st.text_input("Electrical Pad Size", "1.5 x 0.6")
            therm_pad_size_str = st.text_input("Thermal Pad Size", "0")
            pad_ratio_str = st.text_input("Pad Ratio", "1:1")

        if st.button("Predict Lifetime"):
            with st.spinner("Predicting..."):
                ceramic_size = parse_size(ceramic_size_str)
                elec_pad_size = parse_size(elec_pad_size_str)
                therm_pad_size = parse_size(therm_pad_size_str)
                pad_ratio = parse_ratio(pad_ratio_str)

                features = [
                    normalize_feature(creep_strain, *min_max_values['Creep_Strain']),
                    normalize_feature(pad_size, *min_max_values['Pad_Size']),
                    normalize_feature(pad_gap, *min_max_values['Pad_gap']),
                    normalize_feature(ceramic_size, *min_max_values['ceramic_size']),
                    normalize_feature(elec_pad_size, *min_max_values['elec_pad_size']),
                    normalize_feature(therm_pad_size, *min_max_values['therm_pad_size']),
                    normalize_feature(pad_ratio, *min_max_values['Pad_ratio'])
                ]

                X = torch.tensor([features], dtype=torch.float32)
                solder_input = torch.tensor([encoder_solder.transform([solder_type])], dtype=torch.long)
                led_input = torch.tensor([encoder_led.transform([led_name])], dtype=torch.long)
                submount_input = torch.tensor([encoder_submount.transform([submount])], dtype=torch.long)
                pad_count_input = torch.tensor([encoder_pad_count.transform([pad_count])], dtype=torch.long)

                with torch.no_grad():
                    pred_norm = model(solder_input, led_input, submount_input, pad_count_input, X).item()
                lifetime = denormalize_feature(pred_norm, *min_max_values['Lifetime'])

            st.success(f"Predicted LED Lifetime: **{lifetime:.2f} cycles**")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=lifetime,
                title={'text': "LED Lifetime (cycles)"},
                gauge={
                    'axis': {'range': [None, 1817]},
                    'steps': [
                        {'range': [0, 600], 'color': "#FF4C4C"},
                        {'range': [600, 1000], 'color': "#FFC107"},
                        {'range': [1000, 1817], 'color': "#4CAF50"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

    with subtab2:
        st.write('This model predicts the lifetime of LEDs based on 3D point cloud data using a simplified PointNet-inspired neural network. ' \
        'Simply upload a .npy point cloud file to get a reliability prediction.')
        class SimplePointNet(nn.Module):
            def __init__(self):
                super(SimplePointNet, self).__init__()
                self.mlp1 = nn.Linear(4, 128)
                self.mlp2 = nn.Linear(128, 256)
                self.global_pool = nn.AdaptiveMaxPool1d(1)
                self.fc1 = nn.Linear(256, 1)
                self.apply(self._initialize_weights)

            def _initialize_weights(self, m):
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

            def forward(self, x):
                x = F.relu(self.mlp1(x))
                x = F.relu(self.mlp2(x))
                x = x.permute(0, 2, 1)
                x = self.global_pool(x).squeeze(-1)
                x = self.fc1(x)
                return x

        min_vals = np.array([-850.0, -1525.0, -934.36, 5.9427e-10])
        max_vals = np.array([850.0, 765.0, -646.36, 0.60532])
        lifetime_min = 383
        lifetime_max = 1047

        def normalize_point_cloud(pc): return (pc - min_vals) / (max_vals - min_vals)
        def unnormalize_lifetime(val): return val * (lifetime_max - lifetime_min) + lifetime_min

        def load_models(model_dir):
            models = []
            for file in sorted(os.listdir(model_dir)):
                if file.endswith(".pt"):
                    model = SimplePointNet()
                    model.load_state_dict(torch.load(os.path.join(model_dir, file), map_location=torch.device("cpu")))
                    model.eval()
                    models.append(model)
            return models

        def plot_point_cloud(pc, title):
            x, y, z, creep = pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3]
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(x, y, z, c=creep, cmap='viridis', s=1)
            ax.set_title(title)
            plt.colorbar(sc, shrink=0.5)
            st.pyplot(fig)

        def plot_prediction_variance(unnorm_preds, file_name):
            fig, ax = plt.subplots()
            ax.bar(range(1, len(unnorm_preds) + 1), unnorm_preds)
            ax.set_title(f"Prediction Variance - {file_name}")
            st.pyplot(fig)

        uploaded_files = st.file_uploader("📂 Upload `.npy` files", type=["npy"], accept_multiple_files=True, key="tab2_uploader")
        models = load_models("loocv_model")

        if uploaded_files:
            if not models:
                st.error("No models found in 'loocv_model' directory.")
            else:
                for file in uploaded_files:
                    try:
                        pc = np.load(file)
                        if pc.ndim != 2 or pc.shape[1] != 4:
                            st.error("File must have shape (N, 4).")
                            continue

                        norm_pc = normalize_point_cloud(pc)
                        tensor = torch.tensor(norm_pc, dtype=torch.float32).unsqueeze(0)

                        with torch.no_grad():
                            preds = [model(tensor).item() for model in models]

                        final_pred = unnormalize_lifetime(np.mean(preds))
                        unnorm_preds = [unnormalize_lifetime(p) for p in preds]

                        st.success(f"Predicted Lifetime: **{final_pred:.2f} cycles**")
                        with st.expander("📊 Fold-wise Predictions"):
                            plot_prediction_variance(unnorm_preds, file.name)

                        with st.expander("🧬 3D Point Cloud"):
                            plot_point_cloud(norm_pc, f"{file.name} - Creep Strain")

                        with st.expander("📋 Table"):
                            df = pd.DataFrame({
                                "Fold": range(1, len(unnorm_preds)+1),
                                "Prediction (cycles)": unnorm_preds
                            })
                            st.dataframe(df)

                    except Exception as e:
                        st.error(f"Error with {file.name}: {e}")

# === TAB 2: mts MODEL ===
with tab2:
    st.markdown('<h1 class="main-title">mts Model Prediction</h1>', unsafe_allow_html=True)
    st.write("""Here, we showcase our lifetime prediction model, which estimates the expected cycle number for electronic components based on selected component type and solder type.
        The model uses empirically derived coefficients to calculate the predicted lifetime under thermal cycling conditions.""")
    mts_base_cycle = 510.013
    resistance_weight = 49.16141
    component_coefficients = {
        "BGA": 113.2908,
        "R0402": 162.3644,
        "R0805": 149.0211,
        "WLP": -424.6763
    }
    solder_coefficients = {
        "SAC105": 7.746661,
        "SAC387": 11.59086,
        "SAC396+Sb": 22.09496,
        "SAC107+BiIn": -21.15426,
        "SAC387+SbBiNi": -20.27822
    }

    selected_component = st.selectbox("Select Component Type", list(component_coefficients.keys()))
    selected_solder = st.selectbox("Select Solder Type", list(solder_coefficients.keys()))
    selected_resistance = st.slider(label = "Select Initial Resistance", min_value=0, max_value=11, step=1)

    predicted_cycles = mts_base_cycle + component_coefficients[selected_component] + solder_coefficients[selected_solder] - selected_resistance * resistance_weight
    
    # Display highlighted prediction
    # if st.button("Predict Lifetime (mts)"):
    st.markdown(f"### **Predicted Cycle Number: {predicted_cycles:.2f} Cycles**")

# === TAB 2: XITASO MODEL ===

with tab3:
    st.markdown('<h1 class="main-title">XITASO Model</h1>', unsafe_allow_html=True)
    st.write("""Here, we showcase Vision Transformer (ViT) models that were self-pretrained 
             using the Masked Autoencoder (MAE) framework (see https://arxiv.org/abs/2504.10021).
             Upload an SAM image to get a prediction of whether the shown LED is still functional and 
             which type of LED is shown in the image.""")
    # Load models
    model1, model2, model3 = _load_models()

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    st.markdown("""
    <style>
        .highlight {
            background-color: #5F9EA0;
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

        quality_classification = max(0,(get_model_outputs(processed_image, model3) + 7)/8)
     
        # Display outputs
        with col2: 
            st.write("**Model Predictions:**")
            st.markdown(f"LED Type: <span class='highlight'>{led_type}</span>", unsafe_allow_html=True)
            st.markdown(f"LED still functional: <span class='highlight'>{min(100, float(quality_classification*100)):.2f}%</span>", unsafe_allow_html=True)
