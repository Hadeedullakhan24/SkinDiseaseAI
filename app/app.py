import streamlit as st
import numpy as np
from utils.gradcam import make_gradcam_heatmap
import cv2
st.set_page_config(page_title="Skin Disease AI", layout="centered")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
    }

    .stFileUploader {
        background-color: #ffffff20;
        padding: 20px;
        border-radius: 12px;
        border: 2px dashed #4CAF50;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }

    h1, h2, h3 {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

from PIL import Image

from keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
# Load models
cnn_model = load_model("models/cnn_model_5class.h5")
eff_model = load_model("models/efficientnet_model_5class.h5", compile=False)

# Labels
labels = ['actinic keratosis', 'basal cell carcinoma', 'benign keratosis', 'melanoma', 'nevus']

st.markdown("""
# 🧠 Skin Disease Classification AI
### Hybrid CNN + EfficientNet Model
Upload a skin lesion image to get prediction
""")

uploaded_file = st.file_uploader("📤 Upload Skin Image", type=["jpg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((160,160))
    st.image(img, caption="Uploaded Image", width=400)

    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    
    # Predictions
    pred_cnn = cnn_model.predict(img_array)
    pred_eff = eff_model.predict(img_array)

    # Hybrid averaging
    final_pred = (pred_cnn + pred_eff) / 2

    pred_index = np.argmax(final_pred)
    confidence = np.max(final_pred)

    #st.markdown("## 🔍 Prediction Result")
    st.markdown("## 🧠 Diagnosis Result")
    pred_class = labels[pred_index]

    if pred_class == "melanoma":
        st.error(f"⚠️ High Risk: {pred_class}")
    elif pred_class == "basal cell carcinoma":
        st.warning(f"⚠️ Moderate Risk: {pred_class}")
    else:
        st.success(f"✅ Low Risk: {pred_class}")
        
    st.progress(float(confidence))
    st.write(f"Confidence: {confidence:.2f}")
    
    st.markdown("## 📊 Class Probabilities")
    for i, label in enumerate(labels):
        st.write(f"{label}: {final_pred[0][i]:.2f}")
    st.markdown("---")
    st.markdown("## 🔥 Model Attention (Grad-CAM)")
    # Generate heatmap from EfficientNet
    heatmap = make_gradcam_heatmap(
        img_array,
        eff_model,
        last_conv_layer_name="top_conv"
    )

    # Resize heatmap
    heatmap = cv2.resize(heatmap, (160,160))
    heatmap = np.uint8(255 * heatmap)

    # Apply color map
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose heatmap on original image
    superimposed_img = heatmap * 0.4 + img_array[0]*255

    #Show image
    st.image(superimposed_img.astype("uint8"), caption="Grad-CAM Heatmap")
    
    st.progress(float(confidence))
    st.markdown("### 📊 Class Probabilities")

    for i, label in enumerate(labels):
        st.write(f"{label}: {final_pred[0][i]:.2f}")