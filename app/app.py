import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
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
disease_info = {
    "actinic keratosis": {
        "description": "A rough, scaly patch caused by sun exposure.",
        "precautions": [
            "Use sunscreen daily",
            "Avoid excessive sun exposure",
            "Wear protective clothing"
        ],
        "treatment": "Consult a dermatologist for cryotherapy or topical treatment."
    },

    "basal cell carcinoma": {
        "description": "A type of skin cancer that appears as a small bump or lesion.",
        "precautions": [
            "Avoid UV exposure",
            "Use SPF 30+ sunscreen",
            "Regular skin checkups"
        ],
        "treatment": "Requires medical treatment such as surgery or radiation."
    },

    "benign keratosis": {
        "description": "Non-cancerous skin growth.",
        "precautions": [
            "Monitor for changes",
            "Maintain skin hygiene"
        ],
        "treatment": "Usually harmless but consult doctor if changes occur."
    },

    "melanoma": {
        "description": "A serious form of skin cancer.",
        "precautions": [
            "Avoid sunburn",
            "Use sunscreen",
            "Check moles regularly"
        ],
        "treatment": "Immediate medical attention required."
    },

    "nevus": {
        "description": "Common mole, usually harmless.",
        "precautions": [
            "Monitor size and color",
            "Avoid scratching"
        ],
        "treatment": "No treatment needed unless abnormal changes occur."
    }
}


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

    if confidence > 0.8:
        st.success(f"🟢 High Confidence: {labels[pred_index]}")
    elif confidence > 0.5:
        st.warning(f"🟡 Moderate Confidence: {labels[pred_index]}")
    else:
        st.error(f"🔴 Low Confidence: {labels[pred_index]}")
    
    st.markdown("### 👨‍⚕️ Doctor Recommendation")
    high_risk = ["melanoma", "basal cell carcinoma"]
    if pred_class in high_risk:
        st.error("🚨 Immediate medical consultation recommended!")
    else:
        st.info("ℹ️ Regular monitoring is sufficient. Consult doctor if changes occur.")
    
    st.markdown("### 📄 Download Report")
    report = f"""
    Skin Disease Classification Report

    Prediction: {pred_class}
    Confidence: {confidence:.2f}

    Description:
    {disease_info[pred_class]['description']}

    Precautions:
    {', '.join(disease_info[pred_class]['precautions'])}

    Suggested Action:
    {disease_info[pred_class]['treatment']}
    """

    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name="skin_diagnosis_report.txt",
        mime="text/plain"
    )
    
    st.progress(float(confidence))
    st.write(f"Confidence: {confidence:.2f}")
    
    if pred_class in ["melanoma"]:
        st.markdown("### 🔴 High Risk")
    elif pred_class in ["basal cell carcinoma"]:
        st.markdown("### 🟠 Moderate Risk")
    else:
        st.markdown("### 🟢 Low Risk")
        
    
    st.markdown("---")
    st.markdown("## 📊 AI Prediction Dashboard")
    prob_df = pd.DataFrame({
        "Class": labels,
        "Probability": final_pred[0]
    })
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Bar Chart")
        fig, ax = plt.subplots()
        ax.barh(prob_df["Class"], prob_df["Probability"])
        ax.set_xlabel("Probability")
        ax.set_title("Class-wise Confidence")

        st.pyplot(fig)
    
    with col2:
        st.markdown("### 🥧 Distribution")
        pie_fig = px.pie(
            values=final_pred[0],
            names=labels,
            title="Prediction Distribution"
        )

        st.plotly_chart(pie_fig, use_container_width=True)
    
    #PROGRESS BARS
    for i, label in enumerate(labels):
        prob = float(final_pred[0][i])
        st.write(label)
        st.progress(prob)
    
    st.markdown("---")
    st.markdown("## 🩺 Medical Insights & Suggestions")

    info = disease_info[pred_class]

    st.markdown(f"### 📌 About {pred_class}")
    st.write(info["description"])

    st.markdown("### 🛡️ Preventive Measures")
    for p in info["precautions"]:
        st.write(f"- {p}")

    st.markdown("### 💊 Suggested Action")
    st.write(info["treatment"])
    
    st.markdown("### 📈 Detailed Confidence")

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
        
    st.markdown("---")
    st.warning("⚠️ This system is for educational purposes only and should not be used as a substitute for professional medical advice.")