import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="X-Ray Fracture Detection",
    page_icon="🩻",
    layout="centered"
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_SIZE = (224, 224)
MODEL_PATH = "fracture_model.keras"

# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🩻 X-Ray Fracture Detection")

st.write(
    "Upload an X-ray image and the AI model will predict "
    "whether the image is classified as fractured or not fractured."
)

st.warning(
    "⚠️ This is an AI demonstration and should not be used "
    "for clinical diagnosis."
)

# --------------------------------------------------
# File uploader
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an X-ray image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if uploaded_file is not None:

    # Load image
    image = Image.open(uploaded_file).convert("RGB")

    # Display image
    st.subheader("Uploaded X-Ray")

    st.image(
        image,
        caption="X-Ray Image",
        width="stretch"
    )

    # Predict button
    if st.button("🔍 Analyze X-Ray", type="primary"):

        with st.spinner("Analyzing X-ray..."):

            # Resize image
            image_resized = image.resize(IMAGE_SIZE)

            # Convert to numpy
            image_array = np.array(image_resized)

            # Normalize
            image_array = image_array / 255.0

            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)

            # Prediction
            probability = model.predict(
                image_array,
                verbose=0
            )[0][0]

        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        if probability >= 0.5:

            prediction = "NOT FRACTURED"
            confidence = probability

        else:

            prediction = "FRACTURED"
            confidence = 1 - probability

        st.subheader("Prediction")

        if prediction == "FRACTURED":

            st.error(
                f"🦴 FRACTURED\n\n"
                f"Confidence: {confidence:.2%}"
            )

        else:

            st.success(
                f"✅ NOT FRACTURED\n\n"
                f"Confidence: {confidence:.2%}"
            )

        # --------------------------------------------------
        # Probability information
        # --------------------------------------------------

        st.subheader("Prediction Details")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Fracture Probability",
                f"{(1 - probability):.2%}"
            )

        with col2:

            st.metric(
                "Not Fracture Probability",
                f"{probability:.2%}"
            )

        # --------------------------------------------------
        # Progress bar
        # --------------------------------------------------

        st.write("Model confidence")

        st.progress(float(confidence))