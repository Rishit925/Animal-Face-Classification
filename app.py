import streamlit as st
from PIL import Image
import tempfile
import os
from predict import predict_image


# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Animal Face Classifier",
    page_icon="🐾",
    layout="centered"
)

# ---------------------- SIDEBAR ----------------------
st.sidebar.title("📌 About")
st.sidebar.info(
    """
    **Animal Face Classifier**

    This application classifies animal faces into one of the trained categories using a Convolutional Neural Network (CNN).

    **Frameworks Used**
    - PyTorch
    - Streamlit
    - PIL
    """
)

# ---------------------- MAIN TITLE ----------------------
st.title("🐾 Animal Face Classifier")
st.write(
    "Upload an image of an animal face and click **Predict** to classify it."
)

st.divider()

# ---------------------- FILE UPLOADER ----------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Predict"):

        with st.spinner("Predicting..."):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                image.save(temp_file.name)
                temp_path = temp_file.name

            
            label, confidence = predict_image(temp_path)

            os.remove(temp_path)

        st.divider()

        # Display confidence (avoid showing 100.00%)
        display_confidence = min(confidence * 100, 99.9)

        # Show prediction
        st.success(f"### Prediction: {label.capitalize()}")

        # Show confidence level
        if confidence >= 0.90:
            st.info("🟢 Confidence Level: High")
        elif confidence >= 0.70:
            st.info("🟡 Confidence Level: Medium")
        else:
            st.warning(
        "🔴 Confidence Level: Low\n\n"
        "The uploaded image may be different from the images used during training."
    )

# Show confidence score
        st.caption(f"Confidence Score: {display_confidence:.1f}%")

# Progress bar
        st.progress(confidence)

st.divider()

st.caption(
    "Built with using PyTorch, Streamlit and FastAPI"
)