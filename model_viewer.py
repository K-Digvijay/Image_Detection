import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the model
MODEL_PATH = r"D:\Image_Detection\DL_Model_DenseNet121.h5"  # Use raw string for file path
model = load_model(MODEL_PATH)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

st.title("H5 Model Viewer and Predictor")

# Display model summary
st.subheader("Model Summary")
model_summary = []
model.summary(print_fn=lambda x: model_summary.append(x))
st.code("\n".join(model_summary), language="text")

# Upload an image for prediction
st.subheader("Upload an Image for Prediction")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Preprocess image (adjust to model input size)
    image = image.resize((100, 100))  # Example resize to 224x224, adjust if necessary
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Predict
    prediction = model.predict(image_array)
    
    # For binary classification with a sigmoid output (0 to 1)
    probability = prediction[0][0]  # For binary output, we have one value
    class_label = "Real" if probability > 0.5 else "Fake"
    
    # Display prediction and probability
    st.subheader("Prediction Output")
    st.write(f"The image is predicted as: **{class_label}**")
    st.write(f"Prediction Probability: **{probability:.2f}**")

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
