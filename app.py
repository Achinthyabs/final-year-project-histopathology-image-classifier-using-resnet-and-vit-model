import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -----------------------------
# Load model
# -----------------------------
model = tf.keras.models.load_model("resnet8_12class_model.keras")

train_dir = r"C:\Users\achin\Desktop\CNN\Processed_Dataset_Split\train"
class_names = sorted(os.listdir(train_dir))

print(class_names)

# -----------------------------
# Page UI
# -----------------------------
st.set_page_config(page_title="Histopathology CNN Prediction", layout="centered")

st.title("Histopathology Image Classification")
st.write("Upload a histopathology image to predict its tissue class.")

uploaded_file = st.file_uploader(
    "Upload image here",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    st.subheader("Prediction Result")
    st.success(f"Predicted Class: {predicted_class}")
    st.info(f"Confidence Level: {confidence:.2f}%")
    if confidence < 70:
     st.warning("Low confidence prediction. Image may be confusing or needs expert verification.")
    else:
        st.success(f"Predicted Class: {predicted_class}")

    # Top 3 predictions
    st.subheader("Top 3 Class Probabilities")

    top_3 = np.argsort(prediction[0])[-3:][::-1]

    for i in top_3:
        st.write(f"{class_names[i]}: {prediction[0][i] * 100:.2f}%")
        st.progress(float(prediction[0][i]))

    # Overall model performance
    st.subheader("Overall CNN Model Performance")

    st.write("These values are calculated from the full test dataset, not from a single image.")

    st.metric("Test Accuracy", "92.32%")
    st.metric("Precision", "Use classification report")
    st.metric("Recall", "Use classification report")
    st.metric("F1-score", "Use classification report")