import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

Model = tf.keras.models.load_model("RiceLeafDiseasePreTrainedModel.keras", compile=False)

class_names = ['Brown Spot', 'Healthy', 'Hispa', 'Leaf Blast', 'Leaf Scald']

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224,224))
    image = np.array(image).astype("float32")/255.0
    image = np.expand_dims(image, axis=0)
    return image

st.title("🌾🍙Rice Leaf🥬 Disease Prediction")
st.write("Upload an Image of Rice Leaf to check for Diseases")

uploaded_file = st.file_uploader("Choose an Leaf Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    try:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    except TypeError:
        st.image(image, caption="Uploaded Image")

    processed_image = preprocess_image(image)
    prediction = Model.predict(processed_image)[0]
    prediction_index = np.argmax(prediction)
    predicted_class = class_names[prediction_index]
    confidence = round(prediction[prediction_index]*100, 2)

    st.success(f"Predicted Disease is: {predicted_class} with Confidence of {confidence}%")