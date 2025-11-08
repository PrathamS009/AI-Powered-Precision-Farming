import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

MODEL_PATH = "bajra_leaf_model.h5"
IMG_PATH = r"bajra_dataset\train\Diseased\IMG_2443.jpg"  # Path to test image

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load and preprocess image
img = image.load_img(IMG_PATH, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)[0][0]
label = "Diseased" if pred > 0.5 else "Fresh"

print(f"Prediction: {label} (confidence: {pred:.2f})")
