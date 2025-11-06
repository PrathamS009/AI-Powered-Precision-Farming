# brinjal_test_single.py

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# -------------------------
# Load trained model
# -------------------------
model = tf.keras.models.load_model('brinjal_model.h5')

# -------------------------
# Class labels (must match training order)
# -------------------------
class_labels = ['diseased', 'fresh']  # change order if reversed in training

# -------------------------
# Function to test single image
# -------------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0][0]
    predicted_class = 'fresh' if prediction > 0.5 else 'diseased'

    print(f"Image: {os.path.basename(img_path)}")
    print(f"Predicted: {predicted_class}")
    print(f"Confidence: {prediction:.4f}")

# -------------------------
# Example usage
# -------------------------
# Replace with your validation image path
test_image_path = r'D:\Github_Desktop\AI-Powered-Precision-Farming\Crop_Disease_Detection\BrinjalLeaf\Brinjal_dataset\Validation\Fresh\92.jpg'
predict_image(test_image_path)
