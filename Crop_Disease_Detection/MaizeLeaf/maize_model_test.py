import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ===================== CONFIG =====================
model_path = 'maize_disease_model.h5'
img_path = 'dataset/test/Blight/xyz.jpg'  # <-- change to your image path
img_size = (224, 224)
# ==================================================

# Load model
model = tf.keras.models.load_model(model_path)

# Get class labels
train_dir = 'Maize_dataset'
class_names = sorted(os.listdir(train_dir))
print("Classes:", class_names)

# Load and preprocess image
img = image.load_img(img_path, target_size=img_size)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
pred_class = np.argmax(pred, axis=1)[0]
pred_label = class_names[pred_class]

print(f"Predicted class: {pred_label}")
