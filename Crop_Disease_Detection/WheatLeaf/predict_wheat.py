# predict_wheat.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

MODEL_PATH = "checkpoints/wheat_mobilenetv2.keras"
CLASS_NAMES_FILE = "checkpoints/class_names.txt"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

with open(CLASS_NAMES_FILE, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

img_size = (224, 224)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0


    pred = model.predict(img_array)
    class_index = np.argmax(pred, axis=1)[0]
    label = class_names[class_index]

    plt.imshow(img)
    plt.title(f"Prediction: {label}")
    plt.axis("off")

    plt.show()

    return label

predict_image("data/test/yellow_rust_test/yellow_rust_test_43.png")
