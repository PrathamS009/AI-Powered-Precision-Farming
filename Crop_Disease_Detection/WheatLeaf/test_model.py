import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("checkpoints/wheat_mobilenetv2.keras")

# Load class names
with open("checkpoints/class_names.txt", "r") as f:
    class_names = f.read().splitlines()

print("✅ Model and classes loaded successfully!")
