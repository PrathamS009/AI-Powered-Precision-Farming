# test_ova_voting.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# ========================
# CONFIGURATION
# ========================
MODELS_DIR = "."
DISEASES = ["black_rust", "brown_rust", "healthy", "yellow_rust"]
IMG_SIZE = (224, 224)

# ========================
# LOAD MODELS
# ========================
models = {}
for disease in DISEASES:
    path = os.path.join(MODELS_DIR, f"{disease}_binary_model.h5")
    models[disease] = tf.keras.models.load_model(path)
print("All models loaded successfully.\n")

# ========================
# PREDICTION FUNCTION
# ========================
def predict_disease(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

    probabilities = {}
    for disease, model in models.items():
        prob = float(model.predict(arr, verbose=0)[0][0])
        probabilities[disease] = prob

    best_disease = max(probabilities, key=probabilities.get)
    return best_disease, probabilities

# ========================
# TEST EXAMPLE
# ========================
# test_img = "wheat_dataset/test/brown_rust/brown_rust_test_0.png"
test_img = "wheat_dataset/test/black_rust/black_rust_0.png"  # replace with your test image
# test_img = "wheat_dataset/test/yellow_rust/yellow_rust_test_0.png"  # replace with your test image
# test_img = "wheat_dataset/test/healthy/healthy_test_0.png"  # replace with your test image
  # replace with your test image
predicted, probs = predict_disease(test_img)

print("Predicted Disease:", predicted)
for d, p in probs.items():
    print(f"{d}: {p:.4f}")
