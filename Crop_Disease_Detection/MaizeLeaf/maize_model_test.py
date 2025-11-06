import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ----------------------------
# 1. Load the saved model
# ----------------------------
MODEL_PATH = r"D:\Github_Desktop\AI-Powered-Precision-Farming\Crop_Disease_Detection\MaizeLeaf\maize_disease_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ----------------------------
# 2. Define class names (exact order as your dataset folders)
# ----------------------------
class_names = ["Blight", "Common Rust", "Healthy", "Phosphorus Deficiency"]

# ----------------------------
# 3. Prediction function
# ----------------------------
def predict_image(img_path):
    # Load and preprocess
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # same normalization as training

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    print(f"\n📷 Image: {os.path.basename(img_path)}")
    print(f"Predicted Class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.2f}%")

# ----------------------------
# 4. Example usage
# ----------------------------
if __name__ == "__main__":
    # Change this to your test image path
    img_path = r"D:\Github_Desktop\AI-Powered-Precision-Farming\Crop_Disease_Detection\MaizeLeaf\Maize_test\blight1.jpg"
    
    if os.path.exists(img_path):
        predict_image(img_path)
    else:
        print(f"❌ File not found: {img_path}")
