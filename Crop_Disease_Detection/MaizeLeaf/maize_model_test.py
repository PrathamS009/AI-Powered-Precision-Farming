import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# ----------------------------
# 1. Load the saved model
# ----------------------------
MODEL_PATH = "D:\Github_Desktop\AI-Powered-Precision-Farming\Crop_Disease_Detection\MaizeLeaf\maize_disease_model.h5"  # change if you saved with another name
model = tf.keras.models.load_model(MODEL_PATH)

# ----------------------------
# 2. Define class names (in same order as training)
# ----------------------------
class_names = ["Blight", "Common Rust", "Healthy", "Phosphorus Deficiency"]

# ----------------------------
# 3. Function to make prediction
# ----------------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # same size used in training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize if used in training

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    print(f"Predicted Class: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.2f}%")

# ----------------------------
# 4. Example usage
# ----------------------------
if __name__ == "__main__":
    img_path = r"D:\Github_Desktop\AI-Powered-Precision-Farming\Crop_Disease_Detection\MaizeLeaf\Maize_test\blight1.jpg"  # put your image path here
    predict_image(img_path)
