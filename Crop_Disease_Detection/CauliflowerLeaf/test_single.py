import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from vit_keras import vit, layers

# --- CONFIG ---
MODEL_PATH = "final_cauliflower_model.h5"
IMG_PATH = r"D:\Github_Desktop\AI-Powered-Precision-Farming\Crop_Disease_Detection\CauliflowerLeaf\Cauliflower_dataset\Cauli_test\leaf2.jpg"
IMAGE_SIZE = 224

# --- LOAD MODEL ---
custom_objects = {
    'ClassToken': layers.ClassToken,
    'AddPositionEmbs': layers.AddPositionEmbs,
    'TransformerBlock': layers.TransformerBlock
}
model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
print("✅ Model loaded successfully.")

# --- LOAD AND PREPROCESS IMAGE ---
img = image.load_img(IMG_PATH, target_size=(IMAGE_SIZE, IMAGE_SIZE))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

# --- PREDICT ---
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]

# --- CLASS LABELS ---
class_indices = ['Bacterial_spot_rot', 'Black_rot', 'Downy_mildew', 'Healthy_cauli', 'Healthy_leaf']  # your class order

print(f"🖼️ Image: {IMG_PATH}")
print(f"🔍 Predicted Class: {class_indices[predicted_class]}")
