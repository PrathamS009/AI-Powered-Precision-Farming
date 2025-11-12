import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from vit_keras import vit, layers

# --- CONFIG ---
MODEL_PATH = "cauli_model.h5"
DATASET_DIR = r"D:\Github_Desktop\AI-Powered-Precision-Farming\Crop_Disease_Detection\CauliflowerLeaf\Cauliflower_dataset\Cauli_train"
IMAGE_SIZE = 224
BATCH_SIZE = 16

# --- LOAD MODEL WITH VIT CUSTOM LAYERS ---
custom_objects = {
    'ClassToken': layers.ClassToken,
    'AddPositionEmbs': layers.AddPositionEmbs,
    'TransformerBlock': layers.TransformerBlock
}

model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
print("✅ Model loaded successfully.")

# --- GENERATOR FOR TEST DATA ---
datagen = ImageDataGenerator(rescale=1.0/255)
test_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# --- EVALUATION ---
loss, acc = model.evaluate(test_gen)
print(f"\n📊 Test Loss: {loss:.4f}")
print(f"📈 Test Accuracy: {acc:.4f}")

# --- PREDICTIONS + METRICS ---
predictions = model.predict(test_gen, verbose=1)
true_labels = test_gen.classes
predicted_labels = np.argmax(predictions, axis=1)

print("\n--- Classification Report ---")
print(classification_report(true_labels, predicted_labels, target_names=test_gen.class_indices.keys()))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(true_labels, predicted_labels))
