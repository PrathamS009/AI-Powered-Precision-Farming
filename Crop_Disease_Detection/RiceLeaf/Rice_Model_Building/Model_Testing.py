import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report

model = load_model("RiceLeafDiseasePreTrainedModel.keras")

validation_dir = "D:/PyCharm Community Edition 2024.3.5/PROJECTS/Rice Leaf Disease Prediction/Project 1/RiceLeafDisease - Dataset/Validation"

val_data = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False,
    label_mode="int"
)

AUTOTUNE = tf.data.AUTOTUNE

def preprocess_for_model(image, label):
    image = tf.image.resize(image, (224, 224))
    image = preprocess_input(image)
    return image, label

val_data = val_data.map(preprocess_for_model).prefetch(AUTOTUNE)
y_true = []
for _, label in val_data:
    y_true.extend(label.numpy())

y_true = np.array(y_true)

y_pred_probs = model.predict(val_data)
y_pred = np.argmax(y_pred_probs, axis=1)

class_names = ['Brown Spot', 'Healthy', 'Hispa', 'Leaf Blast', 'Leaf Scald']

print(classification_report(y_true, y_pred, target_names=class_names))
