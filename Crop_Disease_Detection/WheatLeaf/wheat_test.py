import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model("wheat_disease_multiclass_model.h5")

img = image.load_img(r"wheat_dataset\val\black_rust\black_rust_0.png", target_size=(224, 224))
x = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

pred = model.predict(x)[0]
classes = ["black_rust", "brown_rust", "yellow_rust", "healthy"]

print("Predicted:", classes[np.argmax(pred)])
print("Probabilities:", dict(zip(classes, [float(p) for p in pred])))
