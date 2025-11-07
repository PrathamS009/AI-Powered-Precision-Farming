import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from vit_keras import vit
import pandas as pd
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
DATASET_DIR = r"D:\Github_Desktop\AI-Powered-Precision-Farming\Crop_Disease_Detection\MaizeLeaf\Maize_dataset"  # update your path
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 8
LR = 1e-4
# ==================================================

# ===================== AUGMENTATION FUNCTION =====================
def data_augment(image):
    p_spatial = tf.random.uniform([], 0, 1.0)
    p_rotate = tf.random.uniform([], 0, 1.0)
    p_pixel_1 = tf.random.uniform([], 0, 1.0)
    p_pixel_2 = tf.random.uniform([], 0, 1.0)
    p_pixel_3 = tf.random.uniform([], 0, 1.0)

    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    if p_spatial > 0.75:
        image = tf.image.transpose(image)
    if p_rotate > 0.75:
        image = tf.image.rot90(image, k=3)
    elif p_rotate > 0.5:
        image = tf.image.rot90(image, k=2)
    elif p_rotate > 0.25:
        image = tf.image.rot90(image, k=1)

    if p_pixel_1 >= 0.4:
        image = tf.image.random_saturation(image, 0.7, 1.3)
    if p_pixel_2 >= 0.4:
        image = tf.image.random_contrast(image, 0.8, 1.2)
    if p_pixel_3 >= 0.4:
        image = tf.image.random_brightness(image, 0.1)
    return image

# ===================== DATA GENERATORS =====================
datagen = ImageDataGenerator(
    rescale=1.0/255,
    samplewise_center=True,
    samplewise_std_normalization=True,
    validation_split=0.2,
    preprocessing_function=data_augment
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    subset="validation"
)

# ===================== MODEL =====================
vit_model = vit.vit_b32(
    image_size=IMAGE_SIZE,
    activation='softmax',
    pretrained=True,
    include_top=False,
    pretrained_top=False,
    classes=train_gen.num_classes
)

model = tf.keras.Sequential([
    vit_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation=tfa.activations.gelu),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(train_gen.num_classes, activation='softmax')
])

optimizer = tfa.optimizers.RectifiedAdam(learning_rate=LR)
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
    metrics=['accuracy']
)

# ===================== CALLBACKS =====================
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, mode='max'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=2, min_lr=1e-6, mode='max'),
    tf.keras.callbacks.ModelCheckpoint('maize_vit_best.h5', save_best_only=True, monitor='val_accuracy', mode='max')
]

# ===================== TRAIN =====================
steps_train = train_gen.samples // train_gen.batch_size
steps_val = val_gen.samples // val_gen.batch_size

history = model.fit(
    train_gen,
    steps_per_epoch=steps_train,
    validation_data=val_gen,
    validation_steps=steps_val,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ===================== SAVE HISTORY =====================
history_df = pd.DataFrame(history.history)
history_df.to_csv("maize_training_history.csv", index=False)
print("📁 Training history saved as 'maize_training_history.csv'")

# ===================== PLOTS =====================
plt.figure(figsize=(10, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("maize_training_curves.png", dpi=300)
plt.show()

print("📊 Plots saved as 'maize_training_curves.png'")

# ===================== SAVE MODEL =====================
model.save('final_maize_vit_model.h5')
print("✅ Model training complete. Saved as 'final_maize_vit_model.h5'")
