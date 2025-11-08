# train_ova_models.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import os

# ========================
# CONFIGURATION
# ========================
BASE_DIR = "wheat_dataset/train"
VAL_DIR = "wheat_dataset/val"
DISEASES = ["black_rust", "brown_rust", "healthy", "yellow_rust"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# ========================
# BINARY GENERATOR (No data duplication)
# ========================
def get_binary_generators(disease_name):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        BASE_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    class_index = train_gen.class_indices[disease_name]

    def to_binary(generator):
        for batch_x, batch_y in generator:
            y_binary = (batch_y[:, class_index] == 1).astype("float32")
            yield batch_x, y_binary

    return to_binary(train_gen), to_binary(val_gen), len(train_gen), len(val_gen)

# ========================
# MODEL CREATION FUNCTION
# ========================
def create_model():
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=out)

    for layer in base.layers:
        layer.trainable = False

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ========================
# TRAINING LOOP
# ========================
for disease in DISEASES:
    print(f"\n=== Training model for {disease.upper()} ===")
    train_gen, val_gen, steps_train, steps_val = get_binary_generators(disease)

    model = create_model()
    model.fit(
        train_gen,
        steps_per_epoch=steps_train,
        validation_data=val_gen,
        validation_steps=steps_val,
        epochs=EPOCHS
    )

    model.save(f"{disease}_binary_model.h5")
    print(f"Model saved: {disease}_binary_model.h5")
