# ============================================================
# Rice Leaf Disease Detection - VGG16
# Classes: Brown Spot, Hispa, Leaf Blast, Leaf Scald, Healthy
# Dataset: Kaggle Private Dataset - 'Rice Leaf Disease Dataset'
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME      = "vgg16"
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 16           # VGG16 is heavy, use smaller batch
EPOCHS_PHASE1   = 10
EPOCHS_PHASE2   = 15           # Fewer epochs, VGG16 converges fast
LEARNING_RATE1  = 1e-3
LEARNING_RATE2  = 1e-5
DROPOUT_RATE    = 0.5

# --- UPDATE THESE PATHS for your environment ---
TRAIN_DIR = "/kaggle/input/rice-leaf-disease-dataset/train"
TEST_DIR  = "/kaggle/input/rice-leaf-disease-dataset/test"

# On Google Colab:
# TRAIN_DIR = "/content/drive/MyDrive/Rice_Leaf_Disease_Dataset/train"
# TEST_DIR  = "/content/drive/MyDrive/Rice_Leaf_Disease_Dataset/test"

OUTPUT_DIR = f"/kaggle/working/{MODEL_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["Brown Spot", "Healthy", "Hispa", "Leaf Blast", "Leaf Scald"]
NUM_CLASSES = len(CLASS_NAMES)

# ============================================================
# DATA GENERATORS
# ============================================================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.15
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"\nClass indices: {train_generator.class_indices}")
print(f"Train samples: {train_generator.samples}")
print(f"Val samples:   {val_generator.samples}")
print(f"Test samples:  {test_generator.samples}")

# ============================================================
# CLASS WEIGHTS
# ============================================================
labels = train_generator.classes
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight_dict = dict(enumerate(class_weights))
print(f"\nClass weights: {class_weight_dict}")

# ============================================================
# BUILD MODEL
# Note: VGG16 fine-tuning - we unfreeze only the last conv block
# to avoid overfitting since VGG16 has many parameters
# ============================================================
def build_model(trainable_base=False):
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = False  # Always start frozen

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(DROPOUT_RATE / 2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    return models.Model(inputs, outputs), base_model

# ============================================================
# CALLBACKS
# ============================================================
def get_callbacks(phase):
    return [
        callbacks.ModelCheckpoint(
            filepath=os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_phase{phase}_best.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=4,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.CSVLogger(
            os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_phase{phase}_history.csv")
        )
    ]

# ============================================================
# PHASE 1 — Train top layers only
# ============================================================
print("\n" + "="*50)
print("PHASE 1: Training top layers (base frozen)")
print("="*50)

model, base_model = build_model(trainable_base=False)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE1),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

history1 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE1,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=get_callbacks(phase=1)
)

# ============================================================
# PHASE 2 — Fine-tune last conv block only (VGG16 specific)
# Unfreeze only block5 (last convolutional block)
# ============================================================
print("\n" + "="*50)
print("PHASE 2: Fine-tuning last conv block (block5)")
print("="*50)

# Unfreeze only the last conv block
for layer in base_model.layers:
    if layer.name.startswith('block5'):
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE2),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE2,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=get_callbacks(phase=2)
)

# ============================================================
# SAVE FINAL MODEL
# ============================================================
final_keras_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_final.keras")
model.save(final_keras_path)
print(f"\nModel saved: {final_keras_path}")

# TFLite Export
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
tflite_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_final.tflite")
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)
print(f"TFLite model saved: {tflite_path}")

# ============================================================
# EVALUATION
# ============================================================
print("\nEvaluating on test set...")
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

test_generator.reset()
preds = model.predict(test_generator, verbose=1)
pred_classes = np.argmax(preds, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print("\nClassification Report:")
print(classification_report(true_classes, pred_classes, target_names=class_labels))

# ============================================================
# PLOTS
# ============================================================
def merge_histories(h1, h2, key):
    return h1.history[key] + h2.history[key]

epochs_range = range(1, len(merge_histories(history1, history2, 'accuracy')) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'{MODEL_NAME.upper()} - Training History', fontsize=14, fontweight='bold')

axes[0].plot(epochs_range, merge_histories(history1, history2, 'accuracy'), label='Train Acc')
axes[0].plot(epochs_range, merge_histories(history1, history2, 'val_accuracy'), label='Val Acc')
axes[0].axvline(x=EPOCHS_PHASE1, color='gray', linestyle='--', label='Fine-tune start')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, merge_histories(history1, history2, 'loss'), label='Train Loss')
axes[1].plot(epochs_range, merge_histories(history1, history2, 'val_loss'), label='Val Loss')
axes[1].axvline(x=EPOCHS_PHASE1, color='gray', linestyle='--', label='Fine-tune start')
axes[1].set_title('Loss')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_training_curves.png"), dpi=150)
plt.show()

cm = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title(f'{MODEL_NAME.upper()} - Confusion Matrix\nTest Accuracy: {test_acc:.4f}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_confusion_matrix.png"), dpi=150)
plt.show()

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("Done!")
