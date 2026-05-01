# ============================================================
# Rice Leaf Disease Detection - ResNet50 (FIXED)
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================================================
# CONFIG
# ============================================================
MODEL_NAME      = "resnet50"
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 32
EPOCHS_PHASE1   = 10
EPOCHS_PHASE2   = 20
LEARNING_RATE1  = 1e-3
LEARNING_RATE2  = 1e-5
DROPOUT_RATE    = 0.5

TRAIN_DIR  = r"E:\GitHub_Desktop\AI-Powered-Precision-Farming\Crop_Disease_Detection\RiceLeaf\Rice_dataset\Train"
TEST_DIR   = r"E:\GitHub_Desktop\AI-Powered-Precision-Farming\Crop_Disease_Detection\RiceLeaf\Rice_dataset\Validation"
OUTPUT_DIR = rf"E:\GitHub_Desktop\AI-Powered-Precision-Farming\Crop_Disease_Detection\RiceLeaf\working\{MODEL_NAME}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# DATA GENERATORS
# NOTE: ResNet uses preprocess_input (NO rescale)
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

CLASS_NAMES = list(train_generator.class_indices.keys())
NUM_CLASSES = len(CLASS_NAMES)

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
# ============================================================
def build_model(trainable_base=False):
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    base_model.trainable = trainable_base

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=trainable_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(DROPOUT_RATE / 2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    return models.Model(inputs, outputs)

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
# PHASE 1 — Train top layers
# ============================================================
print("\n" + "="*50)
print("PHASE 1: Training top layers")
print("="*50)

model = build_model(trainable_base=False)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE1),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history1 = model.fit(
    train_generator,
    epochs=EPOCHS_PHASE1,
    validation_data=val_generator,
    class_weight=class_weight_dict,
    callbacks=get_callbacks(phase=1)
)

# ============================================================
# PHASE 2 — Fine-tune last 30 layers (IMPORTANT FIX)
# ============================================================
print("\n" + "="*50)
print("PHASE 2: Fine-tuning last 30 layers")
print("="*50)

for layer in model.layers[:-30]:
    layer.trainable = False
for layer in model.layers[-30:]:
    layer.trainable = True

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
# SAVE MODEL
# ============================================================
final_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_final.keras")
model.save(final_path)

# ============================================================
# EVALUATION
# ============================================================
print("\nEvaluating...")
test_loss, test_acc = model.evaluate(test_generator)
print(f"Accuracy: {test_acc:.4f}")

test_generator.reset()
preds = model.predict(test_generator)
pred_classes = np.argmax(preds, axis=1)
true_classes = test_generator.classes

print(classification_report(true_classes, pred_classes, target_names=CLASS_NAMES))

# ============================================================
# PLOTS (Training Curves + Confusion Matrix)
# ============================================================

def merge_histories(h1, h2, key):
    return h1.history[key] + h2.history[key]

epochs_range = range(1, len(merge_histories(history1, history2, 'accuracy')) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'{MODEL_NAME.upper()} - Training History', fontsize=14, fontweight='bold')

# Accuracy
axes[0].plot(epochs_range, merge_histories(history1, history2, 'accuracy'), label='Train Acc')
axes[0].plot(epochs_range, merge_histories(history1, history2, 'val_accuracy'), label='Val Acc')
axes[0].axvline(x=EPOCHS_PHASE1, color='gray', linestyle='--', label='Fine-tune start')
axes[0].set_title('Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
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

# ============================================================
# CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(true_classes, pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Purples',
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.title(f'{MODEL_NAME.upper()} - Confusion Matrix\nTest Accuracy: {test_acc:.4f}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_confusion_matrix.png"), dpi=150)
plt.show()

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("Done!")