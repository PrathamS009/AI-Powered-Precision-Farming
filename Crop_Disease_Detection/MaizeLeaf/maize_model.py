import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras import layers, models

# Paths
data_dir = r"D:\Github_Desktop\AI-Powered-Precision-Farming\Crop_Disease_Detection\MaizeLeaf\Maize_dataset"  # 4 folders

# Image parameters
IMG_SIZE = 224
BATCH_SIZE = 32

# Only rescaling since images are already augmented
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    subset='validation'
)

# Base Model
base_model = EfficientNetB2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Freeze base layers initially

# Custom Head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(4, activation='softmax')  # 4 classes
])

for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Initial Training
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=25
)

# Fine-tuning: Unfreeze base model
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
history_finetune = model.fit(train_gen, validation_data=val_gen, epochs=5)

# Save Model
model.save("maize_disease_model.h5")
print("✅ Model training complete and saved!")
