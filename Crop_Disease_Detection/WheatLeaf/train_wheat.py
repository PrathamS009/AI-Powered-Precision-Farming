# train_wheat.py
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import argparse

def build_model(num_classes, input_shape=(224,224,3), base_trainable=False):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = base_trainable
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1./127.5, offset=-1)(inputs)  # [-1,1] for MobileNet
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model

def main(args):
    train_dir = os.path.join(args.data_dir, "train")
    val_dir   = os.path.join(args.data_dir, "val")  
    img_size = (224,224)
    batch_size = args.batch_size
    seed = 123

    # Load datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, labels='inferred', label_mode='int',
        image_size=img_size, batch_size=batch_size, shuffle=True, seed=seed
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, labels='inferred', label_mode='int',
        image_size=img_size, batch_size=batch_size, shuffle=False, seed=seed
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Detected classes:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.08),
    ])
    def augment(images, labels):
        return data_augmentation(images, training=True), labels
    train_ds_aug = train_ds.map(augment, num_parallel_calls=AUTOTUNE)

    model = build_model(num_classes, input_shape=(img_size[0], img_size[1], 3), base_trainable=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Callbacks
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, "epoch_{epoch:02d}_model.keras")
    checkpoint_cb = ModelCheckpoint(
        checkpoint_path,
        save_weights_only=False,
        save_freq='epoch', 
        verbose=1
    )
    early_cb = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)

    history = model.fit(
        train_ds_aug,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[checkpoint_cb, early_cb, reduce_lr]
    )

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    model.save(args.output_model)
    print(f"Saved final model to {args.output_model}")

    with open(os.path.join(args.checkpoint_dir, "class_names.txt"), "w") as f:
        f.write("\n".join(class_names))
    print("Saved class names to checkpoint dir.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data folder containing train/validate subfolders")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_model", type=str, default="checkpoints/wheat_mobilenetv2.keras")
    args = parser.parse_args()
    main(args)
