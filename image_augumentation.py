import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img
import random

# -------- CONFIG --------
input_folder = r"D:\Github_Desktop\AI-Powered-Precision-Farming\Crop_Disease_Detection\CauliflowerLeaf\Cauliflower_dataset\Cauli_train\Healthy_leaf"  # specify your input folder here
output_folder = os.path.join(input_folder, "augmented")  # separate folder
os.makedirs(output_folder, exist_ok=True)

# Number of augmented images per original
NUM_AUG_PER_IMAGE = 10  # fixed for consistency

# -------- AUGMENTATION SETTINGS --------
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    fill_mode='nearest'
)

# -------- AUGMENTATION LOOP --------
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for filename in image_files:
    file_path = os.path.join(input_folder, filename)
    img = load_img(file_path)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)

    base_name, ext = os.path.splitext(filename)

    # generate deterministic random seed for reproducibility
    random.seed(hash(filename) % (2**32))

    aug_iter = datagen.flow(x, batch_size=1)

    for i in range(NUM_AUG_PER_IMAGE):
        aug_img = next(aug_iter)[0].astype('uint8')
        aug_filename = f"{base_name}_aug_{i+1}{ext}"
        save_path = os.path.join(output_folder, aug_filename)
        save_img(save_path, aug_img)

print(f"✅ Augmentation complete! {len(image_files) * NUM_AUG_PER_IMAGE} new images saved to: {output_folder}")
