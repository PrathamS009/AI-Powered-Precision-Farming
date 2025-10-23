import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img
import random

# -------- CONFIG --------
input_folder = r"D:\Github_Desktop\AI-Powered-Precision-Farming\Crop_Disease_Detection\MaizeLeaf\Maize_dataset\Phosphorus_Deficiency"
num_aug_per_image = random.randint(3, 4)
# ------------------------

# define augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    fill_mode='nearest'
)

# create augmented images
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        file_path = os.path.join(input_folder, filename)
        print(f"Augmenting {filename} ...")

        img = load_img(file_path)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # create augmentation iterator
        aug_iter = datagen.flow(x, batch_size=1)

        # generate and save new images
        for i in range(num_aug_per_image):
            aug_img = next(aug_iter)[0].astype('uint8')
            base_name, ext = os.path.splitext(filename)
            save_path = os.path.join(input_folder, f"{base_name}_aug_{i+1}{ext}")
            save_img(save_path, aug_img)

print("✅ Augmentation complete! All new images saved in the same folder.")
