from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = r"Crop_Disease_Detection/MaizeLeaf/Maize_dataset"  # path to training images
datagen = ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_directory(train_dir)

print("Class indices:", train_gen.class_indices)
