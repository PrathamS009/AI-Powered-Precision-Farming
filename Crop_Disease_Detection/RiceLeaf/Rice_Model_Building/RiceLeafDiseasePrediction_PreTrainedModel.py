import tensorflow.keras.utils
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pickle

train_dir = "D:/PyCharm Community Edition 2024.3.5/PROJECTS/Rice Leaf Disease Prediction/Project 1/RiceLeafDisease - Dataset/Train"
validation_dir = "D:/PyCharm Community Edition 2024.3.5/PROJECTS/Rice Leaf Disease Prediction/Project 1/RiceLeafDisease - Dataset/Validation"

Base_Model = MobileNetV2(
    input_shape = (224, 224, 3),
    include_top = False,
    weights = "imagenet"
)

Base_Model.trainable = False

Model = models.Sequential([
    Base_Model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(5, activation="softmax")
])

Model.compile(
    optimizer = Adam(learning_rate = 0.0001),
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)


Train_Data = tensorflow.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size = 32,
    shuffle = True,
    label_mode = "int"
)

Validation_Data = tensorflow.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size = (224, 224),
    batch_size = 32,
    label_mode = "int"
)

def preprocess_for_model(image, label):
    image = tensorflow.image.resize(image, (224, 224))
    image = preprocess_input(image)
    return image, label

AUTOTUNE = tensorflow.data.AUTOTUNE
Train_Data = (Train_Data.map(preprocess_for_model, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE))
Validation_Data = (Validation_Data.map(preprocess_for_model, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE))

Base_Model.trainable = True

Model.compile(
    optimizer = Adam(learning_rate=0.00001),
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

Model_Training = Model.fit(
    Train_Data,
    epochs = 10,
    validation_data = Validation_Data
)

with open("Model_PreTrained_History.pkl","wb") as f:
    pickle.dump(Model_Training.history, f)

Model.save("RiceLeafDiseasePreTrainedModel.keras")