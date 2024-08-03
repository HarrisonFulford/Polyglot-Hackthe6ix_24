import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import pathlib
from tensorflow import keras
print("TensorFlow version:", tf.__version__)

# Set dimensions for images
img_height = 128
img_width = 128

# Loading datasets for finger detection
train_ds = keras.preprocessing.image_dataset_from_directory(
    'hands',
    "HandsTraining",
    validation_split=0.2,
    labels="inferred",
    label_mode="int",
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    subset="training"
)

val_ds = keras.preprocessing.image_dataset_from_directory(
    'hands',
    "HandsValidation",
    validation_split=0.2,
    labels="inferred",
    label_mode="int",
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    subset="validation"
)

class_names = train_ds.class_names
print(class_names)

data_iterator = train_ds.as_numpy_iterator()


# Scale pixel values from (0 ~ 225) --> (0 ~ 1)


# Training the Model

# Reducing the # of images fed to the model - reduces time and saves resources


# Defining the Neural Network

# Using the sequential model - process layers one by one
model = keras.models.Sequential([
    #Conv2D extracts detects the many different features of an image (Ex: edges, patterns, textures)
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    # MaxPooling2D extracts the most prominent parts and dampens the less important parts, reducing computational load and the 
    # image in question to be more obvious. 
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    
    # Flattening - reshaping a multi-dimensional array into a one-dimensional array
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

"""
model.fit(x_train, y_train, epochs=10)

# Checking the model's performance
model.evaluate(x_test, y_test)
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

#
model.save('image_classifier.keras')
"""
