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
ds = keras.utils.image_dataset_from_directory('hands')

class_names = ds.class_names

data_iterator = ds.as_numpy_iterator()
batch = data_iterator.next()

# Scale pixel values from (0 ~ 225) --> (0 ~ 1)
ds = ds.map(lambda x, y: (x/255, y))

# Splitting the Dataset into Training / Validation / Testing Purposes
train_size = int(len(ds) * 0.7)
val_size = int(len(ds) * 0.2)
test_size = int(len(ds) * 0.1) + 1

# Deciding how many batches of the data are allocated for each purpose
train = ds.take(train_size)
val = ds.skip(train_size).take(val_size)
test = ds.skip(train_size + val_size).take(test_size)

# Training the Model
# Defining the Neural Network

# Using the sequential model - process layers one by one
model = keras.models.Sequential([
    #Conv2D extracts detects the many different features of an image (Ex: edges, patterns, textures)
    # 16 filters, 3x3 pixel filter. Relu converts all negative outputs to 0 and preserves the postive outputs.
    keras.layers.Conv2D(16, (3, 3), 1, activation='relu', input_shape=(128, 128, 3)), # Input layer
    # MaxPooling2D extracts the most prominent parts and dampens the less important parts, reducing computational load and the 
    # image in question to be more obvious. 
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, (3, 3), 1, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(16, (3, 3), 1, activation='relu'),
    
    # Flattening - reshaping a multi-dimensional array into a one-dimensional array
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(train, epochs=20, validation_data=val)


