import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
print("TensorFlow version:", tf.__version__)

# Call a load function to store data in tuples
# x = images, y = labels
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# class names list to visualize 16 images out of the total dataset
# assign numerical labels with actual keywords
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'] 

# Scale pixel values from (0 ~ 225) --> (0 ~ 1)
x_train = x_train / 255
x_test =  x_test / 255


# Training the Model

# Reducing the # of images fed to the model - reduces time and saves resources
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:2000]
y_test = y_test[:2000]

#defining the neural network 

# 32 neurons, (3,3) convolution matrix filter, activation function off relu, 32x32 res with 3 color channels
# A convolution layer filters image for prominent features

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

# Checking the model's performance
model.evaluate(x_test, y_test)
loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

#
model.save('image_classifier.keras')
