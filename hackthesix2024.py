import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# Call a load function to store data in tuples
cifar10 = tf.keras.datasets.cifar10
(training_images, training_labels), (testing_images, testing_labels) = cifar10.load_data()

# Scale pixel values from (0 ~ 225) --> (0 ~ 1)
training_images, testing_images = training_images / 255.0, testing_images / 255.0

# class names list to visualize 16 images out of the total dataset
# assign numerical labels with actual keywords
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'] # order is important!

# Creating a plot 
for i in range(16): 
    plt.subplot(4,4,i+1) # 4x4 grid with 
    # Removes coordinate system as they are not useful
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

# Displaying the plot 
plt.show() 

# Training the Model

# Reducing the # of images fed to the model - reduces time and saves resources
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

#defining the neural network 

# 32 neurons, (3,3) convolution matrix filter, activation function off relu, 32x32 res with 3 color channels
# A convolution layer filters image for prominent features

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
])

tf.keras.layers.Flatten()
tf.keras.layers.Dense(64, activation='relu')
tf.keras.layers.Dense(10, activation='softmax')


loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10)
# Checking the model's performance
model.evaluate(testing_images, testing_labels)
loss, accuracy = model.evaluate(testing_images, testing_labels, verbose=2)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")





