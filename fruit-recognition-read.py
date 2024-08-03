import pickle
import PIL
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import pathlib
import tensorflow
import keras
from keras import layers
from keras import Sequential
from keras import Model

batch_size = 64 #Default: 64
img_height = 180 #Default: 180
img_width = 180 #Default: 180
epochs = 7 #Default:5
print("Loading")
with open('fruitTest.npy', "rb") as file_pi:
    history = pickle.load(file_pi)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
