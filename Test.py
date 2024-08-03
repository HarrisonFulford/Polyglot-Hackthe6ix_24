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

model = keras.saving.load_model("fruitModel.keras")
#model = Model.load("fruitModel.keras")

batch_size = 64 #Default: 64
img_height = 180 #Default: 180
img_width = 180 #Default: 180
epochs = 7 #Default:5
train_ds = tf.keras.utils.image_dataset_from_directory(
  'testData',
  validation_split=0.2, #Default: 0.2 
  subset="training",
  seed=123,
  class_names=["apple", "banana", "mango", "orange"],
  image_size=(img_height, img_width),
  batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
  'testData',
  validation_split=0.2, #Default: 0.2
  subset="validation",
  seed=123,
  class_names=["apple", "banana", "mango", "orange"],
  image_size=(img_height, img_width),
  batch_size=batch_size)
class_names = train_ds.class_names

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

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
