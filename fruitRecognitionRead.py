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
img_height = 128 #Default: 180
img_width = 128 #Default: 180
epochs = 7 #Default:5
print("Loading")
new_model = tf.keras.models.load_model('fruitModel.keras')

# Show the model architecture
new_model.summary()

frameno = 0
ssnum = 0
framesPerPhoto = 3 #How often a photo will be taken (Per frame)
photoType = '.jpg' #Photo type (png, jpg, etc)
class_names = ["apple", "banana", "mango", "orange"]
def checkImg():
  cam_path = "screenshots/currentFrame" + photoType
  img = tf.keras.utils.load_img(
      cam_path, target_size=(img_height, img_width)
  )
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = new_model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
  )
checkImg()
"""acc = history['accuracy']
val_acc = history['val_accuracy']
#acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']

loss = history['loss']
val_loss = history['val_loss']
#loss = history.history['loss']
#val_loss = history.history['val_loss']

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
plt.show()"""
