import pickle
import PIL
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import pathlib
import tensorflow
from keras import layers
from keras import Sequential

print("TensorFlow version:", tf.__version__)

batch_size = 64 #Default: 64
img_height = 180 #Default: 180
img_width = 180 #Default: 180
epochs = 2 #Default:5
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
  validation_split=0.8, #Default: 0.2
  subset="validation",
  seed=123,
  class_names=["apple", "banana", "mango", "orange"],
  image_size=(img_height, img_width),
  batch_size=batch_size)
class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

print(class_names)

normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))
num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

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
print("Saving")
with open('fruitTest.npy', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
#model.save("fruitModel.h5")
print("Saved")