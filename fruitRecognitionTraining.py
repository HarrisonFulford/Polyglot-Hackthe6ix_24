import pickle
import PIL
import cv2 as cv
import keras
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import pathlib
import tensorflow
from keras import layers
from keras import Sequential
#import handTracking
#from handTracking import transferAxisCordInfo

print("TensorFlow version:", tf.__version__)

#bottomAxis, topAxis, leftAxis, rightAxis = transferAxisCordInfo()
#print(topAxis, leftAxis, bottomAxis, rightAxis)

batch_size = 64 #Default: 64
img_height = 128 #Default: 180
img_width = 128 #Default: 180
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

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
  for i in range(9):
    augmented_images = data_augmentation(images)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_images[0].numpy().astype("uint8"))
    plt.axis("off")

model.summary()

#Scans each img
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

vid = cv.VideoCapture(0) 
frameno = 0
framesPerPhoto = 5 #How often a photo will be taken (Per frame)
photoType = ".jpg"

def checkImg():
  cam_path = "screenshots/currentFrame" + photoType
  img = tf.keras.utils.load_img(
      cam_path, target_size=(img_height, img_width)
  )
  img_array = tf.keras.utils.img_to_array(img)
  img_array = tf.expand_dims(img_array, 0) # Create a batch

  predictions = model.predict(img_array)
  score = tf.nn.softmax(predictions[0])

  print(
      "This image most likely belongs to {} with a {:.2f} percent confidence."
      .format(class_names[np.argmax(score)], 100 * np.max(score))
  )
  return "a"

#DOne in handTracking.py
"""
while(True):
  # Display the resulting frame 
  if (frameno%framesPerPhoto == 0): 
    ret, frame = vid.read() 
    cv.imshow('frame', frame) 
    name = "screenshots/currentFrame" + photoType
    print ('New frame captured')
    cv.imwrite(name, frame)
    frameno = 0
    fruit = checkImg();
  frameno += 1
  # the 'q' button is set as the 
  # quitting button you may use any 
  # desired button of your choice 
  if cv.waitKey(1) & 0xFF == ord('q'): 
    break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv.destroyAllWindows() 
"""
print("Saving")
#with open('fruitTest.npy', 'wb') as file_pi:
#    pickle.dump(history.history, file_pi)
#model.save("fruitModel.h5")
print("Saved")