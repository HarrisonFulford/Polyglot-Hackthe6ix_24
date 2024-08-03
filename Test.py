import PIL
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import pathlib
import tensorflow
from keras import layers
from keras import Sequential

#model = create_model()
model = model.load("fruitModel.keras")