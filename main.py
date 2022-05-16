import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

print("Importing a bunch of stuff here")

# get to files


from pathlib import Path
data_dir = Path('data')
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

#hotdog = list(data_dir.glob('hotdog/*'))
#image = mpimg.imread(str(hotdog[0]))
#plt.imshow(image)
#plt.show()

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


class_names = train_ds.class_names
print(class_names)
print(tf.version.VERSION)