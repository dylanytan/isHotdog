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

# Get to the data 

from pathlib import Path
data_dir = Path('data')
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# Code to show images to ensure that I got to the files
#hotdog = list(data_dir.glob('hotdog/*'))
#image = mpimg.imread(str(hotdog[0]))
#plt.imshow(image)
#plt.show()


# Create the dataset to train model
# Dataset is split into 2 parts, one for training the model and one for checking
# The 2 need to be seperate to prevent overfitting, something that we will cover later

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


# Display some of the images within the dataset the ensure that nothing is messed up

class_names = train_ds.class_names

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

# Setup the model that is gonna be trained
# This is where a bunch of the problems occured

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#normalization_layer = layers.Rescaling(1./255)

num_classes = len(class_names)




# Notice that the accuracy for training data is going up really fast but the accuracy for test data is kinda stuck
# This is because the model is overfit: The model knows the training data too well and recognizes stuff that doesn't actually matter
# Loss is going down for training but up for testing because of this overfitting phenomenon
# Loss: penalty for bad prediction

# Overfitting in more prevelant with small training data (something that we have here since I can't be bothered to get bigger datasets and takes too much space and might blow my computer up)

# One solution is to augmenmt data to make a bigger dataset 
# (aka messing with the photos a bit so that they are still beliveable but different to the comupter)

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


# Another method is data dropout, but is pretty complex so don't fully understand and can't really explain
# Basically it is messing with the neuro-network behind this so that the model is better


# Now we make a new model that utilized these techniques

model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()




# Train the new model

epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)



# Now that we have a decent model, lets predict on new data
# Picture from google
# website url: https://www.epicurious.com/recipes/food/views/panchos-argentinos-argentine-style-hot-dogs

imageURl = "https://assets.epicurious.com/photos/5cfea9780fb62aae4d2bec74/1:1/w_2560%2Cc_limit/ArgentineHotDog_HERO_060619_2179.jpg"
imagePath = tf.keras.utils.get_file('Red_sunflower', origin=imageURl)

img = tf.keras.utils.load_img(
    imagePath, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
