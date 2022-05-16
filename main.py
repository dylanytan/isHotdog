import matplotlib.pyplot as plt
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
p = Path('.')
print([x for x in p.iterdir() if x.is_dir()])
