# Images in our dataset are of different sizes, to feed them into our Generative Adversarial Neural 
# Network we are going to resize all our images to 128X128.

# select all the images from the folder and resize them to 128X128 and save them on cubism_data.npy file.
'''
# Importing required libraries
import os
import numpy as np
from PIL import Image

# Defining an image size and image channel
# We are going to resize all our images to 128X128 size and since our images are colored images
# We are setting our image channels to 3 (RGB)

IMAGE_SIZE = 128
IMAGE_CHANNELS = 3
IMAGE_DIR = 'dataset/'

# Defining image dir path. Change this if you have different directory
images_path = IMAGE_DIR 

training_data = []

# Iterating over the images inside the directory and resizing them using
# Pillow's resize method.
print('resizing...')

for filename in os.listdir(images_path):
    path = os.path.join(images_path, filename)
    image = Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
    training_data.append(np.asarray(image))

training_data = np.asarray(training_data)

training_data = np.reshape(
    training_data, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))

training_data = training_data / 127.5 - 1

print('saving file...')
np.save('cubism_data.npy', training_data)
'''
# Importing required libraries
import os
import numpy as np
from PIL import Image

# Defining an image size and image channel
# We are going to resize all our images to 128X128 size and since our images are colored images
# We are setting our image channels to 3 (RGB)

IMAGE_SIZE = 128
IMAGE_CHANNELS = 3
IMAGE_DIR = 'dataset/'

# Defining image dir path. Change this if you have different directory
images_path = IMAGE_DIR 

training_data = []
# training_data = np.array([])

# Iterating over the images inside the directory and resizing them using
# Pillow's resize method.
print('resizing...')

for filename in os.listdir(images_path):
    path = os.path.join(images_path, filename)
    image = Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)

    training_data.append(np.asarray(image))

print(training_data[0].shape) # (128, 128, 3)

training_data[3000] = np.reshape(
    training_data[3000], (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))

print(len(training_data))
training_data_range = range(0, len(training_data))
# this loop is broken. The for loop isnt very good at reshaping the data of varying sizes
# ValueError: cannot reshape array of size 65536 into shape (128,128,3)
for i in range(3100, len(training_data)):
    training_data[i] = np.reshape(
            training_data[i], (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))

for i in range(0, len(training_data)):
    training_data[i] = training_data[i] / 127.5 - 1

print('saving file...')
np.save('cubism_data.npy', training_data)