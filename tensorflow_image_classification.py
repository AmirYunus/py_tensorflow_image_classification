# import tensorflow libraries
import tensorflow as tf
from tensorflow.keras import keras
from tensorflow.keras.models import Sequential

# import supporting libraries
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import pathlib

# download and explore the dataset

# we will use a dataset of about 3,700 photos of flowers
# the dataset contains 5 sub-directories, one per class:
"""
flower_photo
    L> daisy
    L> dandelion
    L> roses
    L> sunflowers
    L> tulips
"""

dataset_url='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

"""

"""

# after downloading, we should now have a copy of the dataset available
# There are a total of 3,670 images
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

"""
3670
"""

# here are some roses
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]]))
PIL.Image.open(str(roses[1]))

# and some tulips
tulips - list(data_dir.glob('tulips/*'))
PIL.Image.open(str(tulips[0]))
PIL.Image.open(str(tulips[1]))


# load using keras.preprocessing

# let's load these images off disk using the `image_dataset_from_directory` utility
# this will take us from a directory of images on disk to a `tf.data.Dataset`
# define some parameters for the loader
batch_size = 32
image_height = 180
image_width = 180

# it's good practice to use a validation split when developing our model
# let's use 80% of the images for training and 20% for validation
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split = 0.2, subset='training', seed=123, image_size=(image_height, image_width), batch_size=batch_size)

"""

"""

validate_dataset = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split=0.2, subset='validation', seed=123, image_size=(image_height, image_width), batch_size=batch_size)

"""

"""

# we can find the class names in the `class_names` attribute on these datasets
# these correspond to the directory names in alphabetical order
class_names = train_dataset.class_names
print(class_names)

"""

"""


# visualise the data

# here are the first 9 images from the training dataset
plt.figure(figsize=(10, 10))

for _, each_label in train_dataset.take(1):
    for each_index in range (9):
        ax = plt.subplot(3, 3, each_index + 1)
        plt.imshow(images[each_index].numpy().astype('uint8'))
        plt.title(class_names[each_label[each_index]])
        plt.axis('off')

# we will train a model using these datasets by passing them to `model.fit`
# we can also manually iterate over the dataset and retrieve batches of images
for image_batch, labels_batch in train_dataset:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

"""

"""

# the `image_batch` is a tensor of the shape (32, 180, 180, 3)
# this is a batch of 32 images of shape 180 x 180 x 3
# the last dimension refers to the RGB colour channels
# the `label_batch` is a tensor of the shape (32,), these are corresponding labels to the 32 images
# we can call `/numpy()` on the `image_batch` and `labels_batch` tensors to convert them to a `numpy.ndarray`


# configure the dataset for performance

# let's make sure to use buffered prefetching so we can yield data from disk without having input/output becoming blocked
# these are two important methods we should use when loading data

# `Dataset.cache()` keeps the images in memory after they are loaded off disk during the first epoch
# this will ensure the dataset does not become a bottleneck while training our model
# if our dataset is too large to fit into memory, we can also use another method to create a performat on disk-cache

# `Dataset.prefetch()` overlaps data preprocessing and model execution while training
AUTOTUNE = tf.data.experimental.AUTOTUNE
shuffle = 1_000
train_dataset = train_dataset.cache().shuffle(shuffle).prefetch(buffer_size=AUTOTUNE)
validate_dataset = validate_dataset.cache().prefetch(buffer_size=AUTOTUNE)


# standardise the data

# the RGB channel values are in the [0, 255] range
# this is not ideal for a neural network
# in general we should seek to make our input values small
# we will standardise values to be in the [0, 1] range by using a Rescaling layer
normalisation_layer = layers.experimental.preprocessing.Rescaling(1./255)

# there are two ways to use this layer
# we can apply it to the dataset by calling map
normalised_dataset = train_dataset.map(lambda x, y: (normalisation_layer(x), y))
image_batch, labels_batch = next(iter(normalised_dataset))
first_image - image_batch[0]
print(np.min(first_image), np.max(first_image)) # notice that the pixel values are now in the [0, 1] range

"""

"""

# we can also include the layer inside our model definition which can simplify deployment
# let's use the second approach here


# create the model

# the model consists of three convolution blocks with a max pool layer in each of them
# there's a fully connected layer with 128 units on top of it that is activated by a `relu` activation function
number_of_classes = 5
number_of_channels = 3

model - Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(image_height, image_width, number_of_channels)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(number_of_classes)
])


# Compile the model

# choose the `optimizers.Adam` optimiser and `losses.SparseCategoricalCrossentropy` loss function
# to view training and validation accuracy for each training epoch, pass the `metrics` argument
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.summary() # view all the layers of the network

"""

"""


# train the model

number_of_epochs = 10
history = model.fit(train_dataset, validation_data=validate_dataset, epochs=number_of_epochs)

"""

"""


# visualise training results

# create plots of loss and accuracy on the training and validation sets
accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs_range = range(number_of_epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, accuracy, label='Training Accuracy')
plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, validation_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.shor()

# as we can see from the plots, training accuracy and validation accuracy are off by large margin and the model has achieved only around 60% accuracy on the validation set
# let's look at what went wrong and try to increase the overall performance of the model


# overfitting

# the training accuracy is increasing linearly over time, whereas validation accuracy stalls around 60% in the training process
# also, the difference in accuracy between training and validation accuracy is noticable (i.e. a sign of overfitting)
# when there are a small number of trainin examples, the model sometimes learns from noises or unwanted details from training examples to an extent that it negatively impacts the performance of the model on new examples
# this is known as overfitting
# it means that the model will have a difficult time generalising on a new dataset
# there are mulitple ways to address overfitting in the training process
# we will use data augmentation and add drop out to our model


# Data augmentation

# overfitting generally occurs when there are a small number of training examples
# data augmentation takes the approach of generating additional training data from your existing examples by augmenting them using random transformations that yield believable looking images
# this helps expose the model to more aspects of the data and generalise better
# we will implement data augmentation using experimental keras preprocessing layers
# these can be included inside your model like other layers
data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=(image_height, image_width, number_of_channels)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1)
])

# let's visualise what a few augmented examples look like by applying data augmentation to the same image several times
plt.figure(figsize=(10, 10))
for each_image, _ in train_dataset.take(1):
    for each_index in range (9):
        augmented_images = data_augmentation(each_image)
        ax = plt.subplot(3, 3, each_index + 1)
        plt.imshow(augmented_images[0].numpy().astype('uint8'))
        plt.axis('off')

# dropout

# another technique to reduce overfitting is to introduce Dropout to the network, a form of regularisation
# when we apply Dropout to a layer, it randomly drops out (by setting the activation to zero) a number of output units from the layer during the training process
# Dropout takes a fractional number as its input value, in the form such as 0.1, 0.2, 0.4, etc.
# this means dropping out 10%, 20% or 40% of the output units randomly from the applied layer
# let's create a new neural network using `layers.Dropout`, then train it using augmented images
model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255),
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


# compile and train the mode

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.summary()

"""

"""

number_of_epochs = 15
history = model.fit(train_dataset, validation_data=validate_dataset, epochs=number_of_epochs)

"""

"""

# visualise training results

# after applying data augmentation and Dropout, there is less overfitting than before
# training and validation accuracy are closer aligned
accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs_range = range(number_of_epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, accuracy, label='Training Accuracy')
plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, validation_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# predict on new data

# let's use our model to classify an image that wasn't included in the training or validation sets
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

image = keras.preprocessing.image.load_img(sunflower_path, target_size=(image_height, image_width))
image_array = keras.preprocessing.image.img_to_array(img)
image_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(image_array)
score = tf.nn.softmax(predictions[0])

print(f"This image most likely belongs to {class_names[np.argmax(score)]} with a {100 # np.max(score)} percent confidence.")

"""

"""
