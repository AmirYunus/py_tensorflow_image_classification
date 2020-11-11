# import tensorflow libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
Downloading data from https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
228818944/228813984 [==============================] - 9s 0us/step
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
PIL.Image.open(str(roses[0]))
PIL.Image.open(str(roses[1]))

# and some tulips
tulips = list(data_dir.glob('tulips/*'))
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
Found 3670 files belonging to 5 classes.
Using 2936 files for training.
"""

validate_dataset = tf.keras.preprocessing.image_dataset_from_directory(data_dir, validation_split=0.2, subset='validation', seed=123, image_size=(image_height, image_width), batch_size=batch_size)

"""
Found 3670 files belonging to 5 classes.
Using 734 files for validation.
"""

# we can find the class names in the `class_names` attribute on these datasets
# these correspond to the directory names in alphabetical order
class_names = train_dataset.class_names
print(class_names)

"""
['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
"""


# visualise the data

# here are the first 9 images from the training dataset
plt.figure(figsize=(10, 10))

for each_image, each_label in train_dataset.take(1):
    for each_index in range (9):
        ax = plt.subplot(3, 3, each_index + 1)
        plt.imshow(each_image[each_index].numpy().astype('uint8'))
        plt.title(class_names[each_label[each_index]])
        plt.axis('off')

# we will train a model using these datasets by passing them to `model.fit`
# we can also manually iterate over the dataset and retrieve batches of images
for image_batch, labels_batch in train_dataset:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

"""
(32, 180, 180, 3)
(32,)
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
first_image = image_batch[0]
print(np.min(first_image), np.max(first_image)) # notice that the pixel values are now in the [0, 1] range

"""
0.0 0.9867099
"""

# we can also include the layer inside our model definition which can simplify deployment
# let's use the second approach here


# create the model

# the model consists of three convolution blocks with a max pool layer in each of them
# there's a fully connected layer with 128 units on top of it that is activated by a `relu` activation function
number_of_classes = 5
number_of_channels = 3

model = Sequential([
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
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
rescaling_1 (Rescaling)      (None, 180, 180, 3)       0
_________________________________________________________________
conv2d (Conv2D)              (None, 180, 180, 16)      448
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 90, 90, 16)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 90, 90, 32)        4640
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 45, 45, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 45, 45, 64)        18496
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 22, 22, 64)        0
_________________________________________________________________
flatten (Flatten)            (None, 30976)             0
_________________________________________________________________
dense (Dense)                (None, 128)               3965056
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 645
=================================================================
Total params: 3,989,285
Trainable params: 3,989,285
Non-trainable params: 0
_________________________________________________________________
"""


# train the model

number_of_epochs = 10
history = model.fit(train_dataset, validation_data=validate_dataset, epochs=number_of_epochs)

"""
Epoch 1/10
92/92 [==============================] - 93s 1s/step - loss: 1.3357 - accuracy: 0.4312 - val_loss: 1.0614 - val_accuracy: 0.6035
Epoch 2/10
92/92 [==============================] - 92s 1s/step - loss: 0.9433 - accuracy: 0.6383 - val_loss: 0.9346 - val_accuracy: 0.6444
Epoch 3/10
92/92 [==============================] - 94s 1s/step - loss: 0.7642 - accuracy: 0.7067 - val_loss: 0.9415 - val_accuracy: 0.6322
Epoch 4/10
92/92 [==============================] - 92s 1s/step - loss: 0.5774 - accuracy: 0.7871 - val_loss: 0.8652 - val_accuracy: 0.6703
Epoch 5/10
92/92 [==============================] - 92s 1s/step - loss: 0.3798 - accuracy: 0.8699 - val_loss: 0.8931 - val_accuracy: 0.6826
Epoch 6/10
92/92 [==============================] - 93s 1s/step - loss: 0.2268 - accuracy: 0.9309 - val_loss: 1.1293 - val_accuracy: 0.6308
Epoch 7/10
92/92 [==============================] - 95s 1s/step - loss: 0.1183 - accuracy: 0.9619 - val_loss: 1.2113 - val_accuracy: 0.6458
Epoch 8/10
92/92 [==============================] - 95s 1s/step - loss: 0.0471 - accuracy: 0.9901 - val_loss: 1.5234 - val_accuracy: 0.6785
Epoch 9/10
92/92 [==============================] - 97s 1s/step - loss: 0.0166 - accuracy: 0.9976 - val_loss: 1.6894 - val_accuracy: 0.6580
Epoch 10/10
92/92 [==============================] - 95s 1s/step - loss: 0.0436 - accuracy: 0.9898 - val_loss: 1.5704 - val_accuracy: 0.6608
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
plt.show()

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
  layers.Dense(number_of_classes)
])


# compile and train the mode

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.summary()

"""
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
sequential_1 (Sequential)    (None, 180, 180, 3)       0
_________________________________________________________________
rescaling_2 (Rescaling)      (None, 180, 180, 3)       0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 180, 180, 16)      448
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 90, 90, 16)        0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 90, 90, 32)        4640
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 45, 45, 32)        0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 45, 45, 64)        18496
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 22, 22, 64)        0
_________________________________________________________________
dropout (Dropout)            (None, 22, 22, 64)        0
_________________________________________________________________
flatten_1 (Flatten)          (None, 30976)             0
_________________________________________________________________
dense_2 (Dense)              (None, 128)               3965056
_________________________________________________________________
dense_3 (Dense)              (None, 5)                 645
=================================================================
Total params: 3,989,285
Trainable params: 3,989,285
Non-trainable params: 0
_________________________________________________________________
"""

number_of_epochs = 15
history = model.fit(train_dataset, validation_data=validate_dataset, epochs=number_of_epochs)

"""
Epoch 1/15
92/92 [==============================] - 134s 1s/step - loss: 1.3407 - accuracy: 0.4118 - val_loss: 1.2430 - val_accuracy: 0.5041
Epoch 2/15
92/92 [==============================] - 114s 1s/step - loss: 1.0853 - accuracy: 0.5525 - val_loss: 1.1637 - val_accuracy: 0.5259
Epoch 3/15
92/92 [==============================] - 114s 1s/step - loss: 0.9680 - accuracy: 0.6202 - val_loss: 0.9612 - val_accuracy: 0.6131
Epoch 4/15
92/92 [==============================] - 114s 1s/step - loss: 0.8997 - accuracy: 0.6519 - val_loss: 0.8869 - val_accuracy: 0.6417
Epoch 5/15
92/92 [==============================] - 114s 1s/step - loss: 0.8449 - accuracy: 0.6706 - val_loss: 0.8599 - val_accuracy: 0.6689
Epoch 6/15
92/92 [==============================] - 117s 1s/step - loss: 0.7915 - accuracy: 0.6979 - val_loss: 0.8944 - val_accuracy: 0.6431
Epoch 7/15
92/92 [==============================] - 119s 1s/step - loss: 0.7557 - accuracy: 0.7132 - val_loss: 0.7522 - val_accuracy: 0.7207
Epoch 8/15
92/92 [==============================] - 118s 1s/step - loss: 0.7037 - accuracy: 0.7371 - val_loss: 0.7654 - val_accuracy: 0.6826
Epoch 9/15
92/92 [==============================] - 122s 1s/step - loss: 0.6670 - accuracy: 0.7456 - val_loss: 0.7240 - val_accuracy: 0.7153
Epoch 10/15
92/92 [==============================] - 125s 1s/step - loss: 0.6387 - accuracy: 0.7568 - val_loss: 0.7097 - val_accuracy: 0.7221
Epoch 11/15
92/92 [==============================] - 117s 1s/step - loss: 0.6456 - accuracy: 0.7548 - val_loss: 0.7227 - val_accuracy: 0.7153
Epoch 12/15
92/92 [==============================] - 129s 1s/step - loss: 0.6007 - accuracy: 0.7738 - val_loss: 0.7181 - val_accuracy: 0.7316
Epoch 13/15
92/92 [==============================] - 121s 1s/step - loss: 0.5832 - accuracy: 0.7800 - val_loss: 0.7752 - val_accuracy: 0.7234
Epoch 14/15
92/92 [==============================] - 126s 1s/step - loss: 0.5581 - accuracy: 0.7960 - val_loss: 0.7645 - val_accuracy: 0.7112
Epoch 15/15
92/92 [==============================] - 119s 1s/step - loss: 0.5327 - accuracy: 0.7977 - val_loss: 0.6679 - val_accuracy: 0.7466
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
image_array = keras.preprocessing.image.img_to_array(image)
image_array = tf.expand_dims(image_array, 0) # Create a batch

predictions = model.predict(image_array)
score = tf.nn.softmax(predictions[0])

print(f"This image most likely belongs to {class_names[np.argmax(score)]} with a {100 * np.max(score)} percent confidence.")

"""
This image most likely belongs to sunflowers with a 93.02732944488525 percent confidence.
"""
