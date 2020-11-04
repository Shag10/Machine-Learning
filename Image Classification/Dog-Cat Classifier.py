import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

train_dir = r'C:\path\Datasets\train'                                 
valid_dir = r'C:\path\Datasets\validation'                            # Distributed Data(Images) in three classes(set) Training, Validation and Test sets.
test_dir = r'C:\path\Datasets\test'

# Data Augmentation
train_dgen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

valid_dgen = ImageDataGenerator(rescale = 1./255)

train_generator = train_dgen.flow_from_directory(
            train_dir,
            target_size = (150,150),
            batch_size = 20,
            class_mode = 'binary')

validation_generator = valid_dgen.flow_from_directory(
            valid_dir,
            target_size = (150,150),
            batch_size = 20,
            class_mode = 'binary')
            
# VGG16 Model Used for classification (Transfer learning) 
conv_base = VGG16(weights = 'imagenet', include_top = False, input_shape = (150,150,3))

# Defining Layers of model
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

model.compile(loss = "binary_crossentropy", optimizer = optimizers.RMSprop(lr = 1e-5), metrics = ['acc'])

#Training the model for 30 epochs and 50 validation steps
history = model.fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 6,
        validation_data = validation_generator,
        validation_steps = 50,
        callbacks = [callback_cb])

#Plotting the graph the model. You can check this graph on Graph.png in same directory.
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

model.save("Transfer_Classifier.h5")
keras.backend.clear_session()
del model
model = keras.models.load_model("Transfer_Classifier.h5")

# Scaling Test sets.
test_dgen = ImageDataGenerator(rescale = 1./255)
test_generator = test_dgen.flow_from_directory(
            test_dir,
            target_size = (150,150),
            batch_size = 20,
            class_mode = 'binary')

# This will return test accuracy of the model.
model.evaluate_generator(test_generator, steps = 50)

/* Check Dog-Cat Classifier.txt as well to know more about my model */
