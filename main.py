import os
import cv2  #computer vision is needed to load and process images
import numpy as np  #needed to work with arrays
import matplotlib.pyplot as plt  #purely for visualization purposes
import tensorflow as tf  #used for the ML part
"""
- when we usually train models, we get all the labeled data (we already now #what it is) that we have
- then it is split into training data and testing data
  - train using one and test it using the other
  - it doesn't see the testing data until tested
"""

# loading the MNIST data set
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

## normalize data (values between 0-1)
#x_train = tf.keras.utils.normalize(x_train, axis=1)
#x_test = tf.keras.utils.normalize(x_test, axis=1)
#
## creating the neural network
#model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
#model.add(tf.keras.layers.Dense(256, activation='relu'))
#model.add(tf.keras.layers.Dense(128, activation='relu'))
#model.add(tf.keras.layers.Dense(64, activation='relu'))
#model.add(tf.keras.layers.Dense(10, activation='softmax'))
#
#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
## train the model
#model.fit(x_train, y_train, epochs=3)
#
## save the original model
#model.save('models/handwritten_v2.keras')


#commenting all that out because model already ran
model = tf.keras.models.load_model('handwritten_v2.keras')

loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)