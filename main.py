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

##loading the data set
mnist = tf.keras.datasets.mnist

#x is what is written and y is the label
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#now we normalize the data (making it into values between 0-1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
#
##creating the neural network
#model = tf.keras.models.Sequential()
#
##this is how we add layers in the neural network
#model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
#model.add(tf.keras.layers.Dense(256, activation='relu'))  #rectify linear unit
#model.add(tf.keras.layers.Dense(128, activation='relu'))  #rectify linear unit
#model.add(tf.keras.layers.Dense(64, activation='relu'))  #rectify linear unit
#model.add(tf.keras.layers.Dense(10, activation='softmax'))  #output layer
#
#
#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
#
##now to train the model
#model.fit(x_train, y_train, epochs=3)
#
#model.save('handwritten_v2.keras')

#commenting all that out because model already ran
model = tf.keras.models.load_model('handwritten_v2.keras')

loss, accuracy = model.evaluate(x_test, y_test)

print(loss)
print(accuracy)


#image_num = 1
#
#while os.path.isfile(f"digits/digit{image_num}.png"):
#    try:
#        img_path = f"digits/digit{image_num}.png"
#        img = cv2.imread(img_path)
#
#        if img is None:
#            raise ValueError(f"Could not read image at {img_path}")
#
#        img = img[:, :, 0]  # grayscale
#        img = np.invert(img)
#        img_resized = cv2.resize(img, (28, 28))  # just in case it's not #28x28
#        img_array = np.array([img_resized])
#
#        prediction = model.predict(img_array, verbose=0)
#        predicted_digit = np.argmax(prediction)
#        print(f"{img_path} → Number is probably: {predicted_digit}")
#
#        # show the image
#        plt.imshow(img_resized, cmap=plt.cm.binary)
#        plt.title(f"Prediction: {predicted_digit}")
#        plt.axis('off')
#        plt.show()  # This blocks until the window is closed
#
#    except Exception as e:
#        print(f"[ERROR] digit{image_num}.png: {e}")
#
#    finally:
#        image_num += 1
#
#