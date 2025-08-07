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

# normalize data (values between 0-1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# creating the neural network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

## train the model
model.fit(x_train, y_train, epochs=3)

## save the original model
model.save('models/handwritten_v3.keras')

# ----------------------------------------------------------
# ✅ NEW BLOCK: Fine-tune on your handwritten samples
# ----------------------------------------------------------

# Update this to match the correct labels of your 12 samples
custom_labels = [5, 3, 7, 9, 8, 1, 7, 2, 4, 4, 6, 6]

custom_dir = "digits/"  # <-- Your folder with 28x28 handwritten PNGs
custom_images = []

for i in range(len(custom_labels)):
    img_path = os.path.join(custom_dir, f"digit{i + 1}.png")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"[WARNING] Could not read {img_path}")
        continue

    img = np.invert(img)
    img = img / 255.0
    custom_images.append(img)

# Convert to numpy arrays
x_custom = np.array(custom_images).reshape(-1, 28, 28)
y_custom = np.array(custom_labels)

# Fine-tune the model using your handwriting
print("[INFO] Fine-tuning model on your handwritten digits...")
model.fit(x_custom, y_custom, epochs=3)

# Save the updated model
model.save("models/handwritten_with_finetune_v1.keras")
print("[INFO] Fine-tuned model saved as 'handwritten_with_finetune.keras'")

# ----------------------------------------------------------
# Continue with predictions on test images from digits/
# ----------------------------------------------------------

image_num = 1

while os.path.isfile(f"digits/digit{image_num}.png"):
    try:
        img_path = f"digits/digit{image_num}.png"
        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(f"Could not read image at {img_path}")

        img = img[:, :, 0]  # grayscale
        img = np.invert(img)
        img_resized = cv2.resize(img, (28, 28))  # just in case it's not 28x28
        img_array = np.array([img_resized])

        prediction = model.predict(img_array, verbose=0)
        predicted_digit = np.argmax(prediction)
        print(f"{img_path} → Number is probably: {predicted_digit}")

        plt.imshow(img_resized, cmap=plt.cm.binary)
        plt.title(f"Prediction: {predicted_digit}")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"[ERROR] digit{image_num}.png: {e}")

    finally:
        image_num += 1


#commenting all that out because model already ran
#model = tf.keras.models.load_model('handwritten_v2.keras')
#
#loss, accuracy = model.evaluate(x_test, y_test)
#
#print(loss)
#print(accuracy)