import tensorflow as tf  #used for the ML part
import os

"""
- when we usually train models, we get all the labeled data (we already know what it is) that we have
- then it is split into training data and testing data
  - train using one and test it using the other
  - it doesn't see the testing data until tested
"""

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# loading and normalizing local data
train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/train",
    image_size=(28,28),      # resize images to 28x28 if needed
    color_mode="grayscale",  # assuming dataset is grayscale
    batch_size=32
)

train_ds = train_ds.cache()               # Cache dataset in memory for faster access
train_ds = train_ds.shuffle(1000)         # Shuffle dataset to prevent learning order bias
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)  # Prefetch next batch while current is training for efficiency

test_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(28,28),
    color_mode="grayscale",
    batch_size=32
)

test_ds = test_ds.cache()                  # Cache dataset in memory for faster access
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)  # Prefetch next batch for efficiency

# creating the neural network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Rescaling(1./255, input_shape=(28,28,1)))  #rescaling the data from 0-255 to 0-1

model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu'))  #first convolutional layer to detect edges/strokes
model.add(tf.keras.layers.MaxPooling2D((2,2)))  #pooling layer to reduce spatial dimensions and extract important features

# Second convolution + pooling for deeper features
model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))

model.add(tf.keras.layers.Flatten())  #flattens 2-d feature map into 1-d vector for dense layers
# dense layers to learn non-linear combinations of features
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))

model.add(tf.keras.layers.Dense(14, activation='softmax'))  # output layer: 14 classes (10 digits + 4 operators)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(train_ds, validation_data=test_ds, epochs=3)

# save the original model
model.save('models/handwritten_v3.keras')

#commenting all that out because model already ran
#model = tf.keras.models.load_model('handwritten_v2.keras')
#
#loss, accuracy = model.evaluate(x_test, y_test)
#
#print(loss)
#print(accuracy)