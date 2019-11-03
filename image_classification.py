'''Learning to classify images using fashion mnist (Recognizing elements of clothing).

While preping the data we need to change to remove the saturation making each pixel either 0 or a 1.
'''
import PIL.ImageOps
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from skimage.io import imread
from tensorflow import keras

def show_img(img):
    '''Shows a picture using matplotlib'''
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.grid(False)
    plt.show()

def img_to_array(title):
    pic = Image.open(title).convert('L')
    pic_resized = pic.resize((28, 28), Image.NEAREST)
    pic_reversed = PIL.ImageOps.invert(pic_resized)
    array_from_pic = np.array(pic_reversed)
    array_from_pic = array_from_pic / 255.0
    return array_from_pic.reshape(28, 28)

FASHION_MNIST = keras.datasets.fashion_mnist

# splitting the data to training and testing
(train_images, train_labels), (test_images, test_labels) = FASHION_MNIST.load_data()

# decoding of lables (each label is a number 0-9)
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

# we can see that we have 60k (10k in test data) pictures each 28 by 28 pic
print(train_images.shape)

# viewing an img
# show_img(train_images[0])

# changing the images pixels to either 0 or 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# constructing the model
# the first layer is a flattened (changed to 1-dmientional) input
# the second is the hidden layer of 128 nodes
# the last is a layer of 10 possible outputs
# the output is an array of propabilities
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# training the model
model.fit(train_images, train_labels, epochs=6)

# testing the model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("Test accuracy:", test_acc)

# the cool part - making a prediction
image_uploaded = img_to_array('indeks.jpg')
image_uploaded = np.array(image_uploaded)
image_uploaded = image_uploaded[np.newaxis, :, :]
predictions = model.predict(image_uploaded)

print("I think its", CLASS_NAMES[np.argmax(predictions[0])])
show_img(image_uploaded[0])
# show_img(test_images[0])
# print(image_uploaded.shape)
# print(train_images[0].shape)
