'''A little bit different problem to be solved using ML: predicting cars fuel efficiency.
The main difference is that the output is a float.

An importat thing is changing the "Origin" value. Originally it represens cars
origin as a number between 1 and 3 where 1 -> USA 2 -> Europe and 3-> Japan.
This wont work to well for the model tho. We want to change those umbers to a vector where
[1, 0, 0] represents USA, [0, 1, 0] Europe and so on. Its called "One hot encoding"'''
from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

def convert(mpg):
    '''converts miles per gallon to L/100km'''
    liters_per_km = 100 / ((mpg * 1.609) / 4.546)
    return liters_per_km

def normalize(dset):
    '''Normalize the data (z score)'''
    return (dset - TRAIN_STATS["mean"]) / TRAIN_STATS["std"]

def build_model():
    model_built = keras.Sequential([
        keras.layers.Dense(64, activation="relu", input_shape=[len(X.keys())]),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

    model_built.compile(loss="mse", optimizer=optimizer, metrics=["mse", "mae"])

    return model_built

# downloading the data
DATASET_PATH = keras.utils.get_file("auto-mpg.data",
                                    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

COL_NAMES = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
             'Acceleration', 'Model Year', 'Origin']

RAW_DATASET = pd.read_csv(DATASET_PATH,
                          names=COL_NAMES,
                          na_values="?",
                          comment="\t",
                          sep=" ",
                          skipinitialspace=True)

dataset = RAW_DATASET.copy()

# view last 5 elements of dataset
# print(dataset.tail())

# checking num of missing values
# sum all N/A
# print(dataset.isna().sum())

# deleting instances that miss values
dataset = dataset.dropna()

# one hot encoding
origin = dataset.pop("Origin")

dataset["USA"] = (origin == 1) * 1.0
dataset["Europe"] = (origin == 2) * 1.0
dataset["Japan"] = (origin == 3) * 1.0
# now each instance has a 1 in either USA, Europe or Japan value


# splitting the data, changed to sklearn way
# general convention in ML programming - X is the data and y is what we are trying to predict
X = dataset.drop(columns=["MPG"])
y = dataset["MPG"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# get the statistics
TRAIN_STATS = X_train.describe()
# making it prittier
TRAIN_STATS = TRAIN_STATS.transpose()

# we want to normalize the data, so it is better for our model
X_train = normalize(X_train)
X_test = normalize(X_test)

# model = build_model()

EPOCHS = 1000
# history = model.fit(X_train, y_train, epochs=EPOCHS, validation_split=0.2, verbose=1)
# here we notice that before half of the training we get no better

# we are going to use early stopping, which will stop the training of the model if it doesnt improove
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
model = build_model()

history = model.fit(X_train, y_train, epochs=EPOCHS, validation_split=0.2, verbose=1, callbacks=[early_stop])
