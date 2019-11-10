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


# splitting the data
TRAIN_DATASET = dataset.sample()
