'''Learning TensorFlow - cool python ML enviroment.

IMDB is a database of movie reviews provided by tf, good for learning.

The data is a set of ints coresponding to words in dictionary and labels are
either 0 (positive review) or 1 (negative review).

As we are going to build a neuro net the first chellange to overcome is going to be making all
the entires same size (each movie review has different length).
'''
from __future__ import absolute_import, division, print_function

import numpy as np

from tensorflow import keras
import tensorflow as tf


IMDB = keras.datasets.imdb

# splitting data for training and testing
# num_words specifies that we want to use 10000 most frequently occurring words
(TRAIN_DATA, TRAIN_LABELS), (TEST_DATA, TEST_LABELS) = IMDB.load_data(num_words=10000)

# now thats a lot of data
print(f"Training entries {len(TRAIN_DATA)}. Labels: {len(TRAIN_LABELS)}.")

# whoops, different lengths
print(len(TRAIN_DATA[0]), len(TRAIN_DATA[1]))
