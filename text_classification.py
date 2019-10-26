'''Learning TensorFlow - cool python ML enviroment.

IMDB is a database of movie reviews provided by tf, good for learning.

The data is a set of ints coresponding to words in dictionary and labels are
either 0 (positive review) or 1 (negative review).

As we are going to build a neuro net the first chellange to overcome is going to be making all
the entires same size (each movie review has different length).

Also, the data set is a list of ints coresponding to words in a dictionary. We need to decode that.

The solution for different lengths isnt complcated, but rather strange.
We are going to create another dictionary where each possible of 10 000 words will be
followed by either 0 or 1 depending on whether it is present in a review or not.
So for the review that consists of words [5, 22] we are going to have a dictionary where
9 998 words will be 0 and 2 of them 1. Thats gonna be memory intensive AF. Too intensive tbh.

But there is another. Even simpler - limiting the num of words in a review.
In this case to 256 chars. And if its less than 256 we are going to pad it.
'''
from __future__ import absolute_import, division, print_function

import numpy as np

from tensorflow import keras
import tensorflow as tf

def decode_review(text):
    '''Decodes a review from list of ints to english'''
    # if we know the number of word given return the word of coresponding index
    # if we dont return a ?
    return ' '.join([REVERSE_WORD_INDEX.get(index, "?") for index in text])


IMDB = keras.datasets.imdb

# splitting data for training and testing
# num_words specifies that we want to use 10000 most frequently occurring words
(TRAIN_DATA, TRAIN_LABELS), (TEST_DATA, TEST_LABELS) = IMDB.load_data(num_words=10000)

# now thats a lot of data
print(f"Training entries {len(TRAIN_DATA)}. Labels: {len(TRAIN_LABELS)}.")

# whoops, different lengths
print(len(TRAIN_DATA[0]), len(TRAIN_DATA[1]))

WORD_INDEX = IMDB.get_word_index()

# adding new keywords to word index
WORD_INDEX = {k: (v+3) for k, v in WORD_INDEX.items()}
# adding padding
WORD_INDEX["<PAD>"] = 0
# adding indicator of start of the review
WORD_INDEX["<START>"] = 1
# adding unknown (a word that we don't know that smbdy used in a review) word keyword
WORD_INDEX["<UNK>"] = 2
# adding unused word keyword
WORD_INDEX["<UNUSED>"] = 3

# WORD_INDEX is a mapping of words into ints, we want it the other way
# (switching key val pair to val key)
REVERSE_WORD_INDEX = dict([(value, key) for (key, value) in WORD_INDEX.items()])

# testing the decoding form ints to eng
print(decode_review(TRAIN_DATA[0]))

# padding the reviews if they are too short
TRAIN_DATA = keras.preprocessing.sequence.pad_sequences(TRAIN_DATA,
                                                        value=WORD_INDEX["<PAD>"],
                                                        padding="POST",
                                                        maxlen=256)
TEST_DATA = keras.preprocessing.sequence.pad_sequences(TEST_DATA,
                                                       value=WORD_INDEX["<PAD>"],
                                                       padding="POST",
                                                       maxlen=256)
