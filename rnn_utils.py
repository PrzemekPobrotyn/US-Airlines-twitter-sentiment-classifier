import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
import emoji
import re


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, SimpleRNN, LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D

from keras.callbacks import EarlyStopping, ModelCheckpoint


list_of_arlines = ['virgin america', 'united', 'southwest', 'delta',
                   'us airways', 'american', 'virginamerica', 'usairways']

r = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def preprocess(tweets):

    # lowercase everything
    tweets = [tweet.lower() for tweet in tweets]

    # remove twitter mentions
    tweets = [re.sub(r'@\w+', '', tweet) for tweet in tweets]

    # substitute any url for a string 'url'
    tweets = [re.sub(r, 'url', tweet) for tweet in tweets]

    # remove explicit naming of any of the airlines from the dataset
    tweets = [[word for word in tweet.split() if word not in list_of_arlines]
              for tweet in tweets]
    # join back to strings
    tweets = [' '.join(tweet) for tweet in tweets]

    return tweets


def prepare_text_for_keras(texts, tokenizer, maxlen=35):
    '''
    texts:
        a list of strings
    '''
    tokenizer.fit_on_texts(texts)
    texts = tokenizer.texts_to_sequences(texts)
    texts = pad_sequences(texts, maxlen)

    return texts


def prepare_flags_for_keras(y):
    return np.array(pd.get_dummies(y))


def split_table(X, y, split=(0.8, 0.1, 0.1), seed=42):
    '''Splits the table into train, validation and test sets.
       Returns a tuple (X_train, X_val, X_test, y_train, y_val, yest)'''

    # split into training set and the rest
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - split[0]), random_state=seed, stratify=y)

    # split the remaining data into validation and test sets
    X_val, X_test, y_val, y_test =\
        train_test_split(X_temp, y_temp,
                         test_size=split[2]/(split[1]+split[2]),
                         random_state=seed,
                         stratify=y_temp)

    return X_train, X_val, X_test, y_train, y_val, y_test


def read_embedding(path):

    embeddings_index = {}
    f = open(path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    return embeddings_index


def create_embedding_matrix(tokenizer, embedding_dim, embeddings_index):

    word_index = tokenizer.word_index
    if tokenizer.num_words:  # if num words is set, get rid of words with too high index
        word_index = {key: word_index[key] for key in word_index.keys()
                      if word_index[key] < (tokenizer.num_words + 1)}
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def show_classification_report(model, x_test, y_test):

    y_pred = model.predict_classes(x_test)
    print('\n')
    print(classification_report(pd.DataFrame(y_test).idxmax(axis=1), y_pred))


# following function piece of code is a slight modification of a function
# of the same name from
# https://github.com/carpedm20/emoji/blob/master/emoji/core.py
# essentially, we want the funciton to return a string, not a compiled regex

def get_emoji_regexp():
    '''
    Returns a string representing a regular expression
    caputing all grapical emojis.
    '''

    emojis = sorted(emoji.EMOJI_UNICODE.values(), key=len,
                    reverse=True)
    emoji_regex = u'(' + u'|'.join(re.escape(u) for u in emojis) + u')'

    return emoji_regex


class GraphicsEmojisExtractor(BaseEstimator, TransformerMixin):
    '''Class for extracting graphics emojis'''

    def __init__(self, r=get_emoji_regexp()):
        '''
        Args:
            r (str): regex for emoji extraction
        '''

        self.r = r


    def fit(self, x, y=None):
        ''' Fit method required by sklearn'''
        return self

    def transform(self, text):
        '''
        Args:
            text: list of strings

        Returns:
            A list of transformed strings, ie a list of strings consisting of
            graphics emojis found in tweets, joined by spaces.
        '''

        # extract graphics emojis
        graphics_emojis = [re.findall(self.r, tweet) for tweet in text]

        # join into a format required by CountVectorizer
        graphics_emojis = [' '.join(tweet) for tweet in graphics_emojis]

        return graphics_emojis


def shift_emoji_indices(emoji_list, num_words):
    if emoji_list:
        emoji_list = [x + num_words for x in emoji_list]
    return emoji_list


def append_emojis(X_emojis_, X_):
    x_list = []

    for i in range(len(X_emojis_)):
        if X_emojis_[i]:
            x_list.append(list(X_[i]) + (X_emojis_[i]))
        else:
            x_list.append(list(X_[i]))

    X_ = pad_sequences(x_list, maxlen=75)

    return X_


def concat_weights(embedding_matrix_, emoji_embedding_matrix_):
    return np.concatenate([embedding_matrix_, emoji_embedding_matrix_])
