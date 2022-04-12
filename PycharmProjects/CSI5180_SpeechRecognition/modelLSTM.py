import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation, Embedding
import tensorflow as tf


def trainModel(X_, Y_, mod):
    train = mod.fit(X_, Y_, validation_split=0.2, epochs=50, batch_size=64)
    return train


def buildBiLSTMModel(X_):
    model = Sequential([
        Bidirectional(LSTM(256, return_sequences=False, input_shape=(40, 1))),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(7, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=['accuracy'])
    # print(model.summary())
    return model


def buildLSTMModel():
    model = Sequential([
        LSTM(256, return_sequences=False, input_shape=(40, 1)),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(7, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=['accuracy'])
    print(model.summary())
    return model
