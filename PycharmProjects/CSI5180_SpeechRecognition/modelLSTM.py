import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation, Embedding, ActivityRegularization
import tensorflow as tf


def trainModel(X_, Y_, mod):
    train = mod.fit(X_, Y_, validation_split=0.2, epochs=50, batch_size=64)
    return train


def buildBiLSTMModel():
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=False, input_shape=(40, 1))),
        Dropout(0.2),
        Dense(64, activation='leaky_relu'),
        Dense(7, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=['accuracy'])
    # print(model.summary())
    return model


def buildLSTMModel():
    model = Sequential([
        LSTM(128, return_sequences=False, input_shape=(40, 1)),
        Dropout(0.2),
        Dense(64, activation='leaky_relu'),
        Dense(7, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=['accuracy'])
    print(model.summary())
    return model
