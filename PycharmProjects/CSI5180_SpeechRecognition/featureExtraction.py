import numpy as np
import librosa
import librosa.display
from sklearn.preprocessing import OneHotEncoder
import warnings

warnings.filterwarnings('ignore')


def oneHotEncode(dframe):
    enc = OneHotEncoder()
    y = enc.fit_transform(dframe[['label']])
    y = y.toarray()
    return y


def x_features(dframe):
    X_mfcc = dframe['speech'].apply(lambda x: extractMFCC(x))
    X_ = [x for x in X_mfcc]
    X_ = np.array(X_)
    X_ = np.expand_dims(X_, -1)
    print(X_.shape)
    return X_


def extractMFCC(fn):
    y, sr = librosa.load(fn, duration=3, offset=0.5)
    calculated_mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return calculated_mfcc
