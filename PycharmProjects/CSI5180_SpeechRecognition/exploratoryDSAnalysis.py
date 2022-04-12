import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


def makePlots(dframe):
    emotion = ['fear', 'angry', 'disgust', 'neutral', 'sad', 'ps', 'happy']
    for em in emotion:
        path = np.array(dframe['speech'][dframe['label'] == em])[0]
        data, sampling_rate = librosa.load(path)
        wavePlot(data, sampling_rate, em)
        countPlot(dframe)
        audioSpectogram(data, sampling_rate, em)
        Audio(path)


def countPlot(dframe):
    sns.countplot(dframe['label'])


def wavePlot(data, sr, emotion):
    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr)
    plt.show()


def audioSpectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11, 4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()

