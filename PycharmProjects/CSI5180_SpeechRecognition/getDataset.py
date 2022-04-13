import os
import pandas as pd


def formatDataset():
    paths = []
    labels = []

    for dirname, _, filenames in os.walk('C:\\Datasets\\TESS Toronto emotional speech set data'):
        for fn in filenames:
            paths.append(os.path.join(dirname, fn))
            label = fn.split('_')[-1]
            label = label.split('.')[0]
            labels.append(label.lower())
    print(len(paths))
    print(labels[:5])
    return "Dataset is loaded", paths, labels


def createDF(p, l):
    df = pd.DataFrame()
    df['speech'] = p
    df['label'] = l
    return df


def head(dataframe):
    return dataframe.head()


def valueCounts(dataframe):
    return dataframe['label'].value_counts()

# Debug:
# status, paths_, labels_ = formatDataset()
# dframe = createDF(paths_, labels_)
# print(head(dframe))
# print(valueCounts(dframe))
