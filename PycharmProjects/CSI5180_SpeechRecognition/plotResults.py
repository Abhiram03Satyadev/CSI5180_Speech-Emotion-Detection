import matplotlib.pyplot as plt
import warnings
import numpy as np
from tensorflow import string

import getDataset
from exploratoryDSAnalysis import makePlots
from featureExtraction import x_features, oneHotEncode
from modelLSTM import buildLSTMModel, trainModel, buildBiLSTMModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import seaborn as sns

warnings.filterwarnings('ignore')

status, paths_, labels_ = getDataset.formatDataset()
df = getDataset.createDF(paths_, labels_)
# makePlots(df)
X = x_features(df)

Y = oneHotEncode(df)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

modelBiLSTM = buildBiLSTMModel()
modelLSTM = buildLSTMModel()

BiLSTM = trainModel(X_train, Y_train, modelBiLSTM)
LSTM = trainModel(X_test, Y_test, modelLSTM)

epochs = list(range(50))

acc_lstm = LSTM.history['accuracy']
valAcc_lstm = LSTM.history['val_accuracy']
acc_bilstm = BiLSTM.history['accuracy']
valAcc_bilstm = BiLSTM.history['val_accuracy']

fig1 = plt.figure()
plt.plot(epochs, acc_lstm, label='Train Accuracy')
plt.plot(epochs, valAcc_lstm, label='Val Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
fig1.savefig('LSTM_Accuracy.png', dpi=fig1.dpi)

fig2 = plt.figure()
plt.plot(epochs, acc_bilstm, label='Train Accuracy')
plt.plot(epochs, valAcc_bilstm, label='Val Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
fig2.savefig('BiLSTM_Accuracy.png', dpi=fig2.dpi)

loss_lstm = LSTM.history['loss']
val_loss_LSTM = LSTM.history['val_loss']
loss_bilstm = BiLSTM.history['loss']
val_loss_bilstm = BiLSTM.history['val_loss']

fig3 = plt.figure()
plt.plot(epochs, loss_lstm, label='train loss_lstm')
plt.plot(epochs, val_loss_LSTM, label='val loss_lstm')
plt.xlabel('epochs')
plt.ylabel('loss_lstm')
plt.legend()
fig3.savefig('LSTM_Loss.png', dpi=fig3.dpi)

fig4 = plt.figure()
plt.plot(epochs, loss_lstm, label='train loss_BiLSTM')
plt.plot(epochs, val_loss_LSTM, label='val loss_BiLSTM')
plt.xlabel('epochs')
plt.ylabel('loss_bilstm')
plt.legend()
fig4.savefig('BiLSTM_Loss.png', dpi=fig4.dpi)

# Preprocess for calculating the scores
Y_pred_LSTM = modelLSTM.predict(X_test)
Y_pred_LSTM = np.argmax(Y_pred_LSTM, axis=1)
Y_pred_BiLSTM = modelBiLSTM.predict(X_test)
Y_pred_BiLSTM = np.argmax(Y_pred_BiLSTM, axis=1)
Y_test = np.argmax(Y_test, axis=1)
cf_matrix_lstm = confusion_matrix(Y_test, Y_pred_LSTM)
cf_matrix_Bilstm = confusion_matrix(Y_test, Y_pred_BiLSTM)
print(cf_matrix_lstm)
print(cf_matrix_Bilstm)

# Confusion matrix of LSTM
fig5 = plt.figure()
ax = sns.heatmap(cf_matrix_lstm, annot=True, cmap='Blues')
ax.set_title('Seaborn Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['Ang', 'Dis', 'Fea', 'Hap', 'Neu', 'Pes', 'Sad'])
ax.yaxis.set_ticklabels(['Ang', 'Dis', 'Fea', 'Hap', 'Neu', 'Pes', 'Sad'])
fig5.savefig('ConfusionMatrix_LSTM', dpi=fig5.dpi)

# Confusion matrix from BiLSTM
fig6 = plt.figure()
ax = sns.heatmap(cf_matrix_Bilstm, annot=True, cmap='Blues')
ax.set_title('Seaborn Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['Ang', 'Dis', 'Fea', 'Hap', 'Neu', 'Pes', 'Sad'])
ax.yaxis.set_ticklabels(['Ang', 'Dis', 'Fea', 'Hap', 'Neu', 'Pes', 'Sad'])
fig6.savefig('ConfusionMatrix_BiLSTM', dpi=fig6.dpi)

# Get F1 score

f1_score_LSTM = f1_score(Y_test, Y_pred_LSTM, average='macro')
f1_score_BiLSTM = f1_score(Y_test, Y_pred_BiLSTM, average='macro')
accuracy_score_LSTM = accuracy_score(Y_test, Y_pred_LSTM)
accuracy_score_BiLSTM = accuracy_score(Y_test, Y_pred_BiLSTM)
file2write = open('Results.txt', 'w')
file2write.write(str(f1_score_LSTM))
file2write.write(str(f1_score_BiLSTM))
file2write.write(str(accuracy_score_LSTM))
file2write.write(str(accuracy_score_BiLSTM))
file2write.write(str(cf_matrix_lstm))
file2write.write(str(cf_matrix_Bilstm))

