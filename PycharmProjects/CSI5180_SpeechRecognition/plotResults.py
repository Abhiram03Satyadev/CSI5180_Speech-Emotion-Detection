import matplotlib.pyplot as plt
import warnings
import getDataset
from exploratoryDSAnalysis import makePlots
from featureExtraction import x_features, oneHotEncode
from modelLSTM import buildLSTMModel, trainModel, buildBiLSTMModel

warnings.filterwarnings('ignore')

status, paths_, labels_ = getDataset.formatDataset()
df = getDataset.createDF(paths_, labels_)
#makePlots(df)
X = x_features(df)

Y = oneHotEncode(df)

modelBiLSTM = buildBiLSTMModel(X)
modelLSTM = buildLSTMModel()

BiLSTM = trainModel(X, Y, modelBiLSTM)
LSTM = trainModel(X, Y, modelLSTM)

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




