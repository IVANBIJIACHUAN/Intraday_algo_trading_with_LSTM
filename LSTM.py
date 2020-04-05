import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import keras
import pickle
import os

hidden_dim = 30
n_epochs = 100
time_steps = 50
batch_size = 128
activation = "tanh"
loss = 'categorical_crossentropy'
stop_patience = 20

dpi = 200


def reshape_dataset(x, y):
    if x is not None:
        if len(x.shape) == 2:
            x = x.reshape(x.shape[0], x.shape[1], 1)
    if len(y.shape) == 1:
        y = y.reshape(y.shape[0], 1)
    return x, y


class LSTM_Model():
    def __init__(self):
        self.model = Sequential()
        return

    def build(self, time_steps=time_steps, data_dim=1, output_dim=3):
        # expected input batch shape: (batch_size, timesteps, data_dim)
        # the sample of index i in batch k is the follow-up for the sample i in batch k-1.
        self.model.add(
            LSTM(hidden_dim, activation=activation, return_sequences=True, input_shape=(time_steps, data_dim)))
        self.model.add(LSTM(hidden_dim, activation=activation, return_sequences=True))
        self.model.add(LSTM(hidden_dim, activation=activation))
        self.model.add(Dense(output_dim, activation='softmax'))

        opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
        return self.model

    def train_test(self, x, y, plot=False):

        size = len(x)
        if size != len(y):
            return None
        x = x[: batch_size * (size // batch_size)]
        y = y[: batch_size * (size // batch_size)]

        x, y = reshape_dataset(x, y)

        x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.1, shuffle=False)
        print('train', x_train.shape, y_train.shape)
        print('validation', x_validation.shape, y_validation.shape)

        early_stopping = EarlyStopping(monitor='val_loss', patience=stop_patience, mode="min", verbose=2,
                                       restore_best_weights=True)
        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs,
                                 validation_data=(x_validation, y_validation), callbacks=[early_stopping])

        self.y_pred = self.predict(x_validation)
        self.y_validation_true = y_validation

        if plot == True:
            self.train_plot = self.view_accuracy(self.predict(x_train).argmax(axis=1), y_train.argmax(axis=1), 'Train')
            self.validation_plot = self.view_accuracy(self.predict(x_validation).argmax(axis=1), y_validation.argmax(axis=1), 'Validation')
        return history

    def predict(self, x_validation):
        pred = self.model.predict(x_validation)
        return pred

    def view_accuracy(self, y_pred=None, y_true=None, plot_name='Test', num=100):
        if y_pred is None:
            y_pred = self.y_pred.argmax(axis=1)
            y_true = self.y_validation_true.argmax(axis=1)

        plt.style.use('seaborn')
        plt.figure(figsize=(20, 6), dpi=dpi)
        plt.grid(True)
        plt.plot(y_pred[:num], color='lightcoral')
        plt.plot(y_true[:num], color='cornflowerblue', linewidth=1)
        plt.title('{}_{}'.format(ticker, plot_name))
        plt.legend(['predict', 'true'])
#         if plot_name == 'Test':
#             plt.savefig('{}_{}_{}_{}.png'.format(ticker, plot_name, str(time_steps), str(batch_size)))
#         else:
#             plt.savefig('{}_{}_{}_{}.png'.format(ticker, plot_name, str(time_steps), str(batch_size)))