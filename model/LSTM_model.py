from keras.models import Model
from keras.layers import Dense, Dropout, MaxPooling1D, Conv1D, GlobalMaxPool1D, Activation
from keras.layers import LSTM, Lambda, Bidirectional, concatenate, BatchNormalization, RepeatVector, Embedding
from keras.layers import TimeDistributed
from keras import Sequential, Input
import keras.preprocessing
from keras.utils import np_utils
import pickle
import numpy as np
import os


def HwangLSTM():  # 效果不行
    # model in paper:An LSTM-Based Deep Learning Approach for Classifying Malicious Traffic at the Packet Level
    model = Sequential()
    model.add(Embedding(input_dim=256, output_dim=64, input_shape=(54,)))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=32))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation='softmax'))
    # model.summary()
    return model,'HwangLSTM'

def GRUNet():
    pass
if __name__ == '__main__':
    HwangLSTM()
