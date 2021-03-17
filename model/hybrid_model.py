from keras.models import Input, Model
from keras.layers import Conv2D, Lambda, Activation, SeparableConv2D, MaxPooling2D, Conv1D, MaxPooling1D, \
    GlobalMaxPool1D, Dense, TimeDistributed, GRU, concatenate, Dropout, LSTM
import tensorflow as tf
import os
from keras.utils import plot_model


class BoBoNet:
    '''
    嵌套结构
    '''

    def __init__(self, model_name='BoBoNet'):
        self.model_name = model_name

    def model(self):  # 默认为ISCX2012数据集的model
        seq_input = Input(shape=(54,), dtype='int64')
        embedded = Lambda(self.binarize, output_shape=self.binarize_outshape)(seq_input)
        x = Conv1D(filters=32, kernel_size=3, padding='valid', activation='tanh', strides=1)(embedded)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=64, kernel_size=3, padding='valid', activation='tanh', strides=1)(x)
        x = MaxPooling1D(pool_size=2)(x)
        # x = GlobalMaxPool1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)  # 加了这个，防止过拟合 val_acc显著上升

        gru_layer = GRU(units=92, return_sequences=True, dropout=0.5, recurrent_dropout=0.5, implementation=1)(x)
        gru_layer = MaxPooling1D(pool_size=3, strides=1, padding='valid')(gru_layer)
        gru_layer2 = GRU(units=92, return_sequences=False, dropout=0.5, recurrent_dropout=0.5, implementation=1)(
            gru_layer)
        dense_layer = Dense(5, name='dense_layer', activation='softmax')(gru_layer2)
        # dense_layer = Dense(5, name='dense_layer')(gru_layer2)
        model = Model(inputs=seq_input, outputs=dense_layer)
        # model.summary()
        return model, self.model_name

    def model_USTC(self):
        seq_input = Input(shape=(54,), dtype='int64')
        embedded = Lambda(self.binarize, output_shape=self.binarize_outshape)(seq_input)
        x = Conv1D(filters=32, kernel_size=3, padding='valid', activation='tanh', strides=1)(embedded)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=64, kernel_size=3, padding='valid', activation='tanh', strides=1)(x)
        x = MaxPooling1D(pool_size=2)(x)
        # x = GlobalMaxPool1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)  # 加了这个，防止过拟合 val_acc显著上升

        gru_layer = GRU(units=92, return_sequences=True, dropout=0.5, recurrent_dropout=0.5, implementation=1)(x)
        gru_layer = MaxPooling1D(pool_size=3, strides=1, padding='valid')(gru_layer)
        gru_layer2 = GRU(units=92, return_sequences=False, dropout=0.5, recurrent_dropout=0.5, implementation=1)(
            gru_layer)
        dense_layer = Dense(20, name='dense_layer', activation='softmax')(gru_layer2)  # USTC有20个类别
        model = Model(inputs=seq_input, outputs=dense_layer)
        # model.summary()
        return model, self.model_name

    def model_CICIDS(self):
        seq_input = Input(shape=(54,), dtype='int64')
        embedded = Lambda(self.binarize, output_shape=self.binarize_outshape)(seq_input)
        x = Conv1D(filters=32, kernel_size=3, padding='valid', activation='tanh', strides=1)(embedded)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=64, kernel_size=3, padding='valid', activation='tanh', strides=1)(x)
        x = MaxPooling1D(pool_size=2)(x)
        # x = GlobalMaxPool1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)  # 加了这个，防止过拟合 val_acc显著上升

        gru_layer = GRU(units=92, return_sequences=True, dropout=0.5, recurrent_dropout=0.5, implementation=1)(x)
        gru_layer = MaxPooling1D(pool_size=3, strides=1, padding='valid')(gru_layer)
        gru_layer2 = GRU(units=92, return_sequences=False, dropout=0.5, recurrent_dropout=0.5, implementation=1)(
            gru_layer)
        dense_layer = Dense(6, name='dense_layer', activation='softmax')(gru_layer2)  # CICIDS有7个类别
        model = Model(inputs=seq_input, outputs=dense_layer)
        # model.summary()
        return model, self.model_name

    def model_LSTM(self): #对比GRU
        seq_input = Input(shape=(54,), dtype='int64')
        embedded = Lambda(self.binarize, output_shape=self.binarize_outshape)(seq_input)
        x = Conv1D(filters=32, kernel_size=3, padding='valid', activation='tanh', strides=1)(embedded)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=64, kernel_size=3, padding='valid', activation='tanh', strides=1)(x)
        x = MaxPooling1D(pool_size=2)(x)
        # x = GlobalMaxPool1D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)  # 加了这个，防止过拟合 val_acc显著上升

        gru_layer = LSTM(units=92, return_sequences=True, dropout=0.5, recurrent_dropout=0.5, implementation=1)(x)
        gru_layer = MaxPooling1D(pool_size=3, strides=1, padding='valid')(gru_layer)
        gru_layer2 = LSTM(units=92, return_sequences=False, dropout=0.5, recurrent_dropout=0.5, implementation=1)(
            gru_layer)
        dense_layer = Dense(5, name='dense_layer', activation='softmax')(gru_layer2)
        model = Model(inputs=seq_input, outputs=dense_layer)
        # model.summary()
        return model, self.model_name

    def binarize(self, x, sz=256):
        return tf.cast(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1), dtype='float32')

    def binarize_outshape(self, in_shape):
        return in_shape[0], in_shape[1], 256


class simpleBoBoNet:
    '''
    平行结构
    '''

    def __init__(self, conv1_filters=32, conv2_filters=64, gru1_units=128, gru2_units=64,
                 kernel_size=3, model_name='simpleBoBoNet'):
        # self.dropout_rnn = dropout_rnn
        # self.dropout_cnn = dropout_cnn
        self.model_name = model_name
        self.conv1_filters = conv1_filters
        self.conv2_filters = conv2_filters
        self.gru1_units = gru1_units
        self.gru2_units = gru2_units
        # self.dense_units = dense_units
        self.kernel_size = kernel_size

    def model(self):
        '''
        现在是v3版本的model
        :return:
        '''
        seq_input = Input(shape=(None,), dtype='int64')
        embedded = Lambda(self.binarize, output_shape=self.binarize_outshape)(seq_input)
        x = Conv1D(filters=self.conv1_filters, kernel_size=3, padding='valid', activation='tanh', strides=2)(embedded)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(filters=self.conv2_filters, kernel_size=3, padding='valid', activation='tanh', strides=1)(x)
        # x = MaxPooling1D(pool_size=2)(x)
        x = GlobalMaxPool1D()(x)  # 减少一个维度
        x = Dropout(0.2)(x)  # 加了这个，防止过拟合 val_acc显著上升
        encoder = Dense(8, activation='relu', name='encoder_dense_layer')(x)
        gru_layer = GRU(units=self.gru1_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
                        implementation=1)(
            embedded)
        gru_layer2 = GRU(units=self.gru2_units, return_sequences=False, dropout=0.2, recurrent_dropout=0.2,
                         implementation=1)(
            gru_layer)
        gru_feature = Dense(32, name='gru_dense_layer', activation='relu')(gru_layer2)
        feature_layer = concatenate([encoder, gru_feature], axis=-1)
        ouput_layer = Dense(5, activation='softmax')(feature_layer)
        model = Model(inputs=seq_input, outputs=ouput_layer)
        # model.summary()
        return model, self.model_name

    def binarize(self, x, sz=256):
        return tf.cast(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1), dtype='float32')

    def binarize_outshape(self, in_shape):
        return in_shape[0], in_shape[1], 256


if __name__ == '__main__':
    # simpleBoBoNet().model()
    model, _ = BoBoNet().model()
    model.summary()
    # os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
    # plot_model(model, to_file='Hierarchical_model.png')
