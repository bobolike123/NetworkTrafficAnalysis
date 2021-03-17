from keras.models import Input, Model
from keras.layers import Conv2D, Lambda, Activation, SeparableConv2D, MaxPooling2D, Conv1D, MaxPooling1D, \
    GlobalMaxPool1D, Dense
import tensorflow as tf


def simpleNet():
    img_input = Input(shape=(96, 96, 3))
    x = Conv2D(8, (7, 7), activation='relu', name='C1')(img_input)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='S1')(x)
    x = Conv2D(16, (7, 7), activation='relu', name='C3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='S4')(x)
    # x=Conv2D(80,(4,4),activation='relu',name='C5')(x)
    x = Conv2D(80, (7, 7), activation='relu', name='C5')(x)
    # x=layers.Dense(25,activation='softmax',name='prediction')(x)
    # This creates a model that includes
    # the Input layer and three Dense layers
    return Model(inputs=img_input, outputs=x)


def oneDNet():
    seq_input = Input(shape=(None,),dtype='int64')
    embedded=Lambda(binarize,output_shape=binarize_outshape)(seq_input)
    x = Conv1D(filters=32, kernel_size=3, padding='valid', activation='tanh', strides=2)(embedded)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=64, kernel_size=3, padding='valid', activation='tanh', strides=1)(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(128, activation='relu')(x)
    model = Model(inputs=seq_input, outputs=x)
    model.summary()

class WangNet:
    '''
        Wei Wang, Xuewen Zeng, Xiaozhou Ye, Yqiang Sheng and Ming Zhu. Malware
traffic classification using convolutional neural network for representation
learning[C]//The 31 st International Conference on Information Networking (ICOIN
2017). IEEE, 2017: 712-717.

    '''

    def __init__(self):
        self.model_name = 'WangNet'

    def model(self):

        return model, self.model_name

    def binarize(self, x, sz=256):
        return tf.cast(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1), dtype='float32')

    def binarize_outshape(self, in_shape):
        return in_shape[0], in_shape[1], 256


def binarize(x, sz=256):
    return tf.cast(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1),dtype='float32')
def binarize_outshape(in_shape):
    return in_shape[0], in_shape[1], 256
if __name__ == '__main__':
    oneDNet()
