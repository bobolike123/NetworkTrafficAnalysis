from keras.models import Input,Model
from keras.layers import Conv2D,BatchNormalization,Activation,SeparableConv2D,MaxPooling2D
from keras import layers
from keras.utils import plot_model

import os
def FLNet():
    img_input = Input(shape=(224, 224, 3))
    # Block 1
    x = Conv2D(32, (3, 3), use_bias=False, padding='same', strides=(2,2),name='block1_conv1')(img_input)  # stride->(2,2)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), use_bias=False, padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 2
    x = SeparableConv2D(128, (3, 3), use_bias=False, padding='same', name='block2_sepconv1_bn')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), use_bias=False, padding='same', name='block2_sepconv2_bn')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), use_bias=False, padding='same', strides=(2,2),name='block3_conv1')(x)  # stride->(2,2)
    residual = BatchNormalization()(residual)

    # Block 3
    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), use_bias=False, padding='same', name='block3_sepconv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), use_bias=False, padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(512, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    # Block 4
    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(512, (3, 3), use_bias=False, padding='same', name='block4_sepconv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(512, (3, 3), use_bias=False, padding='same', name='block4_sepconv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(512, (3, 3), use_bias=False, padding='same', name='block4_sepconv3')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(512, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    # Block 5
    x = Activation('relu')(x)
    x = SeparableConv2D(512, (3, 3), use_bias=False, padding='same', name='block5_sepconv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(512, (3, 3), use_bias=False, padding='same', name='block5_sepconv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(512, (3, 3), use_bias=False, padding='same', name='block5_sepconv3')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = Activation('relu')(x)  #是否需要？
    x = layers.add([x, residual])

    # This creates a model that includes
    # the Input layer and three Dense layers
    return Model(inputs=img_input, outputs=x)
'''
# draw the picture of model
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
plot_model(top_model, to_file='VGG-Xception_top_model.png')
'''
