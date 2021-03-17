import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.layers import Input
from keras.models import Model
from keras.callbacks import Callback
from log.logger import *
import os
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K


class Classification:
    def __init__(self, epochs, modelname, classnum, train_data_dir, validation_data_dir, mode='picture'):
        self.modelName = modelname
        self.top_model_weights_path = r'.\weights\model_weights.h5'  # 存放权重的路径
        self.train_data_dir = train_data_dir  # 训练集路径
        self.validation_data_dir = validation_data_dir  # 测试集路径
        self.classnum = classnum
        self.mode = mode
        self.img_width, self.img_height = 224, 224
        self.nb_train_samples = 0  # 声明全局变量
        self.nb_validation_samples = 0
        self.epochs = epochs
        self.batch_size = 32
        self.optimizer = 'Adamax'
        self.pltstyle = 'grayscale'

    def model_structure(self):
        flag = False
        if self.modelName == 'FLNet':
            from model.FLNet_model import FLNet
            model_structure = FLNet()
            self.img_width, self.img_height = 224, 224
            flag = True
        if self.modelName == 'VGG16':
            from keras import applications
            model_structure = applications.VGG16(include_top=False, weights=None)  # 还可以选择weight='imagenet'
            flag = True
        if self.modelName == 'Xception':
            from keras import applications
            model_structure = applications.Xception(include_top=False, weights=None)
            flag = True
        if self.modelName == 'simpleNet':
            from  model.CNN_model import simpleNet
            model_structure = simpleNet()
            self.img_width, self.img_height = 96, 96
            flag = True
        if self.modelName == 'LSTM':
            from model.LSTM_model import simpleLSTM
            model_structure = simpleLSTM()
            # self.img_width, self.img_height = 224, 224
            flag = True
        if flag == False: raise ValueError("未识别的模型名：", self.modelName)
        return model_structure

    def save_bottleneck_features(self, nb_train_samples, nb_validation_samples):
        # datagen = ImageDataGenerator(rescale=1. / 255)

        top_model = self.model_structure()
        top_model.summary()

        from tools.sequenceDatasetGenerator import datasetGenerator
        generator = datasetGenerator(batch_size=self.batch_size, path=r'K:\数据库\ISCX-IDS-2012\3_sampleDataset\train',
                                     ).dataset()
        generator = datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(self.img_width, self.img_height),
            batch_size=self.batch_size,
            class_mode=None,
            shuffle=False)
        bottleneck_features_train = top_model.predict_generator(
            generator, nb_train_samples // self.batch_size)
        np.save(r'.\evaluation\{}_bottleneck_features_train.evaluation'.format(self.modelName), bottleneck_features_train)

        # generator = datagen.flow_from_directory(
        #     self.validation_data_dir,
        #     target_size=(self.img_width, self.img_height),
        #     batch_size=self.batch_size,
        #     class_mode=None,
        #     shuffle=False)
        generator = datasetGenerator(batch_size=self.batch_size,
                                     path=r'K:\数据库\ISCX-IDS-2012\3_sampleDataset\validation',
                                     ).dataset()
        bottleneck_features_validation = top_model.predict_generator(
            generator, nb_validation_samples // self.batch_size)
        np.save(r'.\evaluation\{}_bottleneck_features_validation.evaluation'.format(self.modelName), bottleneck_features_validation)

    def train_top_model(self, trainlabels_nparray, validationlabels_nparray):
        train_data = np.load(r'.\evaluation\{}_malimg(square)_bottleneck_features_train.evaluation'.format(self.modelName))
        train_labels = trainlabels_nparray

        validation_data = np.load(r'.\evaluation\{}_malimg(square)_bottleneck_features_validation.evaluation'.format(self.modelName))
        validation_labels = validationlabels_nparray

        train_labels = to_categorical(train_labels, num_classes=self.classnum)  # 这里填写要待分类数
        validation_labels = to_categorical(validation_labels, num_classes=self.classnum)

        '''
        ----------------------------------------开始--------------------------------------------
        
        "开始"到“结束”之间夹的是神经网络的顶层（top-structure）顶层是根据你的数据集可以灵活调整的，比如待分类数是100，则令self.classnum=100
        相应的如果待分类数多，可以增加Dense层以更全面整合卷积出来的特征
        
        下面只给了一层Dense-256层，可以根据需要调整Dense层节点的个数或者增减Dense层
        '''
        model = Sequential()
        model.add(Flatten(input_shape=train_data.shape[1:]))

        # model.add(Dense(1024,activation='relu'))   #新增的层用于缓冲
        # model.add(Dropout(0.5))
        # model.add(Dense(256, activation='relu'))

        model.add(Dropout(0.5))
        model.add(Dense(self.classnum, activation='softmax'))  # 激活函数的选择上，多分类用softmax 二分类用sigmoid

        '''
        ------------------------------------------------结束----------------------------------------
        '''

        model.compile(optimizer=self.optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])  # 多分类一般用categorical_crossentropy作为损失函数，详情参考keras官方手册

        if self.mode == 'data' or self.mode == 'result':
            history = saveHistory()  # 记录每个epoch的结果

            model.fit(train_data, train_labels,
                      epochs=self.epochs,
                      batch_size=self.batch_size,
                      validation_data=(validation_data, validation_labels),
                      callbacks=[history])

            model.save_weights(self.top_model_weights_path)
            # drawPicture(train_log)
            if self.mode == 'result':
                # 返回最后一个epoch的结果
                return (history.losses[-1], history.val_losses[-1], history.acc[-1], history.val_acc[-1])
            else:
                return (history.losses, history.val_losses, history.acc, history.val_acc)
        if self.mode == 'picture':
            train_log = model.fit(train_data, train_labels,
                                  epochs=self.epochs,
                                  batch_size=self.batch_size,
                                  validation_data=(validation_data, validation_labels))
            model.save_weights(self.top_model_weights_path)
            self.drawPicture(train_log)

    def drawPicture(self, train_log):
        # plot the training loss and accuracy
        plt.style.use(self.pltstyle)
        plt.figure()
        plt.plot(np.arange(0, self.epochs), train_log.history["val_acc"], label="val_acc")
        plt.plot(np.arange(0, self.epochs), train_log.history["acc"], label="train_acc")
        # plt.title("Training Accuracy on malware classifier")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.savefig(r".\img\{}_Accuracy_FLNet_{:d}e({}).jpg".format(self.optimizer, self.epochs,
                                                                    datetime.datetime.now().strftime('%H_%M_%S')))

        plt.style.use(self.pltstyle)
        plt.figure()
        plt.plot(np.arange(0, self.epochs), train_log.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, self.epochs), train_log.history["loss"], label="train_loss")
        # plt.title("Training Loss and Accuracy on sar classifier")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.savefig(r".\img\{}_Loss_FLNet_{:d}e({}).jpg".format(self.optimizer, self.epochs,
                                                                datetime.datetime.now().strftime('%H_%M_%S')))
        '''
        the parameter pltstyle can chose these value below:
        ['bmh',
         'classic',
         'dark_background',
         'fast',
         'fivethirtyeight',
         'ggplot',
         'grayscale',
         'seaborn-bright',
         'seaborn-colorblind',
         'seaborn-dark-palette',
         'seaborn-dark',
         'seaborn-darkgrid',
         'seaborn-deep',
         'seaborn-muted',
         'seaborn-notebook',
         'seaborn-paper',
         'seaborn-pastel',
         'seaborn-poster',
         'seaborn-talk',
         'seaborn-ticks',
         'seaborn-white',
         'seaborn-whitegrid',
         'seaborn',
         'Solarize_Light2',
         'tableau-colorblind10',
         '_classic_test']
            '''

    def readTotalFileNum(self):
        tcount = 0
        vcount = 0

        num_in_each_class = 0
        tarray = []
        varray = []
        for root, dirs, files in os.walk(self.train_data_dir):
            for each in files:
                num_in_each_class += 1
                tcount += 1
            tarray.append(num_in_each_class)
            num_in_each_class = 0
        print('t_count:', tcount)
        tarray = tarray[1:]
        print(tarray)

        for root, dirs, files in os.walk(self.validation_data_dir):
            for each in files:
                num_in_each_class += 1
                vcount += 1
            varray.append(num_in_each_class)
            num_in_each_class = 0
        print('v_count:', vcount)
        varray = varray[1:]
        print(varray)

        return (tcount, vcount, tarray, varray)

    def createTrainLabels(self, train_sample_distribution):
        tlist = []
        for i in range(len(train_sample_distribution)):
            tlist += ([i] * train_sample_distribution[i])
        train_labels = np.array(tlist)
        return train_labels

    def createValidationLabels(self, validation_sample_distribution):
        vlist = []
        for i in range(len(validation_sample_distribution)):
            vlist += ([i] * validation_sample_distribution[i])
        validation_labels = np.array(vlist)
        return validation_labels

    def checkDataNumber(self, trainNum, validationNum):
        if trainNum % self.batch_size != 0:
            raise ValueError('trainNum:{} can not exact div batch_size={},please delete {} sample(s) in manual'
                             .format(trainNum, self.batch_size,
                                     trainNum - trainNum // self.batch_size * self.batch_size))
        if validationNum % self.batch_size != 0:
            raise ValueError('validationNum:{} can not exact div batch_size={},please delete {} sample(s) in manual'
                             .format(validationNum, self.batch_size,
                                     validationNum - validationNum // self.batch_size * self.batch_size))

    def run(self):
        fileInfo = self.readTotalFileNum()
        nts = fileInfo[0]  # nb_train_samples
        nvs = fileInfo[1]  # nb_validation_samples
        tsd = fileInfo[2]  # train_sample_distribution
        vsd = fileInfo[3]  # validation_sample_distribution

        self.checkDataNumber(nts, nvs)
        if self.mode == 'result':
            logName = r'.\log\{}.log'.format(self.modelName)  # 日志文件名
            oldcount = readCount(logName)
            starttime = datetime.datetime.now()  # 简易计时器

            self.save_bottleneck_features(nts, nvs)  # 程序主体
            result_tunple = self.train_top_model(self.createTrainLabels(tsd), self.createValidationLabels(vsd))

            endtime = datetime.datetime.now()  # 简易计时器
            time_cost = (endtime - starttime).seconds
            print('程序运行时间为：', time_cost)
            print('loss:{0[0]}\nval_loss:{0[1]}\nacc:{0[2]}\nval_acc:{0[3]}'.format(result_tunple))
            list = []
            list.append(time_cost)
            list.append(oldcount)
            tp = tuple(list) + result_tunple  # tp = (time_cost,oldcount,losses,val_losses,acc,val_acc)
            writeCount(tp, logName)

        if self.mode == 'data':
            logName = r'.\log\onlydata_{}.log'.format(self.modelName)  # 日志文件名
            self.save_bottleneck_features(nts, nvs)  # 程序主体
            result_tunple2 = self.train_top_model(self.createTrainLabels(tsd), self.createValidationLabels(vsd))
            writeData(result_tunple2, logName)

        if self.mode == 'picture':
            self.save_bottleneck_features(nts, nvs)  # 程序主体
            self.train_top_model(self.createTrainLabels(tsd), self.createValidationLabels(vsd))

        K.clear_session()  # 释放内存，初始化状态
        tf.reset_default_graph()


class saveHistory(Callback):
    def on_train_begin(self, logs=None):
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))


if __name__ == '__main__':
    pass  # 测试代码就不写了，直接在testcode中调用
