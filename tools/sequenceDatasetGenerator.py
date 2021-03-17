# coding:utf-8
# 自己造轮子，实现深度学习批量数据的读取
import os
import glob
import numpy as np
import pickle
import tensorflow as tf
import cv2


class datasetGenerator:
    def __init__(self, batch_size, path):
        self.batch_size = batch_size
        self.path = path
        # self.validation_data_path = validation_data_path

    def get_data(self):
        files = []
        indices = []
        for i, dirs in enumerate(os.listdir(self.path)):
            print(i, dirs)
            # for filenames in os.listdir(os.path.join(self.path,dirs)):
            filenames_in_each_dir = glob.glob(os.path.join(self.path, dirs, '*.pkl'))
            flen = len(filenames_in_each_dir)
            files.extend(filenames_in_each_dir)  # 建立文件名列表files和对应的label索引indices
            indices.extend([i] * flen)
        # print(len(indices))
        # print(indices)
        # print(files[100000],indices[100000])
        return files, indices

    # def readTotalFileNum(self):
    #     count = 0
    #     # vcount = 0
    #
    #     num_in_each_class = 0
    #     array = []
    #     # varray = []
    #     for root, dirs, files in os.walk(self.path):
    #         for each in files:
    #             num_in_each_class += 1
    #             count += 1
    #         array.append(num_in_each_class)
    #         num_in_each_class = 0
    #     print('count:', count)
    #     array = array[1:]
    #     print(array)
    #
    #     return count, array
    #
    # def createLabels(self, sample_distribution):
    #     tlist = []
    #     for i in range(len(sample_distribution)):
    #         tlist += ([i] * sample_distribution[i])
    #     labels = np.array(tlist)
    #     return labels
    #
    #
    # def run(self):
    #     fileInfo = self.readTotalFileNum()
    #     ns = fileInfo[0]  # number of samples
    #     sd = fileInfo[1]  # sample distribution
    #
    #     # self.save_bottleneck_features(nts, nvs)  # 程序主体
    #     result_tunple = self.createLabels(sd)
    #     print(result_tunple)

    def binarize(self, x, sz=256):  # one-hot编码，维度256
        return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))

    def dataset(self):
        """
            写一个读取序列的生成器
            batch_size:批量大小
        """
        # 1. 读取所有序列名字
        data_list, data_indices = self.get_data()
        while True:
            dataList = []
            dataIndicesList = []
            for i in range(len(data_indices)):
                try:
                    data_name = data_list[i]
                    with open(data_name, 'rb') as f:  # 读取序列
                        raw_data = pickle.load(f)
                    convert_list = [eachbyte for eachbyte in raw_data]
                    data_nparray = np.array(convert_list)
                    # with tf.Session() as sess:
                    #     print(sess.run(self.binarize(data_nparray)))
                    one_hot_encoding_data = self.binarize(data_nparray)
                    dataList.append(one_hot_encoding_data)
                    dataIndicesList.append(data_indices[i])

                    if len(dataList) == self.batch_size:
                        yield dataList, dataIndicesList  # 采用函数生成器，生成一个可迭代对象
                        dataList = []
                        dataIndicesList = []

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    continue  # 所有序列已经读完一遍，跳出for循环,再进行第二次读取

    # def _parse_sourcedata(self, filename, label):
    #     # dataset.map对应的解析方法，它对每一个传进来的元素，即filenames[i],labels[i]进行操作
    #     # 实际上filenames[i]== filename,labels[i] == label
    #     data = tf.io.read_file(filename)
    #     data=tf.cast(data,dtype=tf.string)
    #     # data=pickle.loads(rawdata)
    #     print(data)
    #     data = [byte for byte in data]
    #     data = self.binarize(data)
    #     return data, label
    #
    def createdataset(self):  # tensorflow.data提供了创建数据集的功能
        flist = os.listdir(r'K:\数据库\ISCX-IDS-2012\2_extractFlowFeature\BFSSH')
        filenames = tf.constant(flist)
        lables = tf.constant([3] * len(flist))
        dataset = tf.data.Dataset.from_tensor_slices((filenames, lables))
        dataset = dataset.map(self._parse_sourcedata)
        return dataset


if __name__ == '__main__':
    # datasetGenerator(batch_size=32, path=r'K:\数据库\ISCX-IDS-2012\3_sampleDataset\train',
    # ).dataset()
    # datasetGenerator._parse_sourcedata('../preprocess/testbed-11jun.pcap.TCP_111-89-134-93_80_192-168-3-115_4901.pcap_10.pkl',)
    with open('../preprocess/testbed-11jun.pcap.TCP_111-89-134-93_80_192-168-3-115_4901.pcap_10.pkl', 'rb') as f:
        print(pickle.load(f))
