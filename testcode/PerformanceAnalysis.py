from Classification import Classification
import os
from keras import optimizers

os.chdir('../')  # 将工作目录改为根目录，方便所有代码路径统一

run_times = 1  # 运行次数
optimizers.adadelta()


def runClassification(epochs, modelname, classnum, mode, train_data_dir, validation_data_dir):
    '''

    :param mode: mode 有picture和data和result三种模式,
    picture模式将训练结果输出成折线图，
    result模式将最后一个epoch的结果输出成日志文件
    data模式记录每一个epoch的结果并输出成日志文件

    :modelname: 神经网络的模型名
    :classnum :待分类数
    :return:
    '''
    for i in range(run_times):
        Classification(epochs=epochs, modelname=modelname, classnum=classnum, train_data_dir=train_data_dir,
                       validation_data_dir=validation_data_dir, mode=mode).run()


if __name__ == '__main__':
    runClassification(10, 'LSTM', 5, 'data', r'K:\数据库\ISCX-IDS-2012\3_sampleDataset\train',
                      r'K:\数据库\ISCX-IDS-2012\3_sampleDataset\validation')  # mode可选 ：data,picture,result
