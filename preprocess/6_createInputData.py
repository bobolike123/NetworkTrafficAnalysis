# 生成训练，验证，测试阶段的数据集，理论上训练时每个种类样本的比例都占总比例的1/N   (N为种类数)，验证，测试集亦如此
# 此代码就是为了达到上述目的而设计出来的。

import pickle
import os
# import glob  不能和scapy混用
import random
import yaml
from scapy.all import *

# ROOT_PATH = r'K:\数据库\ISCX-IDS-2012\2_clusteringPcap'
# ROOT_PATH = r'K:\数据库\USTC-TFC2016\1_preprocessed\USTC-TFC2016_malware\AllLayers_flow'
ROOT_PATH = r'K:\dataset\CIC-IDS-2017\3_Categorized'
DATA_IS_FROM = 'CICIDS2017'
# DATA_IS_FROM = 'USTC-TFC2016'
NUM_PER_CATEGORY = 20000
# NUM_PER_CATEGORY = 6000
# CLASS_LABEL = {'Normal': 0, 'BFSSH': 1, 'Infilt': 2, 'HttpDoS': 3, 'DDoS': 4}
# LABEL_CLASS = {0: 'Normal', 1: 'BFSSH', 2: 'Infilt', 3: 'HttpDoS', 4: 'DDoS'}
# CLASS_LABEL = {'Normal': 0, }
# LABEL_CLASS = {0: 'Normal'}
# 0-9为正常流量，10-19为恶意流量
# CLASS_LABEL = {'BitTorrent': 0, 'Facetime': 1, 'FTP': 2, 'Gmail': 3, 'MySQL': 4, 'Outlook': 5, 'Skype': 6, 'SMB-1': 7,
#                'SMB-2': 7, 'Weibo-1': 8, 'Weibo-2': 8, 'Weibo-3': 8, 'Weibo-4': 8, 'WorldOfWarcraft': 9, 'Cridex': 10,
#                'Geodo': 11, 'Htbot': 12, 'Miuref': 13, 'Neris': 14, 'Nsis-ay': 15, 'Shifu': 16, 'Tinba': 17,
#                'Virut': 18, 'Zeus': 19}
# LABEL_CLASS = {0: 'BitTorrent', 1: 'Facetime', 2: 'FTP', 3: 'Gmail', 4: 'MySQL', 5: 'Outlook', 6: 'Skype', 7: 'SMB',
#                8: 'Weibo', 9: 'WorldOfWarcraft', 10: 'Cridex', 11: 'Geodo', 12: 'Htbot', 13: 'Miuref', 14: 'Neris',
#                15: 'Nsis-ay', 16: 'Shifu', 17: 'Tinba', 18: 'Virut', 19: 'Zeus'}
# CICIDS2017
CLASS_LABEL = {'Normal': 0, 'BruteForce': 1,  'WebAttack': 2, 'Infiltration': 3, 'BotNet': 4,
               'PortScan': 5}
LABEL_CLASS = {0: 'Normal', 1: 'BruteForce',  2: 'WebAttack', 3: 'Infiltration', 4: 'BotNet',
               5: 'PortScan'}

CURRENT_DATASET_LINE_NUM = 0
DATA2SAVE = []  # 训练集和验证集要保存的内容
DATA2SAVE_test = []  # 测试集要保存的内容


def del_all_pkl():
    flist_to_del = glob(ROOT_PATH + '/*.pkl')
    for filepath in flist_to_del:
        os.remove(filepath)


def create_dataset(train_ratio, validation_ratio, test_ratio):
    '''
    主函数
    因为模型训练代码有把原始数据集分成train和validation了，因此这里我们只分成train_validation和test两个数据集
    :param train_ratio:
    :param validation_ratio:
    :param test_ratio:
    :return:
    '''
    # train_set_num = int(NUM_PER_CATEGORY * train_ratio / (train_ratio + validation_ratio + test_ratio))
    # validation_set_num = int(NUM_PER_CATEGORY * validation_ratio / (train_ratio + validation_ratio + test_ratio))
    global TRAIN_VALIDATION_SET_NUM, TEST_SET_NUM
    TRAIN_VALIDATION_SET_NUM = int(
        NUM_PER_CATEGORY * (train_ratio + validation_ratio) / (train_ratio + validation_ratio + test_ratio))
    TEST_SET_NUM = NUM_PER_CATEGORY - TRAIN_VALIDATION_SET_NUM
    # 上面有个问题是三个数据集总数加起来没有NUM_PER_CATEGORY 如果不能整除的话
    train_flist = []
    vali_flist = []
    test_flist = []
    global PACKET_NOT_ENOUGH
    del_all_pkl()
    # 如果不删除ROOT文件夹下的pkl文件，则os.listdir会把pkl文件也枚举出来，当然也有别的处理方法，比如不要这
    # 个函数把下面的os.listdir()换成os.walk()

    category_list = os.listdir(ROOT_PATH)
    for cate in category_list:
        if cate == 'DoS_DDoS':
            continue
        if cate != 'BruteForce':
            continue
        label = CLASS_LABEL[cate]
        print(cate)
        # print(glob(ROOT_PATH + f'/{cate}/*.pcap'))
        flist_in_cate = glob(ROOT_PATH + f'/{cate}/*.pcap')
        random.shuffle(flist_in_cate)  # 随机打乱下顺序，取样本更可靠
        flag = True
        PACKET_NOT_ENOUGH = True
        for file in flist_in_cate:
            if flag:
                flag = parse_pcap(file, label)
            else:
                break  # 正常来说如果每个cate的包数大于NUM_PER_CATEGORY，会触发这个break,此时data2save这个list为空
        _check_data2save(cate)
    # _create_train_dataset()
    # _create_validation_dataset()
    # 因为model 训练的代码能够自动把载入的pkl文件划分成数据集和验证集，因此我们只需要
    # _create_test_dataset()


def parse_pcap(pcap_file_path, label):
    '''
    解析pcap，如果解析的数量达到NUM_PER_CAEGORY,则中止解析，返回flag=False
    注意，这里的raw(pcapfile[index])没有截断，保留了数据完整，后续处理起来更灵活
    :param pcap_file_path: 
    :param label: 
    :return: False if 解析数量满足要求
    '''
    global PACKET_LEN, NUM_PER_CATEGORY, PACKET_NOT_ENOUGH

    pcapfile = rdpcap(pcap_file_path)
    packet_num = len(pcapfile)
    for index in range(packet_num):
        packet_feature = raw(pcapfile[index])
        content2save = [label, packet_feature]
        print(content2save)
        if PACKET_NOT_ENOUGH:
            PACKET_NOT_ENOUGH = save_packet_feature(content2save, label)
        else:  # Packet数够了
            # PACKET_NOT_ENOUGH=True
            return False
    return True


def save_packet_feature(content, label):
    '''
    如果处理的数量达到了NUM_PER_CATEGORY，pickle.dump()之并返回False,否则返回True
    :param content:
    :param label: 
    :return: 
    '''
    global CURRENT_INDEX, CURRENT_DATASET_LINE_NUM, DATA2SAVE, DATA2SAVE_test
    category = LABEL_CLASS[label]
    print('CURRENT_DATASET_LINE_NUM:', CURRENT_DATASET_LINE_NUM)
    if CURRENT_DATASET_LINE_NUM < TRAIN_VALIDATION_SET_NUM:
        DATA2SAVE.append(content)
        CURRENT_DATASET_LINE_NUM += 1
        return True
    elif TRAIN_VALIDATION_SET_NUM <= CURRENT_DATASET_LINE_NUM < NUM_PER_CATEGORY:
        DATA2SAVE_test.append(content)
        CURRENT_DATASET_LINE_NUM += 1
        return True
    else:
        dataset_path = ROOT_PATH + f'\\{DATA_IS_FROM}_{category}_DataNum{TRAIN_VALIDATION_SET_NUM}.pkl'  # train 和validation的path
        dataset_path_t = ROOT_PATH + f'\\{DATA_IS_FROM}_{category}_DataNum{TEST_SET_NUM}_test.pkl'  # test的path
        print('len:', len(DATA2SAVE))
        print('len_test', len(DATA2SAVE_test))
        with open(dataset_path, 'wb') as f:
            pickle.dump(DATA2SAVE, f)  # Dadaset满了，dump之
            f.close()
        with open(dataset_path_t, 'wb') as f2:
            pickle.dump(DATA2SAVE_test, f2)  # Dadaset满了，dump之
            f2.close()
        DATA2SAVE = []
        DATA2SAVE_test = []
        CURRENT_DATASET_LINE_NUM = 0
        return False


def _create_train_dataset():
    pass


def _create_validation_dataset():
    pass


def _create_test_dataset():
    pass


def _check_data2save(category):
    '''
    这个程序有个bug是如果某个类别的流量包数刚好小于等于NUM_PER_CATEGORY 时，该类别的pkl结果不会被保存，因此在退出该类别的处理
    之前要进行一个检查，保存剩余的数据
    但是还有一个bug就是假设Train:Valid:Test=8:1:1 NUM_PER_CATEGORY=5000 然后有一个类别的流量其数据包少得可怜，假设一共
    只有3000，那么程序处理时Train和Valid的数量应该要等于5000*（9/10）=4500，Test=500，但是实际上3000连4500都达不到，那么
    test数据集没数据，dump的时候会出错，所以NUM_PER_CATEGORY应该遵循木桶原理，在实验前用./tools/PacketNumPerClass.py统计
    每个类别的数据包个数，不能瞎取NUM_PER_CATEGORY
    :return:
    '''
    global DATA2SAVE, DATA2SAVE_test, NUM_PER_CATEGORY, CURRENT_DATASET_LINE_NUM
    if 0 < len(DATA2SAVE) <= NUM_PER_CATEGORY:
        print('我来收尾啦！')
        dataset_path = ROOT_PATH + f'\\{DATA_IS_FROM}_{category}_DataNum{len(DATA2SAVE)}.pkl'  # train 和validation的path
        dataset_path_t = ROOT_PATH + f'\\{DATA_IS_FROM}_{category}_DataNum{len(DATA2SAVE_test)}_test.pkl'  # test的path
        with open(dataset_path, 'wb') as f:
            pickle.dump(DATA2SAVE, f)  # Dadaset满了，dump之
            f.close()
        with open(dataset_path_t, 'wb') as f2:
            pickle.dump(DATA2SAVE_test, f2)  # Dadaset满了，dump之
            f2.close()
        DATA2SAVE = []
        DATA2SAVE_test = []
        CURRENT_DATASET_LINE_NUM = 0


if __name__ == '__main__':
    create_dataset(8, 1, 1)
    # with open(
    #         r'K:\数据库\USTC-TFC2016\1_preprocessed\USTC-TFC2016_benign\AllLayers_flow\USTC-TFC2016_Gmail_DataNum600_test.pkl',
    #         'rb') as f:
    #     data = pickle.load(f)
    #     print(len(data))
    #     print(data[-2:])
