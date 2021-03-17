# Step1 从Splited的Pcap文件中取小量样本移到train validation文件夹中(.tools/5_clusteringPcap.py)
# Step2 对这两个文件夹的样本进行处理，利用scapy库解析每一个pcap文件，根据文件所在的类（Normal,HTTPDOS等等）给它们打上标签
# 每个pcap文件处理的流程为：1.[<label>:Packet1\n<label>:Packet2\n...<label>:PacketN] \n表示换行，最后以二进制编码写入数据库文件DATASET[CURRENT_INDEX]中
# 2.若CURRENT_INDEX=i,文件的行数小于阈值DATASET_SIZE则将1中的结果以追加的形式存在数据库文件DATASET[CURRENT_INDEX]中，否则CURRENT_INDEX+=1,存在数据库文件DATASET[CURRENT_INDEX]中,
# 3.重复1,2直到所有paca文件被处理完，

# Step3:将Train和Validation文件夹的Dataset文件移至新文件夹中，并以pkl形式保存
# 运行顺序:
# run(root_path=r'K:\数据库\ISCX-IDS-2012\3_1sampleDataset\validation')
# run(root_path=r'K:\数据库\ISCX-IDS-2012\3_1sampleDataset\train')
# move_dataset('<new_folder>')

import yaml
from scapy.all import *
import shutil
import pickle

# import vthread

PACKET_LEN = 54
DATASET_SIZE = 3000
CURRENT_INDEX = 0
CURRENT_DATASET_LINE_NUM = 0  # 当前已经写了的行数
DATA2SAVE = []
class_label = {'Normal': 0, 'BFSSH': 1, 'Infilt': 2, 'HttpDoS': 3, 'DDoS': 4}
label_class = {0: 'Normal', 1: 'BFSSH', 2: 'Infilt', 3: 'HttpDoS', 4: 'DDoS'}
DATA_IS_FROM = 'ISCX2012'


def get_current_index():
    with open('config.yaml', 'r')as f:
        yaml_data = yaml.load(f.read(), Loader=yaml.FullLoader)
    return yaml_data['processPcap']['CURRENT_INDEX']


def update_current_index(update_value):
    if not isinstance(update_value, int):
        raise ValueError
    with open('config.yaml', 'r')as f:
        yaml_data = yaml.load(f.read(), Loader=yaml.FullLoader)
        yaml_data['processPcap']['CURRENT_INDEX'] = update_value
    with open('config.yaml', 'w') as f:
        yaml.dump(yaml_data, f)


def reset_current_index():
    with open('config.yaml', 'r')as f:
        yaml_data = yaml.load(f.read(), Loader=yaml.FullLoader)
        yaml_data['processPcap']['CURRENT_INDEX'] = 0
    with open('config.yaml', 'w') as f:
        yaml.dump(yaml_data, f)


def get_lable_pcappath(dir_path):
    '''

    :param dir_path:分好类的文件夹所在的父目录
    :return:
    '''
    label_path = {}
    class_list = os.listdir(dir_path)
    print(class_list)
    for each_class in class_list:
        pcap_name_in_class = glob(os.path.join(dir_path, each_class) + '/*.pcap')
        label = class_label[each_class]
        label_path[label] = pcap_name_in_class
    return label_path


# @vthread.atom
def parse_pcap(pcap_file_path, label):
    global PACKET_LEN
    pcapfile = rdpcap(pcap_file_path)
    packet_num = len(pcapfile)
    for index in range(packet_num):
        packet_feature = raw(pcapfile[index])[:PACKET_LEN]
        content2save = [label, packet_feature]
        print(content2save)
        save_packet_feature(content2save, pcap_file_path)


def save_packet_feature(content, pcap_file_path):
    global CURRENT_INDEX, CURRENT_DATASET_LINE_NUM, DATA2SAVE
    CURRENT_INDEX = get_current_index()
    # current_class = label_class[label]
    print('CURRENT_DATASET_LINE_NUM:', CURRENT_DATASET_LINE_NUM)
    if CURRENT_DATASET_LINE_NUM < DATASET_SIZE:
        DATA2SAVE.append(content)
        CURRENT_DATASET_LINE_NUM += 1
    else:
        dataset_path = ROOT_PATH + f'\\{DATA_IS_FROM}_{CURRENT_INDEX}.pkl'
        check_dataset(dataset_path)
        print('len:', len(DATA2SAVE))
        with open(dataset_path, 'wb') as f:
            pickle.dump(DATA2SAVE, f)  # Dadaset满了，dump之
        DATA2SAVE = []
        update_current_index(CURRENT_INDEX + 1)
        DATA2SAVE.append(content)
        CURRENT_DATASET_LINE_NUM = 1


def run(root_path):
    global ROOT_PATH
    ROOT_PATH = root_path
    reset_current_index()
    labelpath_dict = get_lable_pcappath(root_path)
    for _, label in class_label.items():
        file_path_list = labelpath_dict.get(label)
        for file_path in file_path_list:
            parse_pcap(file_path, label)
    if len(DATA2SAVE):
        # DATA2SAVE中最后长度不足DATASET_SIZE的数据将被舍弃，为了保留这部分数据，我们需要在程序终止前将数据flush出来
        with open(root_path + '/{}_end_remain.pkl'.format(DATA_IS_FROM), 'wb') as f:
            pickle.dump(DATA2SAVE, f)


def check_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        with open(dataset_path, 'w') as f:
            f.close()
        return True
    else:
        return True


def check_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            raise OSError


def move_dataset():
    source_path = ['K:/数据库/ISCX-IDS-2012/3_1sampleDataset/train', 'K:/数据库/ISCX-IDS-2012/3_1sampleDataset/validation']
    goal_path = 'K:/数据库/ISCX-IDS-2012/3_1sampleDataset/Dataset'
    check_dir(goal_path)
    for path in source_path:
        dataset_path = glob(path + '/*/*.pkl')
        print(dataset_path)
        for dataset in dataset_path:
            shutil.move(dataset, os.path.join(goal_path, os.path.basename(dataset)))
    print('over')


if __name__ == '__main__':
    # parse_pcap('testbed-11jun.pcap.TCP_109-72-85-5_80_192-168-2-113_2051.pcap', 0)
    # print(os.path.split('E:\\aaa\\bb/cc.txt')[0])
    run(root_path=r'K:\数据库\ISCX-IDS-2012\3_1sampleDataset\test')
    # run(root_path=r'K:\数据库\ISCX-IDS-2012\3_1sampleDataset\train')
    # move_dataset()
    # with open(r'K:\数据库\ISCX-IDS-2012\3_1sampleDataset\validation\Normal\ISCX2012_Normal_3.pkl','rb') as f:
    #     label_packetList=pickle.load(f)
    #     print(len(label_packetList))
    #     print(label_packetList[:10])
    # print(os.path.split(os.path.split('E:/aaa/bbb/ccc/dd.py')[0]))
