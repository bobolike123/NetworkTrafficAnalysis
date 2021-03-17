from scapy.all import *
import os
import glob
import matplotlib.pyplot as plt

filename = r"F:\MyCode\pyCode\USTC-TK2016\3_ProcessedSession\FilteredSession\Train\BitTorrent-ALL\BitTorrent.pcap.TCP_1-1-0-26_49645_1-2-9-52_443.pcap"


def udp():
    packets = rdpcap(filename)
    for data in packets:
        if 'UDP' in data:
            s = repr(data)
            print(s)
            print(data['UDP'].sport)


def tcp():
    packets = rdpcap(filename)
    for data in packets:
        if 'TCP' in data:
            print(data)
            # print(hexdump(data))
            s = repr(data)
            print(s)
            print(data['TCP'].sport)


def my_sr():
    # sr(IP(dst="192.168.8.1") / TCP(sport=RandShort(), dport=[440, 441, 442, 443], flags="S"))
    ans, unans = srloop(IP(dst=["www.baidu.com", "www.qq.com"]) / ICMP(), inter=.1, timeout=.1, count=100,
                        verbose=False)
    fig = ans.multiplot(lambda p, q: (q[IP].src, (q.time, q[IP].id)), plot_xy=True)


'''

'''


def get_lable_pcappath(dir_path):
    '''

    :param dir_path:分好类的文件夹所在的父目录
    :return:
    '''
    class_label = {'Normal': 0, 'BFSSH': 1, 'Infilt': 2, 'HttpDoS': 3, 'DDoS': 4}

    label_path={}
    class_list = os.listdir(dir_path)
    print(class_list)
    for each_class in class_list:
        pcap_name_in_class = glob.glob(os.path.join(dir_path, each_class) + '/*.pcap')
        label=class_label[each_class]
        label_path[label]=pcap_name_in_class
    return label_path

def parse_pcap(pcap_file_path):
    PACKET_LEN = 54
    pcapfile = rdpcap(pcap_file_path)
    packet_num = len(pcapfile)
    # pcapfile = rdpcap(pcap_file_path,count=10) 加上count=10只读取前10个packet

    # for j, byte in enumerate(raw(pcapfile[0])[:PACKET_LEN]):
    #     print(j,byte)

    # print(raw(pcapfile[0])[:PACKET_LEN])
    # for index in range(packet_num):
    #     pass
    # print(f'index:{index}',pcapfile[index].show())


if __name__ == '__main__':
    # udp()
    # tcp()
    # parse_pcap('testbed-11jun.pcap.TCP_109-72-85-5_80_192-168-2-113_2051.pcap')
    label_path_dict=get_lable_pcappath(r'K:\数据库\ISCX-IDS-2012\3_1sampleDataset\train')
    print(len(label_path_dict.get(0)))