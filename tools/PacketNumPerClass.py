from scapy.all import *
import glob
import gc
# path_list = ['K:/数据库/ISCX-IDS-2012/1_Processed2Session/AllLayers_flow/testbed-17jun-ALL/maliciousFile']
# 总数为49667

path_list = ['K:/数据库/ISCX-IDS-2012/2_clusteringPcap/Infilt']


# 总数147376

def run(path):
    sum = 0
    file_list = glob.glob(path + '/*.pcap')
    # print(file_list)
    for file in file_list:
        packet_num_in_file = len(rdpcap(file))
        sum += packet_num_in_file

    print('包总数为：{}'.format(sum))


def run_all(root_path):
    sum = 0
    sum_dict = {}
    dir_list = os.listdir(root_path)
    print(dir_list)
    for dir in dir_list:
        file_list = glob.glob(root_path + '\\' + dir + '\\*.pcap')
        print(f'类别{dir}的Packet数正在统计...')
        for file in file_list:
            # print(file)
            packet_num_in_file = len(rdpcap(file))
            sum += packet_num_in_file
        print(f'类别{dir}的Packet num = {sum}')
        sum_dict[dir] = sum
        sum = 0
    print(sum_dict)

def run_big_data(root_path,classname):
    '''
    对某一类数量巨大（比如接近一千万）的种类进行分批次数量统计，
    如总数=批次1+批次2+……批次n
    :param root_path:
    :return:
    '''
    sum = 0
    sum_list=[] #储存不同批次的计数结果
    sum_dict = {}
    dir_list = os.listdir(root_path)
    print(dir_list)
    for dir in dir_list:
        if dir != classname :continue
        else:
            file_list = glob.glob(root_path + '\\' + dir + '\\*.pcap')
            flen=len(file_list)
            print(f'类别{dir}的Packet数正在统计,该类packet数为{flen}')
            for i in range(10):
                for file in file_list[int(i*flen/10):int((i+1)*flen/10)]:
                    # print(file)
                    packet_num_in_file = len(rdpcap(file))
                    sum += packet_num_in_file
                print(f'批次{i}的Packet num = {sum}')
                sum_list.append(sum)
                sum = None
                gc.collect()
    print(sum_list)
    '''
    USTC-TFC2016 的benign类统计数据
    {'BitTorrent': 15000, 'Facetime': 6000, 'FTP': 360000, 'Gmail': 25000, 'MySQL': 200000, 'Outlook': 15000,
     'Skype': 12000, 'SMB-1': 771886, 'SMB-2': 153566, 'Weibo-1': 756067, 'Weibo-2': 151674, 'Weibo-3': 151069, 
     'Weibo-4': 151249, 'WorldOfWarcraft': 140000}
     
      USTC-TFC2016 的malware类统计数据
     {'Cridex': 461452, 'Geodo': 213238, 'Htbot': 169371, 'Miuref': 81208, 'Neris': 497857, 'Nsis-ay': 351537,
      'Shifu': 499777, 'Tinba': 21912, 'Virut': 437549, 'Zeus': 86198}
    '''

    '''
    CICIDS2017的数据
    BotNet：18168  BruteForce:274938  DoS_DDoS: 2440595 Infiltration: 148956 PortScan:1604317 WebAttack:40791 Normal:11709971
    '''

if __name__ == '__main__':
    # path=path_list[0]
    # run(path)
    # run_all(r'K:\dataset\CIC-IDS-2017\3_Categorized')
    run_big_data(r'K:\dataset\CIC-IDS-2017\3_Categorized','Normal')