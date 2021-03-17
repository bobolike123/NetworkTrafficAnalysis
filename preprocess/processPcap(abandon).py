import os
import pickle
import glob
import vthread

file_num = 1
CHOOSE_TOP_K_BYTES = 54  # 取每个包前K个字节数，取54是因为一般情况下MAC(14)+IP(20)+TCP(20)
DESTINATION_DIR = r'K:\数据库\ISCX-IDS-2012\2_1_extractFlowFeature'
SOURCE_FLOW_DIR = r'K:\数据库\ISCX-IDS-2012\1_Processed2Session\AllLayers_flow'
dict_5class = {0: 'Normal', 1: 'BFSSH', 2: 'Infilt', 3: 'HttpDoS', 4: 'DDoS'}


# def print_top_k_bytes(k):
#     dirlist = os.listdir(r'K:\数据库\ISCX-IDS-2012\1_Processed2Session\AllLayers\testbed-11jun-ALL')
#     # print(dirlist)
#     for i in range(file_num):
#         with open('K:\数据库\ISCX-IDS-2012\\1_Processed2Session\AllLayers\\testbed-11jun-ALL\\' + dirlist[i], 'rb') as f:
#             content = f.read(k)
#             print(content)


def is_dir_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# @vthread.atom
def process_each_flow(pcapflow_path, method):
    '''

    :param pcapflow_path:文件流路径
    :param method:  str类，传入的pcap_flow文件属于'Normal',  'BFSSH', 'Infilt',  'HttpDoS','DDoS'中的哪种
    :return:
    '''
    PcapHeaderLenth = 24
    PacketHeaderLenth = 16
    packet_indices = 0
    is_dir_exist(os.path.join(DESTINATION_DIR, method))  # 如果目标文件夹不存在，则新建

    with open(pcapflow_path, 'rb') as f:

        f.seek(PcapHeaderLenth, 0)  # 略过PCAP头
        while True:
            s = f.read(PacketHeaderLenth)

            if len(s) < 16:
                break
            else:
                pdl, flag, abs_difference = packetDataLen(s)
                if flag:  # flag == 1 时，包大小大于阈值，截断取前k个（k=阈值大小）字节
                    data = f.read(pdl)
                    f.seek(abs_difference, 1)
                else:
                    print('packet is smaller than expected,padiing zero')
                    data_s = f.read(pdl)
                    data = data_s + bytes.fromhex('00' * abs_difference)
                    # f.seek(total_offset, 1)
                print(data)
                packet_indices += 1

                # with open(os.path.join(DESTINATION_DIR, method,
                #                        os.path.basename(pcapflow_path)) + '_' + str(packet_indices) + '.pkl',
                #           'wb') as f2:
                #     pickle.dump(data, f2)
                with open(os.path.join(DESTINATION_DIR, method,
                                       'ISCX2012_') + method + '_' + str(
                    hash(os.path.basename(pcapflow_path))) + '_' + str(packet_indices) + '.txt',
                          'wb') as f2:
                    f2.write(data)

@vthread.thread(10)
def run(source_pcapflow_path):
    '''
    遍历文件夹，得到文件夹下所有文件的路径,然后执行process_each_flow(),提取每个pcap下每个packet的特征并保存

    用vthread.thread试一下多线程效果
    :param source_pcapflow_path:
    :return: None
    '''
    dir_list = os.listdir(source_pcapflow_path)
    print(dir_list)
    for dir in dir_list:
        print('{}中的数据正在处理：'.format(dir))
        eachDayFlow = os.path.join(source_pcapflow_path, dir)
        eachFile_list = os.listdir(eachDayFlow)
        # print(eachFile_list[0:10])
        for eachFile in eachFile_list:
            if eachFile == 'maliciousFile':  # 统一处理malicious file
                maliciousFile_list = os.listdir(os.path.join(eachDayFlow, eachFile))
                for f in maliciousFile_list:
                    # 处理malicious文件夹的每个文件，按照标签生成新的数据(注意：所有正常的数据会混在一起)
                    eachFilePath = os.path.join(eachDayFlow, eachFile, f)
                    # if '12jun' in dir or '16jun' in dir:  # 运算符优先级 in > or，'11jun'不含malicious文件夹，不讨论
                    #     # 12jun 和16jun 虽然有malicious文件夹，但是根据官网描述，这两天的流量都是正常的，故以正常流量处理
                    #     process_each_flow(eachFilePath, dict_5class[0])  # Normal
                    if '13jun' in dir:
                        process_each_flow(eachFilePath, dict_5class[2])  # Infilt
                    if '14jun' in dir:
                        process_each_flow(eachFilePath, dict_5class[3])  # HttpDoS
                    if '15jun' in dir:
                        process_each_flow(eachFilePath, dict_5class[4])  # DDoS
                    if '17jun' in dir:
                        process_each_flow(eachFilePath, dict_5class[1])  # BFSSH
            else:
                pass
                # eachFilePath = os.path.join(source_pcapflow_path, dir, eachFile)
                # process_each_flow(eachFilePath, dict_5class[0])  # dict_5class[0]表示这些要处理的文件是'Normal'


def packetDataLen(byte_stream):
    # print(byte_stream)
    nextPacketLenth = byte_stream[-4:]
    # print('npl:', nextPacketLenth)
    currentPacketDataLen = int.from_bytes(nextPacketLenth, byteorder='big', signed=False)
    # print('Packet_len:', currentPacketDataLen)
    if currentPacketDataLen >= CHOOSE_TOP_K_BYTES:
        return CHOOSE_TOP_K_BYTES, 1, currentPacketDataLen - CHOOSE_TOP_K_BYTES  # 第三个返回值用于修正文件指针的位置
    else:
        return currentPacketDataLen, 0, CHOOSE_TOP_K_BYTES - currentPacketDataLen


if __name__ == '__main__':
    # print_top_k_bytes(54)
    # flow_path = './testbed-11jun.pcap.TCP_109-72-85-5_80_192-168-2-113_2051.pcap'
    # process_each_flow(flow_path)
    # print(b'\xab' + bytes.fromhex('00' * 4))
    run(SOURCE_FLOW_DIR)
