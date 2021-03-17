# SplitCap程序切分Pcap命名规则：原pcap文件名.<协议名>_原ip地址_原端口_目的ip地址_目的端口
# 其中目的ip和原ip地址用-替代.  即用192-168-0-1代替192.168.0.1

# 本代码用于预处理CICIDS2017数据集，无需解析xml文件
import shutil
import os
import preprocess.configuration
import errno
import glob

InfoList = ["source:", "protocolName:", "sourcePort:", "destination:", "destinationPort:"]

def dealwithresult(glob_list,classname):
    '''
    '''
    for fpath in glob_list:
        fname = os.path.basename(fpath)
        src_path = fpath
        dst_path = os.path.join(destinationPath,classname,fname)
        # print(src_path,dst_path)
        try:
            shutil.move(src_path,dst_path)
        except OSError as e:
            print(e)
    print(f'class: {classname} has been categorized to {destinationPath}')


def process_ip(ip_list):
    combination_1 = [item.replace('.', '-') for item in ip_list]
    protocol = "*"
    sport = "*"
    combination_2 = [item.replace('.', '-') for item in reversed(ip_list)]
    dport = "*"
    output_1 = protocol + '_' + combination_1[0] + '_' + sport + '_' + combination_1[1] + '_' + dport + '.pcap'
    output_2 = protocol + '_' + combination_2[0] + '_' + sport + '_' + combination_2[1] + '_' + dport + '.pcap'
    # print(output_1)
    # print(output_2)
    return output_1, output_2
    # outputSet.add(output)  # 直接用set集合保存，筛除了重名的文件


def ip_defination(class_name):

    Normal_ip = ['*', '*']
    BruteForce_ip = ['172.16.0.1', '192.168.10.50']
    DoS_DDoS_ip = ['172.16.0.1', '192.168.10.50']
    WebAttack_ip = ['172.16.0.1', '192.168.10.50']
    Infiltration_ip = ['205.174.165.73', '*']
    BotNet_ip = ['205.174.165.73', '*']
    PortScan_ip = ['172.16.0.1', '192.168.10.50']
    if class_name == 'Normal':
        ip = Normal_ip
    elif class_name == "BruteForce":
        ip = BruteForce_ip
    elif class_name == 'DoS_DDoS':
        ip = DoS_DDoS_ip
    elif class_name == 'WebAttack':
        ip = WebAttack_ip
    elif class_name == 'Infiltration':
        ip = Infiltration_ip
    elif class_name == 'BotNet':
        ip = BotNet_ip
    else:
        ip = PortScan_ip

    return process_ip(ip)


def glob_and_move(dir_path, fname_tuple,classname):
    global destinationPath
    destinationPath = 'K:/dataset/CIC-IDS-2017/3_Categorized'
    mkdir_p(os.path.join(destinationPath,classname))

    glob_path=os.path.join(dir_path, fname_tuple[0])
    print(glob_path)
    glob_result = glob.glob(glob_path)
    if len(glob_result) > 0 :
        dealwithresult(glob_result,classname)

    glob_path=os.path.join(dir_path, fname_tuple[1])
    print(glob_path)
    glob_result = glob.glob(glob_path)
    if len(glob_result) > 0 :
        dealwithresult(glob_result,classname)




def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def findpcapname(logName):
    '''
     查找configuration中的字典XML2PCAP，找到XML文件对应的PCAP名
    :param logName:
    :return: 对应的PCAP名
    '''
    dict = preprocess.configuration.XML2PCAP
    value = dict.get(os.path.splitext(logName)[0].split('_')[1])
    # print(value)
    return value


'''
CICIDS2017_PathDict = {'Normal': 'K:/dataset/CIC-IDS-2017/2_Session/AllLayers/Monday-ALL',
                       'BruteForce': 'K:/dataset/CIC-IDS-2017/2_Session/AllLayers/Tuesday-ALL',
                       'DoS_DDoS': 'K:/dataset/CIC-IDS-2017/2_Session/AllLayers/Wednesday-ALL',
                       'WebAttack': 'K:/dataset/CIC-IDS-2017/2_Session/AllLayers/Thursday-ALL/WebAttack',
                       'Infiltration': 'K:/dataset/CIC-IDS-2017/2_Session/AllLayers/Thursday-ALL/Inflit',
                       'BotNet': 'K:/dataset/CIC-IDS-2017/Fri-PortScan_Botnet/2_Session/AllLayers/Botnet',
                       'PortScan': 'K:/dataset/CIC-IDS-2017/Fri-PortScan_Botnet/2_Session/AllLayers/PortScan
'''


def run():
    splitedPcapFilePath = preprocess.configuration.CICIDS2017_PathDict
    for k, v in splitedPcapFilePath.items():
        print(k, v)
        fname_tuple = ip_defination(k)
        glob_and_move(v,fname_tuple,k)
        # parselog(k)
        # dealwithresult(splitPcapPath + v+'-ALL')


if __name__ == '__main__':
    # parselog()
    # dealwithresult()
    # findpcapname(currentLogName)
    run()
