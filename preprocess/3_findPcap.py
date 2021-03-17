# SplitCap程序切分Pcap命名规则：原pcap文件名.<协议名>_原ip地址_原端口_目的ip地址_目的端口
# 其中目的ip和原ip地址用-替代.  即用192-168-0-1代替192.168.0.1

# 本代码是根据XMLParser.py找到的恶意流量记录，也就是生成的txt文件中的记录找到对应的切分后的pcap文件‘
import shutil
import os
import preprocess.configuration
import errno

InfoList = ["source:", "protocolName:", "sourcePort:", "destination:", "destinationPort:"]
outputSet = set()
# currentLogName = 'maliciousDataLog_TestbedSunJun13Flows.txt'
outputResultFile = 'output_result.txt'


# sourceSplitedFilesPath = 'K:/数据库/ISCX-IDS-2012/1_Processed2Session/AllLayers/testbed-13jun-ALL'
# destinationPath = sourceSplitedFilesPath + '/maliciousFile'


def parselog(currentLogName):
    # read the log file
    dataDict = {}
    outputSet.clear()
    with open(currentLogName, 'r') as f:
        for line in f.readlines():
            line = line.strip()  # 去掉每行头尾空白
            for attribution in InfoList:
                if line.startswith(attribution):
                    dataDict[attribution] = line[len(attribution):]
            processdatadict(dataDict, findpcapname(logName=currentLogName))
    with open(outputResultFile, 'w') as f:
        for item in outputSet:
            f.write(item + '\n')


def dealwithresult(sourceSplitedFilesPath):
    '''
    Step 1 ：导入outputResult中的文件名
    Step 2 :用shutil.move将Step1 中对应的文件移至新的文件夹
    :return:
    '''
    # folder2read = sourceSplitedFilesPath
    # filelist = os.listdir(folder2read)
    # for files in filelist:
    #     filename1 = os.path.splitext(files)[1]  # 读取文件后缀名
    #     filename0 = os.path.splitext(files)[0]  # 读取文件名
    #     # print("文件名：",filename0," 后缀：",filename1)
    #     print(files)
    destinationPath = sourceSplitedFilesPath + '/maliciousFile'
    mkdir_p(destinationPath)
    with open(outputResultFile, 'r') as f:
        namelist = f.readlines()
        # print(namelist)
        for filename in namelist:
            # print(filename.strip('\n'))
            try:
                shutil.move(sourceSplitedFilesPath + '/' + filename.strip('\n'),
                            destinationPath + '/' + filename.strip('\n'))
            except FileNotFoundError:
                print('file:' + filename.strip('\n') + ' does not exist')
        print("operation finished")


def processdatadict(dict, pcapname):
    sip = str(dict.get("source:")).replace('.', '-')
    protocol = "TCP" if dict.get("protocolName:") == "tcp_ip" else"UDP"
    sport = str(dict.get("sourcePort:"))
    dip = str(dict.get("destination:")).replace('.', '-')
    dport = str(dict.get("destinationPort:"))
    output = pcapname + '.' + protocol + '_' + sip + '_' + sport + '_' + dip + '_' + dport + '.pcap'
    outputSet.add(output)  # 直接用set集合保存，筛除了重名的文件


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


def run():
    splitPcapPath = preprocess.configuration.Split_pcap_path
    LogName_dict = preprocess.configuration.Log2Folder
    dir_list = os.listdir(splitPcapPath)
    for k,v in LogName_dict.items():
        # print(k,v)
        parselog(k)
        dealwithresult(splitPcapPath + v+'-ALL')


if __name__ == '__main__':
    # parselog()
    # dealwithresult()
    # findpcapname(currentLogName)
    run()
