# This file save the configuration setting
# 注意:TestbedThuJun17-1Flows.xml有非法字符 0x00，请使用XMLfilter.py删除，否则无法正常解析

MALICIOUS_DATALOG_PATH = './datalog/'

# a dictionary for XML filename and its corresponding PCAP filename
XML2PCAP = {'TestbedSatJun12Flows': 'testbed-12jun.pcap', 'TestbedSunJun13Flows': 'testbed-13jun.pcap',
            'TestbedMonJun14Flows': 'testbed-14jun.pcap', 'TestbedTueJun15-1Flows': 'testbed-15jun.pcap',
            'TestbedTueJun15-2Flows': 'testbed-15jun.pcap', 'TestbedTueJun15-3Flows': 'testbed-15jun.pcap',
            'TestbedWedJun16-1Flows': 'testbed-16jun.pcap', 'TestbedWedJun16-2Flows': 'testbed-16jun.pcap',
            'TestbedWedJun16-3Flows': 'testbed-16jun.pcap', 'TestbedThuJun17-1Flows': 'testbed-17jun.pcap',
            'TestbedThuJun17-2Flows': 'testbed-17jun.pcap', 'TestbedThuJun17-3Flows': 'testbed-17jun.pcap'}

# total XML file to deal with
total_XML = ['TestbedSatJun12Flows.xml', 'TestbedSunJun13Flows.xml', 'TestbedMonJun14Flows.xml',
             'TestbedTueJun15-1Flows.xml', 'TestbedTueJun15-2Flows.xml', 'TestbedTueJun15-3Flows.xml',
             'TestbedWedJun16-1Flows.xml', 'TestbedWedJun16-2Flows.xml', 'TestbedWedJun16-3Flows.xml',
             'TestbedThuJun17-1Flows.xml', 'TestbedThuJun17-2Flows.xml', 'TestbedThuJun17-3Flows.xml']

# total Log file and its corresponding PCAP folder filename
Log2Folder = {'maliciousDataLog_TestbedSatJun12Flows.txt': 'testbed-12jun',
              'maliciousDataLog_TestbedSunJun13Flows.txt': 'testbed-13jun',
              'maliciousDataLog_TestbedMonJun14Flows.txt': 'testbed-14jun',
              'maliciousDataLog_TestbedTueJun15-1Flows.txt': 'testbed-15jun',
              'maliciousDataLog_TestbedTueJun15-2Flows.txt': 'testbed-15jun',
              'maliciousDataLog_TestbedTueJun15-3Flows.txt': 'testbed-15jun',
              'maliciousDataLog_TestbedWedJun16-1Flows.txt': 'testbed-16jun',
              'maliciousDataLog_TestbedWedJun16-2Flows.txt': 'testbed-16jun',
              'maliciousDataLog_TestbedWedJun16-3Flows.txt': 'testbed-16jun',
              'maliciousDataLog_TestbedThuJun17-1Flows.txt': 'testbed-17jun',
              'maliciousDataLog_TestbedThuJun17-2Flows.txt': 'testbed-17jun',
              'maliciousDataLog_TestbedThuJun17-3Flows.txt': 'testbed-17jun'}

# XML files path
XML_files_path = 'K:/数据库/ISCX-IDS-2012/labeled_flows_xml/'  # 记得以/结尾

# Split pcap file path(the path which pcap files have been done by 1_Pcap2Session.ps1)
Split_pcap_path = 'K:/数据库/ISCX-IDS-2012/1_Processed2Session/AllLayers_flow/'  # 记得以/结尾

# Special construction of XML file
# （some XML file construction is different from others,for example 'TestbedMonJun14Flows.xml'）
special_XML = ['TestbedMonJun14Flows.xml']

# XML file including illegal character
illegal_XML = ['TestbedThuJun17-1Flows.xml']

# CICIDS2017切分后的数据所在的路径，每个种类对应一个key-value键值对
CICIDS2017_PathDict = {'Normal': 'K:/dataset/CIC-IDS-2017/2_Session/AllLayers/Monday-ALL',
                       'BruteForce': 'K:/dataset/CIC-IDS-2017/2_Session/AllLayers/Tuesday-ALL',
                       'DoS_DDoS': 'K:/dataset/CIC-IDS-2017/2_Session/AllLayers/Wednesday-ALL',
                       'WebAttack': 'K:/dataset/CIC-IDS-2017/2_Session/AllLayers/Thursday-ALL/WebAttack',
                       'Infiltration': 'K:/dataset/CIC-IDS-2017/2_Session/AllLayers/Thursday-ALL/Inflit',
                       'BotNet': 'K:/dataset/CIC-IDS-2017/Fri-PortScan_Botnet/2_Session/AllLayers/Botnet',
                       'PortScan': 'K:/dataset/CIC-IDS-2017/Fri-PortScan_Botnet/2_Session/AllLayers/PortScan'}
