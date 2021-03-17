# !/usr/bin/python
# -*- coding: UTF-8 -*-

import xml.sax
import os
import preprocess.configuration


class XMLHandler(xml.sax.ContentHandler):
    def __init__(self, logfile, isspecial):
        self.CurrentData = ""
        self.isSpecial = isspecial
        self.logFile = logfile  # path to save the log file
        self.count = 0
        self.isMalicious = False
        self.dict = {}
        self.maliciousNum = 0

    def startDocument(self):
        with open(self.logFile, 'w')as f:
            f.close()  # 这一步仅为了清空日志内容

    # 元素开始事件处理
    def startElement(self, tag, attributes):
        self.CurrentData = tag
        if str.startswith(tag, 'Testbed'):
            self.count += 1
            # print("*****No:{}*****".format(self.count))
            # title = attributes["title"]
            # print
            # ("Title:", title)

    # 元素结束事件处理
    def endElement(self, tag):
        if self.CurrentData == 'Tag':
            if self.isMalicious:
                self.maliciousNum += 1
                with open(self.logFile, 'a') as f:
                    f.write("*****No:{}*****\n".format(self.count))
                    if self.isSpecial == False:
                        f.write("appName:{appName}\n"
                                "totalSourceBytes:{totalSourceBytes}\n"
                                "totalDestinationBytes:{totalDestinationBytes}\n"
                                "totalDestinationPackets:{totalDestinationPackets}\n"
                                "totalSourcePackets:{totalSourcePackets}\n"
                                "sourcePayloadAsBase64:{sourcePayloadAsBase64}\n"
                                "sourcePayloadAsUTF:{sourcePayloadAsUTF}\n"
                                "destinationPayloadAsBase64:{destinationPayloadAsBase64}\n"
                                "destinationPayloadAsUTF:{destinationPayloadAsUTF}\n"
                                "direction:{direction}\n"
                                "sourceTCPFlagsDescription:{sourceTCPFlagsDescription}\n"
                                "destinationTCPFlagsDescription:{destinationTCPFlagsDescription}\n"
                                "source:{source}\n"
                                "protocolName:{protocolName}\n"
                                "sourcePort:{sourcePort}\n"
                                "destination:{destination}\n"
                                "destinationPort:{destinationPort}\n"
                                "startDateTime:{startDateTime}\n"
                                "stopDateTime:{stopDateTime}\n"
                                "Tag:{Tag}\n".format(**self.dict))
                    else:
                        f.write("appName:{appName}\n"
                                "totalSourceBytes:{totalSourceBytes}\n"
                                "totalDestinationBytes:{totalDestinationBytes}\n"
                                "totalDestinationPackets:{totalDestinationPackets}\n"
                                "totalSourcePackets:{totalSourcePackets}\n"
                                "sourcePayloadAsBase64:{sourcePayloadAsBase64}\n"
                                # "sourcePayloadAsUTF:{sourcePayloadAsUTF}\n"    特殊结构没有这个
                                "destinationPayloadAsBase64:{destinationPayloadAsBase64}\n"
                                "destinationPayloadAsUTF:{destinationPayloadAsUTF}\n"
                                "direction:{direction}\n"
                                "sourceTCPFlagsDescription:{sourceTCPFlagsDescription}\n"
                                "destinationTCPFlagsDescription:{destinationTCPFlagsDescription}\n"
                                "source:{source}\n"
                                "protocolName:{protocolName}\n"
                                "sourcePort:{sourcePort}\n"
                                "destination:{destination}\n"
                                "destinationPort:{destinationPort}\n"
                                "startDateTime:{startDateTime}\n"
                                "stopDateTime:{stopDateTime}\n"
                                "Tag:{Tag}\n".format(**self.dict))
        self.CurrentData = ""
        self.isMalicious = False

    # 内容事件处理
    def characters(self, content):
        '''
        聪明办法（利用字典），扩展性好
        :param content:
        :return:
        '''
        attr_list = ["appName", "totalSourceBytes", "totalDestinationBytes", "totalDestinationPackets",
                     "totalSourcePackets", "sourcePayloadAsBase64", "sourcePayloadAsUTF", "destinationPayloadAsBase64",
                     "destinationPayloadAsUTF", "direction", "sourceTCPFlagsDescription",
                     "destinationTCPFlagsDescription", "source", "protocolName", "sourcePort", "destination",
                     "destinationPort", "startDateTime", "stopDateTime", "Tag"]
        for item in attr_list:
            if self.CurrentData == item:
                if item == "Tag":
                    if content == "Attack":
                        self.isMalicious = True
                self.dict["{}".format(item)] = content
        '''
        笨方法
                if self.CurrentData == "appName":
                    self.appName = content
                elif self.CurrentData == "totalSourceBytes":
                    self.totalSourceBytes = content
                elif self.CurrentData == "totalDestinationBytes":
                    self.totalDestinationBytes = content
                elif self.CurrentData == "totalDestinationPackets":
                    self.totalDestinationPackets = content
                elif self.CurrentData == "totalSourcePackets":
                    self.totalSourcePackets = content
                elif self.CurrentData == "sourcePayloadAsBase64":
                    self.sourcePayloadAsBase64 = content
                elif self.CurrentData == "sourcePayloadAsUTF":
                    self.sourcePayloadAsUTF = content
                elif self.CurrentData == "destinationPayloadAsBase64":
                    self.destinationPayloadAsBase64 = content
                elif self.CurrentData == "destinationPayloadAsUTF":
                    self.destinationPayloadAsUTF = content
                elif self.CurrentData == "direction":
                    self.direction = content
                elif self.CurrentData == "sourceTCPFlagsDescription":
                    self.sourceTCPFlagsDescription = content
                elif self.CurrentData == "destinationTCPFlagsDescription":
                    self.destinationTCPFlagsDescription = content
                elif self.CurrentData == "source":
                    self.source = content
                elif self.CurrentData == "protocolName":
                    self.protocolName = content
                elif self.CurrentData == "sourcePort":
                    self.sourcePort = content
                elif self.CurrentData == "destination":
                    self.destination = content
                elif self.CurrentData == "destinationPort":
                    self.destinationPort = content
                elif self.CurrentData == "startDateTime":
                    self.startDateTime = content
                elif self.CurrentData == "stopDateTime":
                    self.stopDateTime = content
                elif self.CurrentData == "Tag":
                    self.Tag = content
                    if self.Tag != 'Normal':
                        self.isMalicious= True
        '''
        # XML文档操作结束时写入统计信息

    def endDocument(self):
        with open(self.logFile, 'a')as f:
            f.write("###############Summary###################\n"
                    "total file num:{}\n"
                    "malicious file num:{}\n"
                    "malicious rate:{:.4f}".format(self.count, self.maliciousNum, self.maliciousNum / self.count))


def run():
    '''
    如出现 SAXParseException: xxxxx not well-formed (invalid token) 字样说明SML内存在非法字符，使用XMLfilter.py将相应
    XML文件过滤
    :return:
    '''

    XMLfile_list = preprocess.configuration.total_XML  # 一次性解析完所有的XML文件
    XMLfile_path = preprocess.configuration.XML_files_path
    specialXML = preprocess.configuration.special_XML  # 特殊XML文件特殊对待
    for i in range(len(XMLfile_list)):
        # 创建一个 XMLReader
        parser = xml.sax.make_parser()
        # turn off namepsaces
        parser.setFeature(xml.sax.handler.feature_namespaces, 0)

        XMLfileName = XMLfile_list[i]

        logFile = "./maliciousDataLog_{}.txt".format(os.path.split(XMLfileName)[-1].split('.')[-2])
        print('file:', logFile, ' is creating')
        # 重写 ContextHandler

        if specialXML.count(XMLfileName) > 0:  # 如果该XML是特殊XML
            Handler = XMLHandler(logFile, isspecial=True)
        else:  # 如果是普通XML
            Handler = XMLHandler(logFile, isspecial=False)
        parser.setContentHandler(Handler)
        XMLfile = XMLfile_path + XMLfileName
        parser.parse(XMLfile)


if __name__ == "__main__":
    run()
    # parser = xml.sax.make_parser()
    # # turn off namepsaces
    # parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    #
    # XMLfile_path = preprocess.configuration.XML_files_path
    # XMLfileName = 'TestbedMonJun14Flows.xml'
    # logFile = "./maliciousDataLog_{}.txt".format(os.path.split(XMLfileName)[-1].split('.')[-2])
    #
    # print('file:', logFile, ' is creating')
    # # 重写 ContextHandler
    # Handler = XMLHandler(logFile)
    # parser.setContentHandler(Handler)
    #
    # XMLfile = XMLfile_path + XMLfileName
    # parser.parse(XMLfile)
