# TestbedThuJun17-1Flows.xml 内含非法字符0x00,需要将其过滤，否则sax解析报错
import preprocess.configuration
import re


def xmlfilter():
    content = ''
    XML_files_path = preprocess.configuration.XML_files_path
    illegalXML = preprocess.configuration.illegal_XML  # 导入非法XML文件名列表
    for i in range(len(illegalXML)):
        with open(XML_files_path + illegalXML[i], 'r+') as f:
            content = f.read()
            content = re.sub(u"[\x00-\x08\x0b-\x0c\x0e-\x1f]+", u"", content)
        with open(XML_files_path + illegalXML[i], 'w') as f1:
            f1.write(content)
            print('OK')


if __name__ == '__main__':
    xmlfilter()
