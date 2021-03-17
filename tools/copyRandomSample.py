# 拷贝数据集中每个类别的指定数量的数据至新的位置，并将这些数据矢量化
import shutil
import os
import random

# 因为原始数据太多了，这里按照分布数量取原数据集大约0.1比例的数据
# 因为Normal文件夹下本来大概有300w数据的，因为生成慢，只有90w，这里按照300w数据处理
# 如果Normal数据全的话，设置EXTRA_RATIO=0.1就可以取0.1比例数据了，这里也是没有办法才手动设置数量的
dict_num = {'Normal': 300000, 'Infilt': 14000, 'HttpDoS': 7000, 'DDoS': 7000, 'BFSSH': 5000}


def readFilenameList(path):
    if list == None:
        print('filename list is empty')
    else:
        filelist = os.listdir(path)
        return filelist


class createBlankFile:
    def __init__(self, originPath, goalPath):
        self.origin_path = originPath
        self.goal_path = goalPath

    def createNewFiles(self):
        filenameList = readFilenameList(self.origin_path)
        for fn in filenameList:
            newPath = self.goal_path + '\{}'.format(fn)
            if not os.path.exists(newPath):
                os.makedirs(newPath)
                print(newPath + " created")
        print('Files have been created')


class divideDataSet:
    # EXTRACT_RATIO = 0.2   每个家族抽取样本的比例 ,0.2表示按照8:2划分训练集和验证集

    def __init__(self, origin_path, goal_path, EXTRACT_RATIO=0.2):
        self.origin_path = origin_path
        self.goal_path = goal_path
        self.EXTRACT_RATIO = EXTRACT_RATIO

    def checkFileisExist(self):
        try:
            createBlankFile(self.origin_path, self.goal_path).createNewFiles()
            Flag = True
        except:
            print('something wrong in checking File is exist')
            Flag = False
        return Flag

    def copyFile(self):
        list = readFilenameList(self.origin_path)
        # print(list)
        flag = self.checkFileisExist()
        if flag == False: return 'moveFile Failed'
        for malFamily in list:  # 对train文件夹下的每个子文件夹，随机取出EXTRACT_RATIO比例的文件放入validation文件夹中
            file_abs_path = self.origin_path + "\\" + malFamily
            pathDir = readFilenameList(file_abs_path)
            # filenumber = len(pathDir)
            # picknumber = int(filenumber * self.EXTRACT_RATIO)
            picknumber = dict_num[malFamily]
            sample = random.sample(pathDir, picknumber)
            # print(sample)
            for name in sample:
                # shutil.move(self.origin_path + '\\' + malFamily + '\\' + name,
                #             self.goal_path + '\\' + malFamily + '\\' + name)
                shutil.copy(self.origin_path + '\\' + malFamily + '\\' + name,
                            self.goal_path + '\\' + malFamily + '\\' + name)
        return 'moveFile Succeed!'

if __name__ == '__main__':
    divideDataSet(origin_path=r'K:\数据库\ISCX-IDS-2012\2_extractFlowFeature',goal_path=r'K:\数据库\ISCX-IDS-2012\3_sampleDataset').copyFile()