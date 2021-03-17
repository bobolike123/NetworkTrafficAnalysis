import os
import random
import shutil

def readFilenameList(path):
    if list == None:
        print('filename list is empty')
    else:
        filelist = os.listdir(path)
        return filelist

def runAPP(origin_path,goal_path,divide_ratio):

    print('copying from origin_path...')
    copyData(origin_path,goal_path+'\\train').run()

    print('dividing origin dataset into train and validation dataset with the ratio :',divide_ratio)
    dds=divideDataSet(goal_path+'\\train',goal_path+'\\validation',divide_ratio)
    dds.moveFile()

class createBlankFile:
    def __init__(self,originPath,goalPath):
        self.origin_path=originPath
        self.goal_path=goalPath

    def createNewFiles(self):
        filenameList = readFilenameList(self.origin_path)
        for fn in filenameList:
            newPath = self.goal_path + '\{}'.format(fn)
            if not os.path.exists(newPath):
                os.makedirs(newPath)
                print(newPath+" created")
        print('Files have been created')

class copyData:
    def __init__(self,origin_path,goal_path):
        self.origin_path=origin_path
        self.goal_path=goal_path

    def run(self):
        shutil.copytree(self.origin_path,self.goal_path)

class divideDataSet:
    # EXTRACT_RATIO = 0.2   每个家族抽取样本的比例 ,0.2表示按照8:2划分训练集和验证集

    def __init__(self,origin_path,goal_path,EXTRACT_RATIO=0.2):
        self.origin_path=origin_path
        self.goal_path=goal_path
        self.EXTRACT_RATIO=EXTRACT_RATIO
    def checkFileisExist(self):
        try:
            createBlankFile(self.origin_path,self.goal_path).createNewFiles()
            Flag = True
        except:
            print('something wrong in checking File is exist')
            Flag = False
        return Flag

    def moveFile(self):
        list = readFilenameList(self.origin_path)
        # print(list)
        flag = self.checkFileisExist()
        if flag == False: return 'moveFile Failed'
        for malFamily in list:                                              # 对train文件夹下的每个子文件夹，随机取出EXTRACT_RATIO比例的文件放入validation文件夹中
            file_abs_path = self.origin_path + "\\" + malFamily
            pathDir = readFilenameList(file_abs_path)
            filenumber = len(pathDir)
            picknumber = int(filenumber * self.EXTRACT_RATIO)
            sample = random.sample(pathDir, picknumber)
            # print(sample)
            for name in sample:
                shutil.move(self.origin_path + '\\' + malFamily + '\\' + name,
                            self.goal_path + '\\' + malFamily + '\\' + name)
        return 'moveFile Succeed!'

if __name__ == '__main__':
    runAPP(r'G:\实验\malimg_dataset\colorImg_squareSize',r'G:\3PGCIC-2019\实验结果\colorImg(8比2)',0.2)
