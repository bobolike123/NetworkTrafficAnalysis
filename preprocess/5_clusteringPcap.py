# 拷贝数据集中每个类别的指定数量的数据至新的位置，并将这些数据矢量化
# ！！经验证，move太慢了，因为文件数量太多，python不行，但是这个代码不跑6_createInpudData实现不了，或者直接在3_findPcap
# 得到的 结果里面取Inputdata,但是要把6_createInpudData的代码改改
import shutil
import os
import glob
import random

# 因为原始数据太多了，这里按照分布数量取原数据集大约0.1比例的数据
# 因为Normal文件夹下本来大概有300w数据的，因为生成慢，只有90w，这里按照300w数据处理
# 如果Normal数据全的话，设置EXTRA_RATIO=0.1就可以取0.1比例数据了，这里也是没有办法才手动设置数量的
dict_num = {'testbed-11jun-ALL': 'Normal', 'testbed-12jun-ALL': 'Normal', 'testbed-13jun-ALL': 'Normal',
            'testbed-14jun-ALL': 'Normal', 'testbed-15jun-ALL': 'Normal', 'testbed-16jun-ALL': 'Normal',
            'testbed-17jun-ALL': 'Normal', 'testbed-13jun-ALL\\maliciousFile': 'Infilt',
            'testbed-14jun-ALL\\maliciousFile': 'HttpDoS',
            'testbed-15jun-ALL\\maliciousFile': 'DDoS', 'testbed-17jun-ALL\\maliciousFile': 'BFSSH'}


# dict_num = {'testbed-14jun-ALL': 'Normal', 'testbed-15jun-ALL': 'Normal', 'testbed-16jun-ALL': 'Normal',
#             'testbed-17jun-ALL': 'Normal',
#             'testbed-14jun-ALL\\maliciousFile': 'HttpDoS',
#             'testbed-15jun-ALL\\maliciousFile': 'DDoS', 'testbed-17jun-ALL\\maliciousFile': 'BFSSH'}

def readFilenameList(path):
    if list == None:
        print('filename list is empty')
    else:
        filelist = os.listdir(path)
        return filelist


def del_end_file_by_time(file_path):
    EAGER_NUM = 10000
    files = os.listdir(file_path)
    if not files:
        return
    else:
        files = sorted(files, key=lambda x: os.path.getmtime(
            os.path.join(file_path, x)))  # 格式解释:对files进行排序.x是files的元素,:后面的是排序的依据.   x只是文件名,所以要带上join.
        file_to_del = files[EAGER_NUM:]
        print(len(file_to_del), file_to_del)
        for i in file_to_del:
            os.remove(os.path.join(file_path, i))


class createBlankFile:
    def __init__(self, originPath, goalPath):
        self.origin_path = originPath
        self.goal_path = goalPath

    def createNewFiles(self):
        # filenameList = ['train\\Normal', 'train\\Infilt', 'train\\HttpDoS', 'train\\DDoS', 'train\\BFSSH',
        #                 'validation\\Normal', 'validation\\Infilt', 'validation\\HttpDoS', 'validation\\DDoS',
        #                 'validation\\BFSSH']
        for dir, class_belong in dict_num.items():
            newPath = self.goal_path + '\\{}'.format(class_belong)
            if not os.path.exists(newPath):
                os.makedirs(newPath)
                print(newPath + " created")
        print('Files have been created')


class createDataSet:
    # EXTRACT_RATIO = 0.2   每个家族抽取样本的比例 ,0.2表示按照8:2划分训练集和验证集

    def __init__(self, origin_path, goal_path):
        self.origin_path = origin_path
        self.goal_path = goal_path

    def checkFileisExist(self):
        try:
            createBlankFile(self.origin_path, self.goal_path).createNewFiles()
            Flag = True
        except:
            print('something wrong in checking File is exist')
            Flag = False
        return Flag

    # def get_floder_name(self, index):
    #     if 0 <= index <= 6:
    #         name = 'Normal'
    #     elif index == 7:
    #         name = 'Infilt'
    #     elif index == 8:
    #         name = 'HttpDoS'
    #     elif index == 9:
    #         name = 'DDoS'
    #     elif index == 10:
    #         name = 'BFSSH'
    #     else:
    #         print('index:', index, ' is error')
    #         raise ValueError
    #     return name

    def moveFile(self):
        list = readFilenameList(self.origin_path)
        # print(list)
        flag = self.checkFileisExist()
        if flag == False: return 'copyFile Failed'
        # index = 0
        for sub_folder, class_belong in dict_num.items():
            print(sub_folder, class_belong)
            class_path = os.path.join(self.origin_path, sub_folder)
            sample_list = glob.glob(class_path + '/*.pcap')
            # sample_train = sample_list[:int(num2pick * 0.8)]
            # sample_validation = sample_list[int(num2pick * 0.8):]
            # print(sample_list)
            for name in sample_list:  # 这里的name带绝对路径
                source_path = self.origin_path + '\\' + sub_folder + '\\' + os.path.basename(name)
                destination_path = self.goal_path + '\\' + class_belong + '\\' + f'{class_belong}_{hash(name)}.pcap'
                print(f'file:{os.path.basename(name)} will be moved from {source_path} to {destination_path}')
                shutil.move(source_path, destination_path)
            # index += 1
        return 'moveFile Succeed!'

        #     file_abs_path = self.origin_path + "\\" + class_list
        #     pcap_list = readFilenameList(file_abs_path)
        #     picknumber = dict_num[malFamily]
        #     sample = random.sample(pathDir, picknumber)
        #     # print(sample)
        #     for name in sample:
        #         # shutil.move(self.origin_path + '\\' + malFamily + '\\' + name,
        #         #             self.goal_path + '\\' + malFamily + '\\' + name)
        #         shutil.copy(self.origin_path + '\\' + malFamily + '\\' + name,
        #                     self.goal_path + '\\' + malFamily + '\\' + name)
        # return 'moveFile Succeed!'


if __name__ == '__main__':
    createDataSet(origin_path=r'K:\数据库\ISCX-IDS-2012\1_Processed2Session\AllLayers_flow',
                  goal_path=r'K:\数据库\ISCX-IDS-2012\2_clusteringPcap').moveFile()
    # sample_list=glob.glob(r'K:\数据库\ISCX-IDS-2012'+'/*.pcap')
    # for i in sample_list:
    #     print(os.path.basename(i))
    # del_end_file_by_time(r'K:\数据库\ISCX-IDS-2012\3_1sampleDataset\validation\Normal')
