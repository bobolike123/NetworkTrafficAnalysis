# 拷贝数据集中每个类别的指定数量的数据至新的位置，并将这些数据矢量化
import shutil
import os
import random

# 因为原始数据太多了，这里按照分布数量取原数据集大约0.1比例的数据
# 因为Normal文件夹下本来大概有300w数据的，因为生成慢，只有90w，这里按照300w数据处理
# 如果Normal数据全的话，设置EXTRA_RATIO=0.1就可以取0.1比例数据了，这里也是没有办法才手动设置数量的
dict_num = {'testbed-11jun-ALL': 20000, 'testbed-12jun-ALL': 5000, 'testbed-13jun-ALL': 5000,
            'testbed-14jun-ALL': 5000, 'testbed-15jun-ALL': 5000, 'testbed-16jun-ALL': 5000,
            'testbed-17jun-ALL': 5000, 'testbed-13jun-ALL\\maliciousFile': 10000,
            'testbed-14jun-ALL\\maliciousFile': 3000,
            'testbed-15jun-ALL\\maliciousFile': 20000, 'testbed-17jun-ALL\\maliciousFile': 4500}


def readFilenameList(path):
    if list == None:
        print('filename list is empty')
    else:
        filelist = os.listdir(path)
        return filelist

def del_end_file_by_time(file_path):
    EAGER_NUM=10000

    files = os.listdir(file_path)
    if not files:
        return
    else:
        files = sorted(files, key=lambda x: os.path.getmtime(
            os.path.join(file_path, x)))  # 格式解释:对files进行排序.x是files的元素,:后面的是排序的依据.   x只是文件名,所以要带上join.
        file_to_del=files[EAGER_NUM:]
        print(len(file_to_del),file_to_del)
        for i in file_to_del:
            os.remove(os.path.join(file_path,i))
class createBlankFile:
    def __init__(self, originPath, goalPath):
        self.origin_path = originPath
        self.goal_path = goalPath

    def createNewFiles(self):
        filenameList = ['train\\Normal', 'train\\Infilt', 'train\\HttpDoS', 'train\\DDoS', 'train\\BFSSH',
                        'validation\\Normal', 'validation\\Infilt', 'validation\\HttpDoS', 'validation\\DDoS',
                        'validation\\BFSSH']
        for fn in filenameList:
            newPath = self.goal_path + '\\{}'.format(fn)
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

    def get_floder_name(self, index):
        if 0 <= index <= 6:
            name = 'Normal'
        elif index == 7:
            name = 'Infilt'
        elif index == 8:
            name = 'HttpDoS'
        elif index == 9:
            name = 'DDoS'
        elif index == 10:
            name = 'BFSSH'
        else:
            print('index:', index, ' is error')
            raise ValueError
        return name

    def copyFile(self):
        list = readFilenameList(self.origin_path)
        # print(list)
        flag = self.checkFileisExist()
        if flag == False: return 'copyFile Failed'
        index = 0
        for sub_folder, num2pick in dict_num.items():
            print(sub_folder, num2pick)
            class_path = os.path.join(self.origin_path, sub_folder)
            sample_list = random.sample(os.listdir(class_path), num2pick)
            sample_train = sample_list[:int(num2pick * 0.8)]
            sample_validation = sample_list[int(num2pick * 0.8):]

            goal_sub_path = self.get_floder_name(index)
            for name in sample_train:
                shutil.copy(self.origin_path + '\\' + sub_folder + '\\' + name,
                            self.goal_path + '\\train' + '\\' + goal_sub_path + '\\' + f'{goal_sub_path}_{hash(name)}.pcap')
            for name in sample_validation:
                shutil.copy(self.origin_path + '\\' + sub_folder + '\\' + name,
                            self.goal_path + '\\validation' + '\\' + goal_sub_path + '\\' + f'{goal_sub_path}_{hash(name)}.pcap')
            index += 1
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
                  goal_path=r'K:\数据库\ISCX-IDS-2012\3_1sampleDataset').copyFile()
    # print(dict_num.items()[7:])
    # del_end_file_by_time(r'K:\数据库\ISCX-IDS-2012\3_1sampleDataset\validation\Normal')
