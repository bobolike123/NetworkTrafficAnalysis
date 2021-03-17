import os
import glob
def del_end_file_by_time(root_path):
    EAGER_NUM=0
    files=glob.glob(root_path+'/*/*.*')
    print(files)
    # files = os.listdir(file_path)
    # if not files:
    #     return
    # else:
    #     files = sorted(files, key=lambda x: os.path.getmtime(x))
    #     # 格式解释:对files进行排序.x是files的元素,:后面的是排序的依据.   x只是文件名,所以要带上join.
    #     file_to_del=files[EAGER_NUM:]
    #     print(len(file_to_del),file_to_del)
    #     for i in file_to_del:
    #         os.remove(i)

if __name__ == '__main__':
    del_end_file_by_time(r'K:\数据库\ISCX-IDS-2012\2_1_extractFlowFeature')