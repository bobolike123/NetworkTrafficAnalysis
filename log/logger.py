import logging
import os
import time
import linecache
import random
import datetime
startwords='#this file is used to record run time ,initial count:0'
value = 0

'''
To create log after training
'''

def readCount(filename='count.log'):

    if not os.path.exists(filename):
        with open(filename,mode='w+') as f:
            print('创建{}成功'.format(filename))
            f.truncate()
            f.write(startwords+'\n')  #注意实际写入\n\r两个字节
            global value
            value = 0
    else:
        linecache.checkcache(filename)
        str_list=linecache.getlines(filename)
        # print(str_list)

        with open(filename, mode='rb+') as f:
            # print('打开文件成功')
            if str_list == []:
                f.truncate()  # 清空文件
                f.write(bytes(startwords + '\n', encoding='UTF-8'))  # 注意实际写入\n\r两个字节
                value = 0
            else:
                if not str(str_list[0]).startswith('#'):
                    f.truncate()  # 清空文件
                    f.write(bytes(startwords + '\n',encoding='UTF-8'))  # 注意实际写入\n\r两个字节
                    value = 0
                else:
                    if len(str_list) >= 2:
                        if str_list[-1] == '\n':  #如果最后一行是空行，则删除之
                            f.seek(-2,os.SEEK_END)
                            f.truncate()
                            f.close() # 感觉可有可无
                            print('最后一行是空行，则删除之')
                            if str_list[-2].startswith('#'): value = 0
                            if str_list[-2].startswith('>'):
                                c_str=str_list[-2].split('=')[1][:-1] #剔除掉末尾换行符
                                print('选取倒数第二行过滤出count')
                                value = int(c_str)
                        if str_list[-1].startswith('>'):
                            c_str = str_list[-1].split('=')[1][:-1]  # 剔除掉末尾换行符
                            # print('value:',c_str)
                            value = int(c_str)
                    if len(str_list) == 1: value =  0  # 说明只有startwords一行，没有数据
    return value

'''
运行一次后
'''

def writeCount(tp,filename='count.log'):
    '''
    :param tunple:  tunple = (time_cost,losses,val_losses,acc,val_acc)
    :param filename:  the file you want to save logger ,default:count.log
    :return:
    '''
    currentCount=tp[1]+1
    # print('writeCount_current:',currentCount)

    logging.basicConfig(filename=filename,filemode='a+',
                        format='>%(asctime)s  %(message)s',level=logging.INFO)
    logging.info("time_cost:{0[0]}s   losses:{0[2]},val_losses:{0[3]},acc:{0[4]},val_acc:{0[5]} ,current run times={1}".format(tp,currentCount))
    logging.shutdown()

def writeData(tp,filename2):
    logging.basicConfig(filename=filename2, filemode='a+',
                        format='>%(asctime)s\n  %(message)s', level=logging.INFO)
    logging.info("losses:{0[0]}\n val_losses:{0[1]}\n acc:{0[2]}\n val_acc:{0[3]}".format(tp))
    logging.shutdown()

if __name__ == '__main__':
    for i in range(5):
        starttime=datetime.datetime.now()
        # time_cost = random.random()*3600 # 仅测试用
        count=readCount('VGG_count.log')
        time.sleep(3)   #用sleep()代替程序主体运行时间
        endtime=datetime.datetime.now()
        time_cost=(endtime-starttime).seconds

        list = []
        result_tunple=(0.22,0.43,95.22,97.21)
        list.append(time_cost)
        list.append(count)
        tp = tuple(list) + result_tunple  # tp = (time_cost,oldcount,losses,val_losses,acc,val_acc)
        writeCount(tp,'VGG_count.log')
