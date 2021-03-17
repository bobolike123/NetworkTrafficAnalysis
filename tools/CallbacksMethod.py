from keras.callbacks import Callback
from timeit import default_timer as timer
# import pickle
import time


class CustomHistory(Callback):
    def __init__(self, model_name):
        # self.epochs = epochs
        self.model_name = model_name
        self.start_time = None
        self.end_time = None

    def on_train_begin(self, logs=None):
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        outputstr = f'------------训练开始时间为：{current_time} ---------------\n'
        with open(f'./evaluation/{self.model_name}_costTime.txt', 'a')as f:
            f.write(outputstr)
            f.close()

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = timer()

    def on_epoch_end(self, epoch, logs=None):
        self.end_time = timer()
        time_cost_epoch = self.end_time - self.start_time
        outputstr = f'第{epoch}轮的训练耗时为{time_cost_epoch}秒\n'
        with open(f'./evaluation/{self.model_name}_costTime.txt', 'a')as f0:
            f0.write(outputstr)
            f0.close()
        # with open('./callbacks_log.pkl', 'ab') as f:
        #     pickle.dump(logs, f)
        #     f.close()

    def on_train_end(self, logs=None):
        pass
