'''
论文配套源代码
引用格式：B. Wang, Y. Su, M. Zhang and J. Nie, "A Deep Hierarchical Network for Packet-Level Malicious Traffic Detection,"
in IEEE Access, vol. 8, pp. 201728-201740, 2020, doi: 10.1109/ACCESS.2020.3035967.
'''
import numpy as np
import pickle
import glob
import random
from keras import callbacks
from keras import Input, Model
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from timeit import default_timer as timer
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold

# num_sample = 2000  # number of samples per pkl dataset to train on
CHAR_DIMENSION_PER_PACKET = 256
DIVIDE_RATE = 1/9  # validation//train sample ratio in total sample
PACKET_LEN = 54
EPOCH = 20
MINI_BATCH = 100
CHECKPOINTS_DIR = 'checkpoints/'
MODEL_NAME = ''
STEP_PER_EPOCH = 0
VALIDATION_STEP = 0


def load_data(root_path):
    global STEP_PER_EPOCH, VALIDATION_STEP
    '''
    从当前目录glob所有数据库类型的数据，载入数据
    :return:
    '''
    start_time = timer()
    target_texts = []
    input_texts = []
    label_packets = []
    file_list = glob.glob(root_path + '/ISCX*.pkl')
    for file in file_list:
        with open(file, 'rb') as f:
            label_packetList = pickle.load(f)
            # for label_packet in label_packetList[:min(num_sample, len(label_packetList) - 1)]:
            for label_packet in label_packetList:
                label = label_packet[0]
                packet = label_packet[1]
                # input_text= [byte for byte in input_text]
                # print(type(packet), packet)
                label_packets.append([label, packet[:PACKET_LEN]])
                # target_texts.append(label)
                # input_texts.append(packet[:PACKET_LEN])
                # print(line.split())
    random.shuffle(label_packets)
    for i in label_packets:
        label = i[0]
        packet = i[1]
        target_texts.append(label)
        input_texts.append(packet)
    STEP_PER_EPOCH = int(len(target_texts) / MINI_BATCH * (1 - DIVIDE_RATE))
    VALIDATION_STEP = int(len(target_texts) / MINI_BATCH * DIVIDE_RATE)
    print('label_num:', len(target_texts))
    print('packet_num:', len(input_texts))
    end_time = timer()
    print(f'time cost of loading data:{end_time - start_time} seconds')
    return input_texts, target_texts


def dataset_generator(packets, labels, indices, batch_size):
    '''
    带OHE
    :param packets:
    :param labels:
    :param indices:
    :param batch_size:
    :return:
    '''
    Xbatch = np.zeros((batch_size, PACKET_LEN, CHAR_DIMENSION_PER_PACKET,), dtype=np.int64)
    Ybatch = np.zeros((batch_size, 5), dtype=np.int64)
    batch_idx = 0
    while True:
        for idx in indices:
            for i, bytes in enumerate(packets[idx]):
                # print('bytes:',bytes)
                Xbatch[batch_idx, i, bytes] = 1  # 用OHE
            Ybatch[batch_idx] = np_utils.to_categorical(labels[idx], num_classes=5)
            batch_idx += 1
            if batch_idx == batch_size:
                batch_idx = 0
                # print(Xbatch, Ybatch)
                # print(Xbatch.shape,Ybatch.shape)
                # time.sleep(2)
                yield (Xbatch, Ybatch)


def dataset_generator2(packets, labels, indices, batch_size):
    '''
    不带OHE
    :param packets:
    :param labels:
    :param indices:
    :param batch_size:
    :return:
    '''
    Xbatch = np.zeros((batch_size, PACKET_LEN), dtype=np.int64)
    Ybatch = np.zeros((batch_size, 5), dtype=np.int64)
    batch_idx = 0
    while True:
        for idx in indices:
            for i, byte in enumerate(packets[idx]):
                # print('bytes:',bytes)
                Xbatch[batch_idx, i] = byte  # 不用OHE，保持原数据
            Ybatch[batch_idx] = np_utils.to_categorical(labels[idx], num_classes=5)
            batch_idx += 1
            if batch_idx == batch_size:
                batch_idx = 0
                # print('x:', Xbatch)
                # print('y:', Ybatch)
                # print(Xbatch.shape,Ybatch.shape)
                # time.sleep(2)
                yield (Xbatch, Ybatch)


def get_indices(target_texts):
    # print(target_texts)
    target_texts = np.array(target_texts)  # important ,不然用不了np.where
    normal_indices = np.where(target_texts == 0)[0]
    attack_indices = [np.where(target_texts == i)[0] for i in range(1, 5)]

    test_normal_indices = np.random.choice(normal_indices, int(len(normal_indices) * DIVIDE_RATE))
    test_attack_indices = np.concatenate(
        [np.random.choice(attack_indices[i], int(len(attack_indices[i]) * DIVIDE_RATE)) for i in range(4)])
    # print('tai:', test_attack_indices)
    test_indices = np.concatenate([test_normal_indices, test_attack_indices]).astype(int)
    np.random.shuffle(test_indices)
    train_indices = np.array(list(set(np.arange(len(target_texts))) - set(test_indices)))
    np.random.shuffle(train_indices)
    return train_indices, test_indices


def model_structure():
    global MODEL_NAME
    # from model.hybrid_model import simpleBoBoNet
    # model, model_name = simpleBoBoNet(conv1_filters=32, conv2_filters=64, gru1_units=128, gru2_units=64,
    #                                   kernel_size=3, model_name='simpleBoBo_v3_plus').model()
    # model, model_name = simpleBoBoNet(conv1_filters=16, conv2_filters=32, gru1_units=32, gru2_units=32, dense_units=16,
    #                                   kernel_size=3, model_name='simpleBoBo_v2').model()

    from model.hybrid_model import BoBoNet
    model, model_name = BoBoNet('BoBoNet_ISCX_LSTM').model_LSTM()

    MODEL_NAME = model_name

    # from model.LSTM_model import HwangLSTM
    # model, model_name = HwangLSTM()
    # MODEL_NAME = model_name
    return model


def callbacks_method(K):
    from tools.CallbacksMethod import CustomHistory
    import time
    customhistory_cb = CustomHistory(MODEL_NAME)
    if K != None:
        weight_file = CHECKPOINTS_DIR + str(MODEL_NAME) + '_' + str(PACKET_LEN) + '_' + str(
            time.strftime("%b%d", time.localtime())) + f'_K{K}'+'_{epoch:02d}_{val_loss:.2f}.hdf5'
    else:
        weight_file = CHECKPOINTS_DIR + str(MODEL_NAME) + '_' + str(PACKET_LEN) + '_' + str(
            time.strftime("%b%d", time.localtime())) + '_{epoch:02d}_{val_loss:.2f}.hdf5'
    check_cb = callbacks.ModelCheckpoint(weight_file, monitor='val_loss', verbose=0, save_best_only=False,
                                         save_weights_only=True, mode='min')
    earlystop_cb = callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    csvlogger_cb = callbacks.CSVLogger(filename='log/{}.csv'.format(MODEL_NAME), append=True)
    return check_cb, earlystop_cb, csvlogger_cb, customhistory_cb


def train_model(model, train_data_generator, validation_data_generator,weight_path,K=None):
    model.summary()
    if weight_path:
        model.load_weights(weight_path)
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    # model.compile("RMSProp", "categorical_crossentropy", metrics=["accuracy"])
    check_cb, earlystop_cb, csvlogger_cb, customhistory_cb = callbacks_method(K)
    model.fit_generator(generator=train_data_generator,
                        steps_per_epoch=STEP_PER_EPOCH,
                        epochs=EPOCH,
                        callbacks=[check_cb, earlystop_cb, csvlogger_cb, customhistory_cb],
                        validation_data=validation_data_generator,
                        validation_steps=VALIDATION_STEP)


def run(dataset_path,weight_path=None):
    # input_texts, target_texts = load_data(dataset_path)
    # train_indices, test_indices = get_indices(target_texts)
    # model = model_structure()
    # train_data_generator = dataset_generator2(input_texts, target_texts, train_indices, MINI_BATCH)
    # validation_data_generator = dataset_generator2(input_texts, target_texts, test_indices, MINI_BATCH)
    # train_model(model, train_data_generator, validation_data_generator,weight_path)
    input_texts, target_texts = load_data(dataset_path)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    K_start = 0
    for train_indices, test_indices in kfold.split(input_texts, target_texts):
        model = model_structure()
        train_data_generator = dataset_generator2(input_texts, target_texts, train_indices, MINI_BATCH)
        validation_data_generator = dataset_generator2(input_texts, target_texts, test_indices, MINI_BATCH)
        train_model(model, train_data_generator, validation_data_generator,weight_path, K_start)
        K_start += 1


if __name__ == '__main__':
    run(r'K:\dataset\ISCX-IDS-2012\3_1sampleDataset\train')
    # input_texts, target_texts = load_data(r'K:\数据库\ISCX-IDS-2012\3_1sampleDataset\Dataset')
    # # print('labels_len:',len(target_texts))
    # train_indices, test_indices = get_indices(target_texts)
    # # # for idx in test_indices:
    # #     print(idx,target_texts[idx])
    # # print(np_utils.to_categorical(4,num_classes=5))
    # train_data_generator = dataset_generator(input_texts, target_texts, train_indices, MINI_BATCH)
