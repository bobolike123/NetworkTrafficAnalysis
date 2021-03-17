import numpy as np
import pickle
import glob
import random
import matplotlib.pyplot as plt
from timeit import default_timer as timer

MINI_BATCH = 32
PACKET_LEN = 54
# plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
'''
    真实值是positive，模型认为是positive的数量（True Positive=TP）
    真实值是positive，模型认为是negative的数量（False Negative=FN）：这就是统计学上的第二类错误（Type II Error）
    真实值是negative，模型认为是positive的数量（False Positive=FP）：这就是统计学上的第一类错误（Type I Error）
    真实值是negative，模型认为是negative的数量（True Negative=TN）

将这四个指标一起呈现在表格中，就能得到如下这样一个矩阵，我们称它为混淆矩阵（Confusion Matrix）：
               真实值
             pos        neg
          -----------------
 预   pos    TP    |   FP
 测        --------|--------
 值   neg    FN    |   TN
'''


def update_confusion_matrix(confusion_matrix, actual_lb, predict_lb):
    for idx, value in enumerate(actual_lb):
        p_value = predict_lb[idx]
        confusion_matrix[value, p_value] += 1
    return confusion_matrix


def load_data_p(test_dir_path):
    '''
    仅供predict时load data用
    :return:
    '''
    start_time = timer()
    target_texts = []
    input_texts = []
    label_packets = []
    # 载入自己生成的test set pkl文件
    file_list = glob.glob(test_dir_path + '/*.pkl')
    for file in file_list:
        with open(file, 'rb') as f:
            label_packetList = pickle.load(f)
            for label_packet in label_packetList:
                label = label_packet[0]
                packet = label_packet[1]
                # target_texts.append(label)
                # input_texts.append(packet[:PACKET_LEN])
                label_packets.append([label, packet[:PACKET_LEN]])
    random.shuffle(label_packets)
    for i in label_packets:
        label = i[0]
        packet = i[1]
        target_texts.append(label)
        input_texts.append(packet)

    print('label_num:', len(target_texts))
    print('packet_num:', len(input_texts))
    end_time = timer()
    print(f'time cost of loading data:{end_time - start_time} seconds')
    return input_texts, target_texts


def model_structure(name_of_model):
    global MODEL_NAME
    # from model.hybrid_model import simpleBoBoNet
    # model, model_name = simpleBoBoNet(conv1_filters=32, conv2_filters=64, gru1_units=128, gru2_units=64,
    #                                   kernel_size=3, model_name='simpleBoBo_v3_plus').model()
    # model, model_name = simpleBoBoNet(conv1_filters=16, conv2_filters=32, gru1_units=32, gru2_units=32, dense_units=16,
    #                                   kernel_size=3, model_name='simpleBoBo_v2').model()
    if name_of_model == 'BOBO_USTC':
        from model.hybrid_model import BoBoNet
        model, model_name = BoBoNet(model_name='BOBO_USTC').model_USTC()
        MODEL_NAME = model_name
        return model

    if name_of_model == 'BOBO_ISCX':
        from model.hybrid_model import BoBoNet
        model, model_name = BoBoNet(model_name='BOBO_ISCX').model()
        MODEL_NAME = model_name
        return model

    if name_of_model == 'BOBO_ISCX_LSTM':
        from model.hybrid_model import BoBoNet
        model, model_name = BoBoNet(model_name='BOBO_ISCX_LSTM').model_LSTM()
        MODEL_NAME = model_name
        return model

    if name_of_model == 'BOBO_CICIDS' or name_of_model == 'BOBO_CICIDS(RS)':
        from model.hybrid_model import BoBoNet
        model, model_name = BoBoNet(model_name=name_of_model).model_CICIDS()
        MODEL_NAME = model_name
        return model

    raise ValueError(f"model name :{name_of_model} is invalid")


def get_result_cm(confuse_matrix, classnum):
    '''

    :param confuse_matrix:
    :return:
    '''
    total_element_sum = sum(map(sum, confuse_matrix))
    ma_len = len(confuse_matrix)
    '''
    以下说明及注释部分的代码仅适用于ISCX2012数据集
    虽然共有5种流量，但是如果把流量看成两类，即正常流量和恶意流量，则可以用二分类方法得到
    overall_accuracy,overall_tpr,overall_fpr
    其中tpr= TP/(TP+FN)  fpr= FP/(FP+TN) ，这里的TP是恶意流量被正确识别的数量，那么TN就是正常流量被正确识别的数量，
    FN是恶意流量被错误识别为正常流量的数量，FP是正常流量被错误识别为恶意流量的数量
    '''

    overall_TP = confuse_matrix[1:, 1:].sum()
    overall_FN = confuse_matrix[0].sum() - confuse_matrix[0, 0].sum()
    overall_TN = confuse_matrix[0, 0].sum()
    overall_FP = confuse_matrix[:, 0].sum() - confuse_matrix[0, 0].sum()

    print(f'overall_TP={overall_TP}, overall_FN={overall_FN}, overall_TN={overall_TN}, overall_FP={overall_FP}')

    overall_DR = overall_TP / (overall_TP + overall_FN)
    overall_accuracy = (overall_TP + overall_TN) / (overall_FN + overall_TN + overall_FP + overall_TP)
    overall_FAR = overall_FP / (overall_FP + overall_TN)
    overall2write = "overall_DR=%f,overall_accuracy=%f,overall_FAR=%f" % (overall_DR, overall_accuracy, overall_FAR)
    print(overall2write)

    TN_list, TP_list, FP_list, FN_list = complement_minor(confuse_matrix, classnum)
    result2write = []
    for i in range(ma_len):
        category = LABEL_CLASS[i]
        TP = TP_list[i]
        TN = TN_list[i]
        FP = FP_list[i]
        FN = FN_list[i]
        precision = TP / (TP + FP)
        DetectionRate = TP / (TP + FN)  # 实际上为recall
        accuracy = (TP + TN) / (TP + FP + TN + FN)
        Falsealarmrate = FP / (FP + TN)  # FAR
        F1_score = 2 * precision * DetectionRate / (precision + DetectionRate)
        result = "category \"%s\" result:TP=%d,FP=%d,TN=%d,FN=%d,precision=%f,DetectionRate=%f,accuracy=%f,FAR=%f,F1-score=%f \n" % (
            category, TP, FP, TN, FN, precision, DetectionRate, accuracy, Falsealarmrate, F1_score)
        print(result)
        result2write.append(result)
    with open('evaluation/{}_evaluation_result.txt'.format(MODEL_NAME), 'w') as f:
        f.writelines(result2write)
        f.write(overall2write)


def complement_minor(matrix, classnum):
    '''
    解析混淆矩阵，返回TN,TP，FP,FN四元列表
    :param matrix:
    :return:
    '''
    TN_list = []
    TP_list = []
    FP_list = []
    FN_list = []
    for i in range(classnum):
        TN = matrix[:i, :].sum() + matrix[i + 1:, :].sum() - matrix[:, i].sum() + matrix[i, i].sum()
        TP = matrix[i, i]
        FP = matrix[i].sum() - matrix[i, i].sum()
        FN = matrix[:, i].sum() - matrix[i, i].sum()
        TN_list.append(TN)
        TP_list.append(TP)
        FP_list.append(FP)
        FN_list.append(FN)
        print(f'{LABEL_CLASS[i]}的TN数为{TN}，TP数为{TP},FP数为{FP},FN数为{FN}')
    return TN_list, TP_list, FP_list, FN_list


def get_result_binary(y_true, y_pred):
    TP = sum(y_true * y_pred == 1)
    FP = sum((y_pred - y_true) == 1)
    TN = sum((y_true + y_pred) == 0)
    FN = sum((y_true - y_pred) == 1)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    Falsealarmrate = FP / (FP + TN)
    F1_score = 2 * precision * recall / (precision + recall)
    print("TP=%d,FP=%d,TN=%d,FN=%d,precision=%f,recall=%f,accuracy=%f,FAR=%f,F1-score=%f" % (
        TP, FP, TN, FN, precision, recall, accuracy, Falsealarmrate, F1_score))


def draw_ROC(label, pred_raw):
    '''
    因为ROC曲线仅用于二分类任务，而我们传入的label有五种（从0-4）所以还要对label和pred_raw进行一个预处理
    思想是，把label==0的标为Neg类，把label==1,2,3,4的标为Pos类
    注意，pred_raw是一个5维向量，比如[0.97,0.01,0.01,0.01,0.00]用argmax()后输出0（向量最大值元素的索引）
    因此我们要把5维向量变成2维向量（合并索引位置为1,2,3,4的值），如上面的向量应变成[0.97,0.03]

    考虑到pred_raw.shape为n*5 因此我们设计一个5*2的矩阵，使得[]n*5  *  []5*2  -> []n*2
    经过设计，这个5*2矩阵，我们称之为P=[[1,0],[0,1],[0,1],[0,1],[0,1]]可以实现此功能
    :param label:
    :param pred_label:
    :return:
    '''
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    P = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [0, 1]])  # 用于实现把五维向量变成二维向量
    pred_binary = np.matmul(pred_raw, P)
    # print('pred_binary:', pred_binary)
    y_true = np.where(label > 0, 1, 0)
    y_pred = [np.take(y, 1) for y in pred_binary]  # 返回一个list.   np.take(y, 1)表示取y中索引为1的值
    print('y_pred:', y_pred)
    # y_pred = np.where(pred_binary > 0, 1, 0)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    auc = auc(fpr, tpr)
    print("AUC : ", auc)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='S3< val (AUC = {:.3f})'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(f'./evaluation/{MODEL_NAME}_ROC.jpg')
    # plt.show()


def draw_CM(cm, labels_name, title):
    # from sklearn.metrics import confusion_matrix  # 生成混淆矩阵函数
    import matplotlib.pyplot as plt  # 绘图库
    # with open(f'evaluation/{MODEL_NAME}-confuse_matrix.pkl') as f:
    #     cm=pickle.load(f)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像
    plt.title(title)  # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./evaluation/Confusion_Matrix.png', format='png')
    # plt.show()


def draw_CM_2(cm, labels_name, title, isChinese=False, title_size=20, label_size=15, isManyClasses=False):
    '''
    第二种风格的CM图
    :return:
    '''

    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig = plt.figure(figsize=(12, 8), dpi=120)
    ind_array = np.arange(len(labels_name))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.001:
            if y_val == x_val:
                plt.text(x_val, y_val, "%0.3f" % (c,), color='white', fontsize=7 if isManyClasses else 12, va='center',
                         ha='center')
            else:
                plt.text(x_val, y_val, "%0.3f" % (c,), color='black', fontsize=7 if isManyClasses else 12, va='center',
                         ha='center')
    # offset the tick
    tick_marks = np.array(range(len(labels_name))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    # plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.binary)
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.OrRd)
    plt.title(title, fontdict={'weight': 'normal', 'size': title_size})
    plt.colorbar()
    xlocations = np.array(range(len(labels_name)))
    plt.xticks(xlocations, labels_name, fontproperties='Times New Roman', rotation=45, ha='right')
    plt.yticks(xlocations, labels_name, fontproperties='Times New Roman')
    plt.tick_params(labelsize=15)

    if isChinese:
        plt.ylabel('真实标签', fontdict={'family': 'SimSun', 'weight': 'normal', 'size': label_size})
        plt.xlabel('预测标签', fontdict={'family': 'SimSun', 'weight': 'normal', 'size': label_size})
    else:
        plt.ylabel('True label', fontdict={'weight': 'normal', 'size': label_size})
        plt.xlabel('Predicted label', fontdict={'weight': 'normal', 'size': label_size})
    # show confusion matrix
    if len(LABEL_CLASS) == 5:
        dataset_name = 'ISCX2012'
    elif len(LABEL_CLASS) == 20:
        dataset_name = 'USTC-TFC2016'
    elif len(LABEL_CLASS) == 6:
        dataset_name = 'CICIDS2017'
    # fig.autofmt_xdate()  # x轴斜着打印
    plt.savefig(f'./evaluation/{MODEL_NAME}_Confusion_Matrix_style2_{dataset_name}.png', format='png', dpi=600)
    plt.show()


def choose_label_name(model_name):
    global LABEL_NAME, LABEL_CLASS, CLASS_NUM
    if model_name == 'BOBO_USTC':
        LABEL_CLASS = {0: 'BitTorrent', 1: 'Facetime', 2: 'FTP', 3: 'Gmail', 4: 'MySQL', 5: 'Outlook', 6: 'Skype',
                       7: 'SMB',
                       8: 'Weibo', 9: 'WorldOfWarcraft', 10: 'Cridex', 11: 'Geodo', 12: 'Htbot', 13: 'Miuref',
                       14: 'Neris',
                       15: 'Nsis-ay', 16: 'Shifu', 17: 'Tinba', 18: 'Virut', 19: 'Zeus'}
        LABEL_NAME = ['BitTorrent', 'Facetime', 'FTP', 'Gmail', 'MySQL', 'Outlook', 'Skype', 'SMB', 'Weibo', 'WOW',
                      'Cridex', 'Geodo', 'Htbot', 'Miuref', 'Neris', 'Nsis-ay', 'Shifu', 'Tinba', 'Virut', 'Zeus']
        dataset = 'USTC-TFC2016'
        CLASS_NUM = len(LABEL_CLASS)
        return dataset

    if model_name == 'BOBO_ISCX' or model_name == 'BOBO_ISCX_LSTM':
        LABEL_CLASS = {0: 'Normal', 1: 'BFSSH', 2: 'Infilt', 3: 'HttpDoS', 4: 'DDoS'}
        LABEL_NAME = ['Normal', 'BFSSH', 'Infiltrating', 'HttpDoS', 'DDoS']
        dataset = 'ISCX2012'
        CLASS_NUM = len(LABEL_CLASS)
        return dataset

    if model_name == 'BOBO_CICIDS' or model_name == 'BOBO_CICIDS(RS)':
        # LABEL_CLASS = {0: 'Normal', 1: 'BruteForce', 2: 'DoS_DDoS', 3: 'WebAttack', 4: 'Infiltration', 5: 'BotNet',
        #                6: 'PortScan'}
        # LABEL_NAME= ['Normal','BruteForce','DoS','WebAttack','Infiltration','BotNet','PortScan']
        LABEL_CLASS = {0: 'Normal', 1: 'BruteForce', 2: 'WebAttack', 3: 'Infiltration', 4: 'BotNet',
                       5: 'PortScan'}
        LABEL_NAME = ['Normal', 'BruteForce', 'WebAttack', 'Infiltration', 'BotNet', 'PortScan']
        dataset = 'CICIDS2017'
        CLASS_NUM = len(LABEL_CLASS)
        return dataset

    raise ValueError(f"model_name:{model_name} is not supported")
    # assert CLASS_NUM > 0


def model_predictor(weight_path, test_data_path, model_name, isChinese=False):
    '''
    主函数
    :param weight_path:
    :param test_data_path:
    :param model_name:
    :param isChinese: 是否以中文语言输出图片，默认为否（英文输出）
    :return:
    '''
    pred_start_time = timer()
    model = model_structure(model_name)
    dataset_name = choose_label_name(model_name)  # 根据模型选择数据标签字典
    print(dataset_name)
    model.load_weights(weight_path)
    data, label = load_data_p(test_data_path)

    total_test_num = len(label)
    print('top 10 data:', data[:10])
    data_packet_format = []
    data_all_format = []
    for i, packet in enumerate(data):
        for j, byte in enumerate(packet):
            data_packet_format.append(byte)
        data_all_format.append(data_packet_format)
        data_packet_format = []
    data_all_format = np.array(data_all_format)
    # print(data_all_format)
    pred_raw = model.predict(x=data_all_format, batch_size=MINI_BATCH)  # 未经过处理的pred数据
    print('shape:', pred_raw.shape)
    pred_label = np.argmax(pred_raw, axis=1)  # 经过处理的pred数据，取pred_raw中值最大的元素的索引为label值
    pred_end_time = timer()
    pred_efficiency = (pred_end_time - pred_start_time) / total_test_num * 10000
    print(f'预测时间为{pred_efficiency} seconds /10000 samples')

    label = np.array(label, dtype=np.int8)
    # print(CLASS_NUM)
    init_matrix = np.zeros((CLASS_NUM, CLASS_NUM), dtype=int)
    confuse_matrix = update_confusion_matrix(init_matrix, label, pred_label)  # 生成混淆矩阵
    # draw_ROC(label, pred_raw)  # 画ROC曲线，用pred_raw,否则用pred_label就是一条非常完美的曲线
    # draw_CM(confuse_matrix, LABEL_NAME, 'Confusion Matrix of Test Dataset')
    if isChinese:
        manyclasses = True if dataset_name == 'USTC-TFC2016' else False
        # draw_CM_2(confuse_matrix, LABEL_NAME, f'{dataset_name}测试集上归一化的混淆矩阵', isChinese=True,
        #           isManyClasses=manyclasses)
        draw_CM_2(confuse_matrix, LABEL_NAME, '', isChinese=True,
                  isManyClasses=manyclasses)
    else:
        draw_CM_2(confuse_matrix, LABEL_NAME, f'Normalized Confusion Matrix of {dataset_name} Test Dataset')
    print(confuse_matrix)
    with open(f'./evaluation/{MODEL_NAME}-confuse_matrix.pkl', 'wb') as f:
        pickle.dump(confuse_matrix, f)
        f.close()

    get_result_cm(confuse_matrix, classnum=CLASS_NUM)

    # 保存预测值和真实值
    with open(f'evaluation/{MODEL_NAME}_pred_raw.pkl', 'wb') as f:
        pickle.dump(pred_raw, f)
        print('pred_raw保存成功')
        f.close()
    with open(f'evaluation/{MODEL_NAME}_labels.pkl', 'wb') as f:
        pickle.dump(label, f)
        print('label保存成功')
        f.close()
    '''
    class_label = {'Normal': 0, 'BFSSH': 1, 'Infilt': 2, 'HttpDoS': 3, 'DDoS': 4}
    confuse_matrix:
    [[ 598    0    0    0    0]
    [   0  118    0    0    0]
    [ 125    0   91    3   11]
    [   0    0    0  237    0]
    [   0    0    0    0 1817]]
    '''


if __name__ == '__main__':
    # with open('./confuse_matrix.pkl', 'rb') as f:
    #     cf_ma = pickle.load(f)
    # get_result_cm(cf_ma)
    # A = np.array([[0.97, 0.01, 0.01, 0.01, 0.00], [0.03, 0.65, 0.21, 0.02, 0.09]])
    # P = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [0, 1]])
    #
    # AP=np.matmul(A,P)
    # print(AP)
    # y_2 = np.array(
    #     [[2.24347630e-07, 9.99999833e-01], [2.90791888e-07, 9.99999747e-01], [7.40729647e-08, 9.99999857e-01]])
    # y = [np.take(i, 1) for i in y_2]
    # print(y, type(y))
    # model_predictor(weight_path='./checkpoints/(Best)BoBoNet_USTC_54_Jun16_20_0.00.hdf5',
    #                 test_data_path=r'K:\dataset\USTC-TFC2016\2_dataset\test', model_name='BOBO_USTC', isChinese=True)
    # model_predictor(weight_path='checkpoints/(Best)BoBoNet_ISCX_LSTM_54_Jun18_05_0.00.hdf5',
    #                 test_data_path=r'K:\dataset\ISCX-IDS-2012\3_1sampleDataset\test',model_name='BOBO_ISCX_LSTM')
    # model_predictor(weight_path='checkpoints/(Best)BoBoNet_ISCX_54_Jun16_13_0.00.hdf5',
    #                 test_data_path=r'K:\dataset\ISCX-IDS-2012\3_1sampleDataset\test', model_name='BOBO_ISCX',isChinese=True)
    # model_predictor(weight_path='checkpoints/Net_CICIDS_54_Oct12_K5_10_0.00.hdf5',
    #                 test_data_path=r'K:\dataset\CIC-IDS-2017\4_Dataset\test', model_name='BOBO_CICIDS',isChinese=True)
    model_predictor(weight_path='checkpoints/BoBoNet_CICIDS(RS)_54_Oct12_K5_13_0.00.hdf5',
                    test_data_path=r'K:\dataset\CIC-IDS-2017\4_Dataset\unbalanced\test', model_name='BOBO_CICIDS(RS)',
                    isChinese=True)
