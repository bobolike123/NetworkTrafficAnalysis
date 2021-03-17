from sklearn.datasets import load_digits
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
import pickle
import matplotlib as mpl
import numpy as np
from cycler import cycler


def run():
    # digits = load_digits()
    # print(digits.target)
    NAME_COLOR = ['Normal:blue', 'BFSSH:orange', 'Infilt:yellow', 'HttpDoS:green', 'DDoS:brown']

    with open('evaluation/BOBO_ISCX_pred_raw.pkl', 'rb') as f:
        X = pickle.load(f)
    with open('evaluation/BOBO_ISCX_labels.pkl', 'rb') as f2:
        labels = pickle.load(f2)

    print(f'载入{len(labels)}组测试数据')
    embeddings = TSNE(n_jobs=4).fit_transform(X)
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    fig, ax = plt.subplots()
    # labels = digits.target

    cate_x = []
    cate_y = []
    for i in range(len(NAME_COLOR)):
        cate_index = np.where(labels == i)[0]  # 寻找元素值为i对应的列表索引号，返回索引list
        cate_i_x = [vis_x[j] for j in cate_index]
        cate_i_y = [vis_y[j] for j in cate_index]
        cate_x.append(cate_i_x)
        cate_y.append(cate_i_y)

    # for index,color in enumerate(['A:blue','B:orange','C:yellow','D:green','E:brown','F:coral','G:darkgray','H:deeppink','I:deepskyblue','J:firebrick']):
    #     ax.scatter(cate_x[index], cate_y[index], c=color.split(':')[1],  marker='.',label=color.split(':')[0])
    for index, color in enumerate(NAME_COLOR):  # 在子图上分别画图，每次画一种颜色的点
        ax.scatter(cate_x[index], cate_y[index], c=color.split(':')[1], marker='.', label=color.split(':')[0])
    # LABEL_CLASS = {0: 'Normal', 1: 'BFSSH', 2: 'Infilt', 3: 'HttpDoS', 4: 'DDoS'}
    # LABEL_NAME = ['Normal', 'BFSSH', 'Infiltrating', 'HttpDoS', 'DDoS']
    # plt.scatter(vis_x, vis_y, c=labels, cmap=plt.cm.get_cmap("jet", 5), marker='.',label=['a','b','c','d','e'])
    # print(vis_x)
    # print(vis_y)
    # plt.scatter(vis_x, vis_y, c=digits.target, cmap=plt.cm.get_cmap("jet", 10), marker='.',label='a')
    # plt.colorbar(ticks=range(5))
    # plt.clim(-0.5, 9.5)
    # plt.show()
    ax.legend(loc='lower right')
    plt.savefig('./tsne_result.png', format='png', dpi=400)
    plt.show()


def run_USTC():
    '''
    12000组测试数据
    :return:
    '''
    # CLASS_LABEL = {'BitTorrent': 0, 'Facetime': 1, 'FTP': 2, 'Gmail': 3, 'MySQL': 4, 'Outlook': 5, 'Skype': 6, 'SMB-1': 7,
    #                'SMB-2': 7, 'Weibo-1': 8, 'Weibo-2': 8, 'Weibo-3': 8, 'Weibo-4': 8, 'WorldOfWarcraft': 9, 'Cridex': 10,
    #                'Geodo': 11, 'Htbot': 12, 'Miuref': 13, 'Neris': 14, 'Nsis-ay': 15, 'Shifu': 16, 'Tinba': 17,
    #                'Virut': 18, 'Zeus': 19}
    NAME_COLOR = ['BitTorrent:aliceblue', 'Facetime:aqua', 'FTP:aquamarine', 'Gmail:greenyellow', 'MySQL:lightgreen',
                  'Outlook:black', 'Skype:blue', 'SMB:brown', 'Weibo:burlywood', 'WOW:cadetblue',
                  'Cridex:chartreuse', 'Geodo:chocolate', 'Htbot:coral', 'Miuref:cornflowerblue', 'Neris:cornsilk',
                  'Nsis-ay:darkblue','Shifu:darkgray','Tinba:red','Virut:yellow','Zeus:deeppink']
    with open('evaluation/BOBO_USTC_pred_raw.pkl', 'rb') as f:
        X = pickle.load(f)
    with open('evaluation/BOBO_USTC_labels.pkl', 'rb') as f2:
        labels = pickle.load(f2)

    print(f'载入{len(labels)}组测试数据')
    embeddings = TSNE(n_jobs=4).fit_transform(X)
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    fig, ax = plt.subplots()
    cate_x = []
    cate_y = []
    for i in range(len(NAME_COLOR)):
        cate_index = np.where(labels == i)[0]  # 寻找元素值为i对应的列表索引号，返回索引list
        cate_i_x = [vis_x[j] for j in cate_index]
        cate_i_y = [vis_y[j] for j in cate_index]
        cate_x.append(cate_i_x)
        cate_y.append(cate_i_y)

    # for index,color in enumerate(['A:blue','B:orange','C:yellow','D:green','E:brown','F:coral','G:darkgray','H:deeppink','I:deepskyblue','J:firebrick']):
    #     ax.scatter(cate_x[index], cate_y[index], c=color.split(':')[1],  marker='.',label=color.split(':')[0])
    for index, color in enumerate(NAME_COLOR):  # 在子图上分别画图，每次画一种颜色的点
        ax.scatter(cate_x[index], cate_y[index], c=color.split(':')[1], marker='.', label=color.split(':')[0])
    ax.legend(loc='lower right',fontsize='x-small')
    plt.savefig('./tsne_result_USTC.png', format='png', dpi=400)
    plt.show()

def run_USTC_binary():
    # classname=['BitTorrent', 'Facetime', 'FTP', 'Gmail', 'MySQL', 'Outlook', 'Skype', 'SMB', 'Weibo', 'WOW',
    #                   'Cridex', 'Geodo', 'Htbot', 'Miuref', 'Neris', 'Nsis-ay', 'Shifu', 'Tinba', 'Virut', 'Zeus']
    classes=['benign','malicious']
    colors=['green','red']
    # mpl.rcParams['axes.prop_cycle'] = cycler(markevery=classes, color=colors)
    with open('evaluation/BOBO_USTC_pred_raw.pkl', 'rb') as f:
        X = pickle.load(f)
    with open('evaluation/BOBO_USTC_labels.pkl', 'rb') as f2:
        labels = pickle.load(f2)

    print(f'载入{len(labels)}组测试数据')
    embeddings = TSNE(n_jobs=4).fit_transform(X)
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    fig, ax = plt.subplots()
    # labels = digits.target

    cate_x = []
    cate_y = []
    extend_list=[]
    for i in range(len(classes)):
        if i == 0:
            cate_index = [np.where(labels == k)[0] for k in range(0,10)]
        else :
            cate_index = [np.where(labels == k)[0] for k in range(10, 20)]
        for j in cate_index:
            extend_list.extend(j)
        cate_i_x = [vis_x[m] for m in extend_list]
        cate_i_y = [vis_y[m] for m in extend_list]
        cate_x.append(cate_i_x)
        cate_y.append(cate_i_y)
    print(cate_x)
    for index, _ in enumerate(classes):  # 在子图上分别画图，每次画一种颜色的点
        ax.scatter(cate_x[index], cate_y[index],  marker='.',c=colors[index],label=classes[index])
    ax.legend(loc='lower right')
    plt.savefig('./tsne_result_USTC_b.png', format='png', dpi=400)
    plt.show()

def example_code():
    # Define a list of markevery cases and color cases to plot
    cases = [None,
             8,
             (30, 8),
             [16, 24, 30],
             [0, -1],
             slice(100, 200, 3),
             0.1,
             0.3,
             1.5,
             (0.0, 0.1),
             (0.45, 0.1)]

    colors = ['#1f77b4',
              '#ff7f0e',
              '#2ca02c',
              '#d62728',
              '#9467bd',
              '#8c564b',
              '#e377c2',
              '#7f7f7f',
              '#bcbd22',
              '#17becf',
              '#1a55FF']

    # Configure rcParams axes.prop_cycle to simultaneously cycle cases and colors.
    mpl.rcParams['axes.prop_cycle'] = cycler(markevery=cases, color=colors)

    # Create data points and offsets
    x = np.linspace(0, 2 * np.pi)
    offsets = np.linspace(0, 2 * np.pi, 11, endpoint=False)
    yy = np.transpose([np.sin(x + phi) for phi in offsets])

    # Set the plot curve with markers and a title
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

    for i in range(len(cases)):
        ax.plot(yy[:, i], marker='o', label=str(cases[i]))
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.title('Support for axes.prop_cycle cycler with markevery')

    plt.show()
if __name__ == '__main__':
    # run()
    # run_USTC()
    run_USTC_binary()
    # example_code()
    # labels=np.array([0,1,2,3,1,2,3,4,1,22,11,13,14,23,5,1,2,4])
    # cate_index = [np.where(labels == k)[0] for k in range(0, 10)]
    # print(cate_index,type(cate_index))
    # alist=[]
    # for j in cate_index:
    #     alist.extend(j)
    # print(alist)
    # import numpy as np
    #
    # digits = load_digits()
    # label_list = digits.target
    # cate_0_index = np.where(label_list == 0)[0]
    # cate_0_list=[label_list[i] for i in cate_0_index]
    # print(cate_0_index)
    # print(cate_0_list)
