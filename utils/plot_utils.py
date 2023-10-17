import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import umap
import os
import torch

def set_figure(font_size=10., tick_size=8., ms=7., lw=1.2, fig_w=8.):
    # print(plt.rcParams.keys())  # 很有用，查看所需属性
    # exit()
    cm_to_inc = 1 / 2.54  # 厘米和英寸的转换 1inc = 2.54cm
    w = fig_w * cm_to_inc  # cm ==> inch
    h = w * 3 / 4
    plt.rcParams['figure.figsize'] = (w, h)  # 单位 inc
    plt.rcParams['figure.dpi'] = 300
    # plt.rcParams['figure.figsize'] = (14 * cm_to_inc, 6 * cm_to_inc)

    plt.rc('font', family='Times New Roman', weight='normal', size=str(font_size))
    plt.rcParams['axes.linewidth'] = lw  # 图框宽度

    # plt.rcParams['lines.markeredgecolor'] = 'k'
    plt.rcParams['lines.markeredgewidth'] = lw
    plt.rcParams['lines.markersize'] = ms

    # 刻度在内
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.labelsize'] = tick_size
    plt.rcParams['xtick.major.width'] = lw
    plt.rcParams['xtick.major.size'] = 2.5  # 刻度长度

    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.labelsize'] = tick_size
    plt.rcParams['ytick.major.width'] = lw
    plt.rcParams['ytick.major.size'] = 2.5

    plt.rcParams["legend.frameon"] = True  # 图框
    plt.rcParams["legend.framealpha"] = 0.8  # 不透明度
    plt.rcParams["legend.fancybox"] = False  # 圆形边缘
    plt.rcParams['legend.edgecolor'] = 'k'
    plt.rcParams["legend.columnspacing"] = 1  # /font unit 以字体大小为单位
    plt.rcParams['legend.labelspacing'] = 0.2
    plt.rcParams["legend.borderaxespad"] = 0.5
    plt.rcParams["legend.borderpad"] = 0.3

def check_creat_new(path):
    if os.path.exists(path):
        split_f = os.path.split(path)
        new_f = os.path.join(split_f[0], split_f[1][:-4] + '(1).jpg')
        new_f = check_creat_new(new_f)  # in case the new file exist
    else:
        new_f = path
    return new_f


def plot_confusion_matrix(y_true, y_pred, disp_acc=True):
    """
    :param y_pred: (nc*nq, )
    :param y_true: (nc*nq, )
    :param disp_acc:
    """
    plt.rc('font', family='Times New Roman', style='normal', weight='light', size=11)
    f, ax = plt.subplots()
    cm = confusion_matrix(y_true, y_pred)
    font1 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 12,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 10,
             }

    if disp_acc:  # 归一化,可显示准确率accuracy,默认显示准确率
        cm = cm.astype('float32') / (cm.sum(axis=1)[:, np.newaxis])
        sns.heatmap(cm, annot=True, ax=ax, cmap='YlGnBu', fmt='.2f',
                    linewidths=0.02, linecolor="w", vmin=0, vmax=1)
        # cmap如: cividis, Purples, PuBu, viridis, magma, inferno; fmt: default=>'.2g'
    else:
        sns.heatmap(cm, annot=True, ax=ax, cmap='plasma')
        # cmap如: , viridis, magma, inferno; fmt: default=>'.2g'

    ax.set_xlabel('Predicted label', fontdict=font1)
    ax.set_ylabel('True label', fontdict=font1)

    root = r'E:\code'
    file = r'TRN-EWM\photos' + r'\14' + r'.jpg'
    path = os.path.join(root, file)
    path = check_creat_new(path)
    if not os.path.exists(path):
        plt.savefig(path, dpi=600)
        print('Save confusion matrix.eps to \n', path)
        plt.show()
    # ax.set_title('Confusion Matrix', fontdict=font)
    # 注意在程序末尾加 plt.show()

def tSNE_fun(input_data, shot, name=None, labels=None, n_dim=2):
    """
    :param shot:
    :param labels:
    :param input_data:  (n, dim)
    :param name: name
    :param n_dim: 2d or 3d
    :return: figure
    """
    t0 = time()
    classes = input_data.shape[0] // shot
    # da = umap.UMAP(n_neighbors=shot, n_components=n_dim, random_state=0).fit_transform(input_data)
    da = TSNE(n_components=n_dim, perplexity=shot, init='pca', random_state=0,
              angle=0.3).fit_transform(input_data)  # (n, n_dim)
    da = MinMaxScaler().fit_transform(da)  # [0, 1]

    color_set = [
        [0.00, 0.45, 0.74],  # 蓝色
        [0.93, 0.69, 0.13],  # 黄色
        [0.85, 0.33, 0.10],  # 橘红色
        [0.49, 0.18, 0.56],  # 紫色
        [0.47, 0.67, 0.19],  # 绿色
        [0.30, 0.75, 0.93],  # 淡蓝色
        [0.64, 0.08, 0.18],  # 棕色
    ]
    color = [
        [0.00, 0.45, 0.74],  # 蓝色
        [0.64, 0.08, 0.18],  # 棕色
        [0.46, 0.65, 0.20],  # 绿色
        [0.30, 0.75, 0.93],  # 淡蓝色
        [0.85, 0.33, 0.10],  # 橘红色
        [0.73, 0.92, 0.47],  # 淡绿色
    ]
    # color = color_set[:classes // 2] + color_set[:classes // 2]
    color = np.asarray(color)
    color = np.tile(color[:classes][:, None], (1, shot, 1)).reshape(-1, 3)
    mark = ['o', '^', '.', 'v', 's', 'D']
    # method 1:
    # m1 = [mark[0]] * (classes // 2)
    # m2 = [mark[1]] * (classes // 2)
    # mark = m1 + m2
    # method 2:
    # mark = mark[:(classes // 2)] + mark[:(classes // 2)]
    # method 3:
    mark = [mark[0]] * classes

    label = []
    if labels is None:
        for i in range(1, classes // 2 + 1):
            lb = 'S-' + str(i)
            label.append(lb)
        for i in range(1, classes // 2 + 1):
            lb = 'T-' + str(i)
            label.append(lb)
        labels = label
    # print(len(labels), classes)
    assert len(labels) == classes

    set_figure(ms=6., fig_w=5., font_size=8, tick_size=8, lw=1.)  # 一行 2图，宽度可设为 8cm，3图设为 6cm以下。
    figs = plt.figure()  # figsize:[6.4, 4.8]
    ax = figs.add_subplot(111)
    for i in range(1, classes + 1):
        # s: 大小 建议50-100, alpha: 不透明度 0.5-0.8
        ax.scatter(da[(i - 1) * shot:i * shot, 0], da[(i - 1) * shot:i * shot, 1], s=30,
                   c=color[(i - 1) * shot:i * shot], alpha=1,
                   marker=mark[i - 1], label=labels[i - 1])

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    # ax.legend(prop=font, ncol=4, bbox_to_anchor=(0.5, -0.1), loc="upper center")
    ax.legend(ncol=2, loc="upper left").get_frame().set_linewidth(1)

    if name is not None:
        title = 'UMAP embedding of %s (time %.2fs)' % (name, (time() - t0))
        plt.title(title)
    print('t-SNE Done!')
    return figs

def plot_tsne( x_s, x_t ):
    root = r'E:\code'
    file = r'TRN-EWM\photos' + r'\14' + r'.jpg'
    path = os.path.join(root, file)
    save_fig_path = path
    print('TRN-EWM labels used for t-sne!')
    labels = ['NC', 'IF', 'OF']
    classes=3
    shot = int(np.ceil(x_s.shape[0] / classes))
    # print('CW2SA labels used for t-sne!')
    # labels = ['NC-s', 'OF-s', 'ReF-s', 'NC-t', 'OF-t', 'ReF-t']  # CW2SA

    #x = torch.cat((x_s, x_t), dim=0)  # [n, dim]
    t_sne(x_s, classes=classes, name=None, labels=labels, n_dim=2)
    plt.show()
    order = input('Save fig? Y/N\n')
    if order == 'y' or order == 'Y':
        t_sne(x_s, classes=classes, name=None, labels=labels, n_dim=2)
        new_path = check_creat_new(save_fig_path)
        plt.savefig(new_path, dpi=600, format='svg', bbox_inches='tight', pad_inches=0.01)
        print('Save t-SNE.eps to \n', new_path)

def t_sne(input_data, classes, input_label=None, name=None, labels=None, n_dim=2):
    """
    :param labels:
    :param input_label:(n,)
    :param input_data:  (n, dim)
    :param name: name
    :param classes: number of classes
    :param n_dim: 2d or 3d
    :return: figure
    """
    # input_label = input_label.astype(dtype=int)
    shot = int(np.ceil(input_data.shape[0] / classes))
    # t0 = time()
    da = TSNE(n_components=n_dim, perplexity=shot,
              init='pca', random_state=0, angle=0.3).fit_transform(input_data)
    da = MinMaxScaler().fit_transform(da)  # (n, n_dim)

    figs = plt.figure()
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 10}
    mark = ['o', 'v', 's', 'p', '*', 'h', '8', '.', '4', '^', '+', 'x', '1', '2']
    # 实心圆，正三角，正方形，五角，星星，六角，八角，点，tri_right, 倒三角...

    if labels is None:
        if classes == 3:
            labels = ['NC', 'IF', 'OF']
        elif classes == 4:
            labels = ['NC', 'IF', 'OF', 'RF']  # for EB
        elif classes == 5:
            labels = ['NC', 'IF', 'OF', 'RF', 'ReF']  # for EB
        elif classes == 7:
            labels = ['NC', 'IF-1', 'IF-2', 'IF-3', 'OF-1', 'OF-2', 'OF-3']  # for SQ
            # labels = ['NC', 'OF-1', 'OF-2', 'OF-3', 'OF-P', 'ReF', 'RoF']  # for SA
        elif classes == 13:
            # labels = ['NC', 'IF-1', 'IF-2', 'IF-3', 'OF-1', 'OF-2', 'OF-3',
            #           'RF-1', 'RF-2', 'RF-3', 'CF-1', 'CF-2', 'CF-p']
            labels = ['NC', 'IF-1', 'IF-2', 'IF-3', 'OF-1', 'OF-2', 'OF-3',
                      'RF-1', 'RF-2', 'RF-3', 'rF-1', 'rF-2', 'rF-3']
    assert len(labels) == classes

    plt.rc('font', family='Times New Roman', style='normal', weight='light', size=10)
    ax = figs.add_subplot(111)
    # "husl", "muted"
    palette = np.array(sns.color_palette(palette="husl", n_colors=classes))[:, np.newaxis]  # [classes, 1, 3]
    # print(palette.shape)
    palette = np.tile(palette, (1, shot, 1)).reshape(-1, 3)
    for i in range(1, classes + 1):
        ax.scatter(da[(i - 1) * shot:i * shot, 0], da[(i - 1) * shot:i * shot, 1], s=100,
                   c=palette[(i - 1) * shot:i * shot], alpha=0.8,
                   marker=mark[i - 1], label=labels[i - 1])  # 每次画一个类
    ax.set_xlim(-0.1, 1.2)
    ax.set_ylim(-0.1, 1.2)
    ax.legend(loc='upper right', prop=font, labelspacing=1)

    if shot == 200:
        root = r'E:\code'
        file = r'TRN-EWM' + r'\photos\imgs\CITN_meta.jpg'
        path = os.path.join(root, file)
        path = check_creat_new(path)
        plt.savefig(path, dpi=600)
        print('Save t-SNE.jpg to \n', path)
        plt.show()

    # title = 't-SNE embedding of %s (time %.2fs)' % (name, (time() - t0))
    # plt.title(title)
    print('t-SNE Done!')
    return figs