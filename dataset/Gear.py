import os
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from utils.sequence_transform import *
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader,Subset
import torch
from sklearn.utils import resample
from collections import Counter

import matplotlib.pyplot as plt
from collections import Counter

def plot_label_counts(label_counter):
    # 提取标签和计数
    labels = label_counter.keys()
    counts = label_counter.values()

    # 创建柱状图
    fig, ax = plt.subplots()
    ax.bar(labels, counts)

    # 设置标题和标签
    ax.set_title('Label Counts')
    ax.set_xlabel('Labels')
    ax.set_ylabel('Counts')

    # 自动调整标签旋转
    plt.xticks(rotation=45)

    # 显示图形
    plt.show()

def get_label_counts(dataloader):
    labels = dataloader.dataset.labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))
    return label_counts


def filter_df_by_labels(df, labels_dict):
    '''
    df: Input pandas dataframe with data path and labels
    labels_dict: Dictionary with label column names as keys and desired label values as values.
                 For example, {"Label1": "value1", "Label2": "value2"}.
    Returns a new dataframe filtered by the given labels.
    '''
    if not labels_dict:
        print('No labels provided for filtering, return origin data')
        return df
    df_filtered = df.copy()
    for label, value in labels_dict.items():
        df_filtered = df_filtered[df_filtered[label] == value]
    return df_filtered

def create_labels_dict(**kwargs):
    """
    Creates a dictionary with label column names as keys and provided label values as values.
    **kwargs: Key-value pairs where keys are label names and values are desired label values.
    """
    return kwargs

def get_files(dir,label='state',is_train=True,plot_counter=None,**kwargs):
    '''

    :param dir: csv direction, pd.dataframe
    :return:
    '''
    csv_list=dir['path'].tolist()
    csv_label=dir[label].tolist()
    if plot_counter is True:
        element_counts = Counter(csv_label)
        plot_label_counts(element_counts)
        print('label_counter',element_counts)

    data_dict ,label_dict= data_load(csv_list,labels=csv_label,**kwargs)

    return [data_dict ,label_dict]

def data_load(dir,labels,length=1024,data_num=20,ch=None):
    # assert len(dir)==len(labels)
    data_all=[]
    label_all=[]
    # data_dict={}
    for i in range(len(dir)):
        data_df = pd.read_csv(dir[i])
        label_df = labels[i]

        # 获取第2列的数值并转换为数组
        if ch is None:
            fl = data_df.iloc[:, 1:].values
        else:
            fl = data_df.iloc[:, 1:].values
            assert 0 <= ch <= fl.shape[-1] - 1
            fl=fl[:,ch].reshape(-1,1)

        if data_num=='all':
            print('all_data_num for one csv is', int(fl.shape[0]//length))
            data = [fl[start:end].T for start, end in zip(range(0, fl.shape[0], length),
                                                                    range(length, fl.shape[0] + 1,length))]
        elif data_num< fl.shape[0]//length:
             data = [fl[i:i+length, :].T for i in range(0, data_num*length, length)]
        else:
            raise ('data length choose wrong')

        for j in data:
            data_all.append(j)
            label_all.append(label_df)

    return data_all, label_all
    # print(label_all)

def data_transforms(dataset_type="train", normlize_type="-1-1"):
    transforms = {
        'train': Compose([
            # Reshape(),
            Normalize(normlize_type),
            RandomAddGaussian(),
            RandomScale(),
            RandomStretch(),
            # RandomCrop(),
            # Retype()

        ]),
        'val': Compose([
            # Reshape(),
            Normalize(normlize_type),
            # Retype()
        ])
    }
    return transforms[dataset_type]

def get_loaders(train_dataset, seed, batch, val_ratio=0.2):
    dataset_len = int(len(train_dataset))
    train_use_len = int(dataset_len * (1 - val_ratio))
    val_use_len = int(dataset_len * val_ratio)
    val_start_index = random.randrange(train_use_len)
    indices = torch.arange(dataset_len)

    train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index + val_use_len:]])
    train_subset = Subset(train_dataset, train_sub_indices)

    val_sub_indices = indices[val_start_index:val_start_index + val_use_len]
    val_subset = Subset(train_dataset, val_sub_indices)

    train_dataloader = DataLoader(train_subset, batch_size=batch,
                                  shuffle=True)

    val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False)

    return train_dataloader, val_dataloader

def convert_labels_to_numbers(label_set):
    """
    Convert string labels in a set to numeric labels.
    label_set: Set of string labels.
    Returns a dictionary with numeric labels.
    """
    unique_labels = sorted(list(set(label_set)))
    label_mapping = {label: index for index, label in enumerate(unique_labels)}

    # Ensure 'norm' is mapped to 0
    if 'norm' in label_mapping:
        # swap 'norm' label with the one currently having 0
        zero_label = [k for k, v in label_mapping.items() if v == 0][0]
        label_mapping['norm'], label_mapping[zero_label] = 0, label_mapping['norm']

    numeric_label_dict = {label: label_mapping[label] for label in unique_labels}

    return numeric_label_dict


class Gear(Dataset):

    def __init__(self, dir,label_index, normlizetype, is_train=True,**kwargs):
        self.data_dir = dir
        self.normlizetype = normlizetype
        self.is_train = is_train
        self.label_index=label_index
        list_data= get_files(self.data_dir,label=self.label_index, is_train=self.is_train,**kwargs)
        self.data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
        self.cls_num=set(list_data[1])
        self.cls_no=convert_labels_to_numbers( self.cls_num)

        if self.is_train:
            self.seq_data = self.data_pd['data'].tolist()
            self.labels = [self.cls_no[label] for label in self.data_pd['label'].tolist()]
            self.transform = data_transforms('train', self.normlizetype)
        else:
            self.seq_data = self.data_pd['data'].tolist()
            self.labels = [self.cls_no[label] for label in self.data_pd['label'].tolist()]
            self.transform = None

    def __len__(self):
        return len(self.data_pd)

    def __getitem__(self, idx):

            data = self.seq_data[idx]
            label = self.labels[idx]

            if self.transform:
                data = self.transform(data)
            return data, label

    def get_classes_num(self):
        return len(self.cls_num)# num, name

    def get_class_number(self):
        return  self.cls_no

class Gear_IM(Dataset):

    def __init__(self, dir,label_index, normlizetype,
                 is_train=True,is_im=None,
                 imb_type='exp', imb_factor=0.3,
                 **kwargs):
        self.data_dir = dir
        self.normlizetype = normlizetype
        self.is_train = is_train
        self.is_im=is_im
        self.label_index=label_index
        list_data= get_files(self.data_dir,label=self.label_index, is_train=self.is_train, **kwargs)
        self.cls_no=convert_labels_to_numbers(list_data[1])
        self.cls_num = len(self.cls_no)
        if self.is_train:
            if is_im is None:
                self.data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            else:
                self.data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
                self._imbalance_sampling(imb_type=imb_type, imb_factor=imb_factor)

            self.seq_data = self.data_pd['data'].tolist()
            self.labels = [self.cls_no[label] for label in self.data_pd['label'].tolist()]
            self.transform = data_transforms('train', self.normlizetype)
        else:
            self.data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            self.seq_data = self.data_pd['data'].tolist()
            self.labels = [self.cls_no[label] for label in self.data_pd['label'].tolist()]
            self.transform = None

    def __len__(self):
        return len(self.data_pd)

    def __getitem__(self, idx):
            data = self.seq_data[idx]
            label = self.labels[idx]
            if self.transform:
                data = self.transform(data)
            return data, label

    def get_num_classes(self):
        return self.cls_num# num, name

    def get_class_number(self):
        return  self.cls_no

    def _imbalance_sampling(self, imb_type, imb_factor):
        self.sample_num_per_cls = self._get_sample_num_per_cls(
            imb_type=imb_type,
            imb_factor=imb_factor,
            max_num=len(self.data_pd)//self.cls_num
        )
        data_pd_list = []
        for cls, group in self.data_pd.groupby('label'):
            num_samples = self.sample_num_per_cls[self.cls_no[cls]]
            data_pd_list.append(resample(group, replace=True, n_samples=num_samples, random_state=123))
        # Concatenate all dataframes
        self.data_pd = pd.concat(data_pd_list)

    def _get_sample_num_per_cls(self, imb_type, imb_factor, max_num):
        sample_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(self.cls_num):
                num = max_num * (imb_factor ** (cls_idx / (self.cls_num - 1.0)))
                sample_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(self.cls_num // 2):
                sample_num_per_cls.append(int(max_num))
            for cls_idx in range(self.cls_num // 2):
                sample_num_per_cls.append(int(max_num * imb_factor))
        elif imb_type == 'polar':
            for cls_idx in range(1):
                sample_num_per_cls.append(int(max_num))
            for cls_idx in range(self.cls_num - 1):
                sample_num_per_cls.append(int(max_num * imb_factor))
        elif imb_type == 'balance':
            for cls_idx in range(self.cls_num):
                num = max_num
                sample_num_per_cls.append(int(num))
        else:
            sample_num_per_cls.extend([int(max_num)] * self.cls_num)
        return sample_num_per_cls

    def get_cls_num_list(self):
        if self.is_train and self.is_im is not None:
                cls_num_list = []
                for i in range(self.cls_num):
                    cls_num_list.append(self.sample_num_per_cls[i])
                return cls_num_list
        else:
                cls_num_list = []
                data_num = len(self.data_pd) // self.cls_num
                for i in range(self.cls_num):
                    cls_num_list.append(data_num)
                return cls_num_list

if __name__ == '__main__':
    ori_root = '/home/lucian/Documents/datas/Graduate_data/Gear/dataframe.csv'
    ori_csv_pd = pd.read_csv(ori_root)
    # print(df.info)
    labels_dict = create_labels_dict(rpm=1000,load=0)
    # label=['path','label_all','box','rpm','load','state']

    # labels_dict={}
    df_out = filter_df_by_labels(ori_csv_pd, labels_dict)
    label_index='state'
    # print(df_out.info)
    # out=get_files(df_out)

    normlizetype = 'mean-std'
    datasets = {}
    datasets_train = Gear_IM(df_out,
                             label_index,
                             normlizetype,
                             is_train=True,
                             is_im=True,
                             imb_type='polar',
                             imb_factor=0.3,
                             length=1024,
                             ch=2,
                             data_num=20,
                                )
    datasets_val = Gear_IM(df_out,
                             label_index,
                             normlizetype,
                             is_train=False,
                             is_im=True,
                             imb_type='polar',
                             imb_factor=0.1,
                             length=1024,
                             ch=2,
                             data_num=20,
                                )

    train_dataloader=DataLoader(datasets_train, batch_size=128,shuffle=True)
    val_dataloader = DataLoader(datasets_val, batch_size=128, shuffle=False)
    # train_dataloader, val_dataloader = get_loaders(datasets_train, seed=5, batch=128,val_ratio=0.2)

    # labelc=get_label_counts(train_dataloader)
    # for id, (data, label) in enumerate(train_dataloader):
    #     print(id, data.shape, label)
    # # datasets_test = Gear(df_out,label_index, normlizetype, is_train=False)
    cls_num_=datasets_train.get_num_classes()
    cls=datasets_train.get_class_number()
    cls_list=datasets_train.get_cls_num_list()

    cls_num2=datasets_val.get_num_classes()
    cls2=datasets_val.get_class_number()
    cls_list2=datasets_val.get_cls_num_list()


