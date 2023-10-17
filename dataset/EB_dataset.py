import numpy as np
import pandas as pd
import torch
from abc import abstractmethod
from dataset.EBdata_dir import EB_5way_1,EB_5way_2
from dataset.EBdata_dir import EB_3way_1,EB_3way_2,EB_3way_3
from utils.training_utils import my_normalization
from dataset.mat2csv import get_data_csv
from utils.training_utils import meta_tr_tasks_utils
from torch.utils.data.sampler import Sampler
import random
from torch.utils.data import DataLoader, Dataset

normalization = my_normalization

def sample_shuffle(data):
    """
    required: data.shape [Nc, num, ...]
    :param data: [[Nc, num, ...]]
    """
    for k in range(data.shape[0]):
        np.random.shuffle(data[k])
    return data

class DataGenFn:
    def __init__(self):
        # EB data:
        self.case5 = [EB_3way_1,EB_3way_2,EB_3way_3]  # C01, C02...
        #self.case_cross = dict(sq=T_sq, sa=T_sa)  # cw2sq:NC, IF, OF; cw2sa:NC, OF, RoF

    def EB_5way(self, way, order, examples=200, split=30, data_len=1024, shuffle=False,
                 normalize=True, label=False):
        """
        1. examples each file <= 119 * 1024
        2. if examples>=119, the value of overlap should be True
        """
        file_dir = [self.case5[order]]
        print(file_dir)
        print('EB_{}way load [{}] loading ……'.format(way, order))
        n_way = len(file_dir[0])  # 10 way
        print(n_way)
        assert n_way == way
        n_file = len(file_dir)  # how many files for each way
        print(n_file)
        num_each_file = examples
        num_each_way = num_each_file * n_file
        data_size = num_each_file * data_len
        data_set = None
        for i in range(n_way):
            data_ = np.zeros([n_file, num_each_file, data_len])
            for j in range(n_file):
                data = DataGenFn.get_data_txt(file_dir=file_dir[j][i], num=data_size, header=0, shift_step=200)
                data = data.reshape([-1, data_len])
                print(data.shape)
                data_[j] = data
            data_ = data_.reshape([-1, data_len])  # [num_each_way, 2048]
            if normalize:
                data_ = normalization(data_)
            if i == 0:
                data_set = data_
            else:
                data_set = np.concatenate((data_set, data_), axis=0)
        data_set = data_set.reshape([n_way, num_each_way, 1, data_len])[:, :examples]
        if shuffle:
            data_set = sample_shuffle(data_set)  # 数据少 不建议打乱 不利于训练和测试集有序分开
        train_data, test_data = data_set[:, :split], data_set[:, split:]  # 先shuffle
        train_data, test_data = torch.from_numpy(train_data), torch.from_numpy(test_data)
        train_data, test_data = train_data.float(), test_data.float()

        if label:
            label = torch.arange(n_way, dtype=torch.long).unsqueeze(1)#增加一个维度
            label = label.repeat(1, examples)  # [Nc, examples]
            train_lab, test_lab = label[:, :split], label[:, split:]
            print(train_data.shape)
            print(train_lab.shape)
            print(test_data.shape)
            return train_data, train_lab, test_data, test_lab  # [Nc,num_each_way,1,2048], [Nc, 50]
        else:
            return train_data, test_data  # [Nc, num_each_way, 1, 2048]

    def get_data_txt(file_dir, num=100000, header=0, shift_step=200):
        """
        :param shift_step:
        :param num:
        :param header:
        :param file_dir:
        """
        data = pd.read_table(file_dir,sep='\t', header=header).values.reshape(-1)  # DataFrame ==> array
        data = data.reshape(-1,2)
        data = data[:, 1]
        #print(data)
        while data.shape[0] < num:
            header = header + shift_step
            data_ = pd.read_csv(file_dir, header=header).values.reshape(-1)
            data = np.concatenate((data, data_), axis=0)
        data = data[:num]

        # data = np.transpose(data, axes=[1, 0]).reshape(-1)
        return data

    def data_folder(tr_d, tr_l):
        train_files = []
       # test_files = []
        train_labels = []
        #test_labels = []
        n_way=tr_d.shape[0]
        examples=tr_d.shape[1]
        data_len=tr_d.shape[3]
        print('n_way:{:d} |examples:{:d} |data_len:{:d} '.format(n_way,examples,data_len))
        data_temp = tr_d
        #label_temp = tr_l
        label = np.arange(n_way).reshape(-1, 1)
        # print(label2)
        label = label.repeat(examples, axis=-1)
        for i in range(n_way):
            data1=[]
            for j in range(examples):
                data1 = data_temp[i][j]
                label = i
                data1 = data1.cuda().data.cpu().numpy()
                data1 = data1[np.newaxis,:, : ]
                # print(data1.shape)
                # print(label)
                if j == 0:
                    file = data1
                else:
                    file = np.vstack(
                        [file, data1])
            train_labels.append(label)
            train_files.append(file)
        train_folders = list(zip(train_files, train_labels))
        return train_folders

class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        else:
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

class SimpleDataset:
    def __init__(self, character_folders):
        self.character_folders = character_folders
        self.train_files = []
        self.train_labels = []
        d = self.character_folders

        self.meta = {}

        self.meta['image_names'] = []
        self.meta['image_labels'] = []


        for i, (data, label) in enumerate(d):
            print(i)
            self.meta['image_names'].append(data)
            self.meta['image_labels'].append(label)

    def __getitem__(self, i):

        img = self.meta['image_names'][i]
        target = self.meta['image_labels'][i]

        return img, target

    def __len__(self):
        return len(self.meta['image_names'])

class DataManager(object):
    @staticmethod
    def get_data_loader(self, data_file):
        pass

class SetDataManager(DataManager):
    def __init__(self, signal_size=1024, n_way=5, n_support=4,n_query=0, n_shot=1, n_eposide = 100,train_num = 500):
        super(SetDataManager, self).__init__()
        self.signal_size = signal_size
        self.n_way = n_way
        self.batch_size = n_support +n_query
        self.n_eposide = n_eposide
        self.train_num = train_num
        self.shot = n_shot
        self.query = n_query


    def get_data_loader(self,shuffle=False,split='train'): #parameters that would change on train/val set

        tr_d, tr_l,  te_d, te_l = DataGenFn().EB_5way(way=self.n_way, order=0, examples=1000, split=self.train_num, data_len=self.signal_size, shuffle=False,
                                        normalize=True, label=True)
        print(tr_d.shape)
        print(tr_l.shape)

        tr_d, tr_l = meta_tr_tasks_utils(tr_d, tr_l, self.n_way, self.shot, self.query, length=self.signal_size,split='Source')
        #dataset = torch.utils.data.TensorDataset(tr_d,tr_l)
        if split=='train':
            train_loader = DataGenFn.data_folder(tr_d, tr_l)
            dataset = EB_train(train_loader,class_num=self.n_way)
        elif split=='test':
            test_loader = DataGenFn.data_folder( te_d, te_l)
            dataset = EB_train(test_loader,class_num=self.n_way)
        else:
            raise ValueError('Unknown split')
        # sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide )
        # sampler = ClassBalancedSampler(self.n_way, self.shot, self.train_num, shuffle=shuffle)
        # data_loader_params = dict(batch_sampler = sampler,  num_workers = 0, pin_memory = True)
        # data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        print("data_loader")
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        #print(data_loader)
        return data_loader

class SetDataManager2(DataManager):
    def __init__(self, signal_size=1024, n_way=5, n_support=5,n_query=16, n_shot=1, n_eposide = 100,train_num = 500,examples = 1000,order = 0):
        super(SetDataManager2, self).__init__()
        self.signal_size = signal_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide
        self.train_num = train_num
        self.shot = n_shot
        self.examples = examples
        self.order = order
        self.query = n_query


    def get_data_loader(self,shuffle=False,split='train',domain = 'Target'): #parameters that would change on train/val set

        tr_d, tr_l,  te_d, te_l = DataGenFn().EB_5way(way=self.n_way, order=self.order, examples=self.examples, split=self.train_num, data_len=self.signal_size, shuffle=False,
                                        normalize=True, label=True)
        print(tr_d.shape)
        print(tr_l.shape)
        if domain == 'Target':
            support,  support_l , query ,query_l=meta_tr_tasks_utils(tr_d, tr_l, self.n_way, self.shot, self.query, length=self.signal_size ,split='Target')

        elif domain == 'Source':
            support, support_l = meta_tr_tasks_utils(tr_d, tr_l, self.n_way, self.shot, self.query, length=self.signal_size, split='Source')

        if split=='train':
            if domain == 'Source':
                train_loader = DataGenFn.data_folder(support, support_l)
                dataset = EB_train(train_loader,class_num=self.n_way)
        elif split=='test':
            test_loader = DataGenFn.data_folder(query ,query_l)
            dataset = EB_train(test_loader,class_num=self.n_way)

        print("data_loader")
        data_loader = DataLoader(dataset, batch_size=5, shuffle=True)

        print(data_loader)
        return data_loader



class EB_train(Dataset):

    def __init__(self, character_folders, class_num):
        self.character_folders = character_folders
        self.train_files = []
        self.train_labels = []
        class_folders = self.character_folders
        index = 0
        for class_folder in class_folders[:class_num]:
            (file, label) = class_folder
            np.random.shuffle(file)
            self.train_files += list(file)
            self.train_labels += [index for i in range(file.shape[0])]
            index += 1
        np.array(self.train_labels)

    def __getitem__(self, idx):
        image = self.train_files[idx]
        label = self.train_labels[idx]
        return image, np.int64(label)

    def __len__(self):
        return len(self.train_files)

if __name__ == "__main__":
    d = DataGenFn()
    tr_d, tr_l, te_d, te_l = d.EB_5way(way=3, order=0, examples=200, split=20,
                                     normalize=False, data_len=1024, label=True)

    #base_loader = SetDataManager(signal_size=1024).get_data_loader(shuffle=False)