import numpy as np
import pandas as pd
import torch
import random
import os
from scipy.io import loadmat
from dataset.PUdata_dir import PU_3way_1,PU_3way_2,PU_3way_3,PU_3way_4
from dataset.PUdata_dir import PU_5way_2,PU_8way_2
from utils.training_utils import my_normalization
from dataset.mat2csv import get_data_csv,mat2csv_cw
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset
from utils.training_utils import meta_tr_tasks_utils


normalization = my_normalization

def sample_shuffle(data):
    """
    required: data.shape [Nc, num, ...]
    :param data: [[Nc, num, ...]]
    """
    for k in range(data.shape[0]):
        np.random.shuffle(data[k])
    return data


def get_class(sample):
    return sample.split('\\')[-3]

class DataGenFn:
    def __init__(self):
        # EB data:
        self.case5 = [PU_3way_1,PU_3way_2,PU_3way_3,PU_3way_4]  # C01, C02...
        #self.case_cross = dict(sq=T_sq, sa=T_sa)  # cw2sq:NC, IF, OF; cw2sa:NC, OF, RoF

    def PU_5way(self, way, order, examples=200, split=30, data_len=1024, shuffle=False,
                 normalize=True, label=False):
        """
        1. examples each file <= 119 * 1024
        2. if examples>=119, the value of overlap should be True
        """
        random.seed(1)
        file_dir = [self.case5[order]]
        print(file_dir)
        part=len(file_dir[0])
        print('part长度',part)
        train_files = []
        train_labels = []
        samples = dict()
        for i in range(part):
            temps1 = file_dir[0][i][0].split('N09',2)[0]
            name0 = 'N09_M07_F10_'
            name1 = temps1.split('\\')[-2] + '_'
            temps = [os.path.join(temps1,name0+name1+str(x)) for x in range(1, 21)]
            #print(temps)
            samples[i] = random.sample(temps, len(temps))
            part = samples[i]

            for j in range(part.__len__()):
                temp = part[j]
                #print(temp)
                data0 = loadmat(temp)[temp.split('\\')[-1]][0][0][2][0][6][2][0]
                data1 = data0[:data0.size // 2048 * 2048].reshape(-1, 2048)
                #print(data1.shape)
                if j == 0:
                    file = data1
                else:
                    file = np.vstack(
                        [file, data1])
                #print(file.shape)
                #file1=file[:,None]
                file1=np.expand_dims(file, 0)
            file1=file1[:,:2000,:]
            print(file1.shape)
            if i==0:
                train_labels.append(get_class(temp))
                train_files.append(file1)
            else:
                train_labels.append(get_class(temp))
                train_files.append(file1)
                #print(file1)

        print(train_labels)
        print(train_files)
        # for i in train_files[:5]:
        #     print(len(i))
        print('PU_{}way load [{}] loading ……'.format(way, order))
        n_way = len(train_files)  # 10 way
        print(n_way)
        assert n_way == way
        n_file = len(train_files)  # how many files for each way
        print(n_file)
        train_files=np.array(train_files).reshape(-1,data_len)
        if normalize:
            data_ = normalization(train_files)
        print(train_files.shape)
        data_size = train_files.shape[0]
        num_each_file = examples
        num_each_way = int(data_size/n_way)
        print(num_each_way)
        data_set = None
        data_set = train_files.reshape([n_way, num_each_way, 1, data_len])[:, :examples]
        print(data_set.shape)
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

    def data_folder(tr_d, tr_l):
        train_files = []
       # test_files = []
        train_labels = []
        #test_labels = []
        n_way=tr_d.shape[0]
        examples=tr_d.shape[1]
        data_len=tr_d.shape[3]
        # print(n_way)
        # print(examples)
        # print(data_len)
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



class DataManager(object):
    @staticmethod
    def get_data_loader(self, data_file):
        pass

class SetDataManager(DataManager):
    def __init__(self, signal_size=1024, n_way=5, n_support=5, n_shot=1, n_eposide = 100,train_num = 50):
        super(SetDataManager, self).__init__()
        self.signal_size = signal_size
        self.n_way = n_way
        self.batch_size = n_support
        self.n_eposide = n_eposide
        self.train_num = train_num
        #self.data_len = data_len
        self.shot = n_shot



    def get_data_loader(self,shuffle=False,split='train'): #parameters that would change on train/val set

        tr_d, tr_l,  te_d, te_l = DataGenFn().PU_5way(way=self.n_way, order=0, examples=200, split=self.train_num, data_len=self.signal_size, shuffle=False,
                                        normalize=True, label=True)
        print(tr_d.shape)
        print(tr_l.shape)
        if split=='train':
            train_loader = DataGenFn.data_folder(tr_d, tr_l)
            dataset = PU_train(train_loader, class_num=5)
        elif split=='test':
            test_loader = DataGenFn.data_folder( te_d, te_l)
            dataset = PU_train(test_loader, class_num=5)
        else:
            raise ValueError('Unknown split')
        print("data_loader")
        data_loader = DataLoader(dataset, batch_size=5, shuffle=True)

        print(data_loader)
        return data_loader

class SetDataManager2(DataManager):
    def __init__(self, signal_size=1024, n_way=5, n_support=10,n_query=16, n_shot=500, n_eposide = 100,train_num = 800,examples = 1000,order = 0):
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


    def get_data_loader(self,shuffle=False,split='train',domain = 'Source',num_meta = 3): #parameters that would change on train/val set

        tr_d, tr_l,  te_d, te_l = DataGenFn().PU_5way(way=self.n_way, order=self.order, examples=self.examples,
                             split=self.train_num, data_len=self.signal_size, shuffle=False,normalize=True, label=True)
        print('train data shape',tr_d.shape)
        print('train label shape',tr_l.shape)
        support, support_l, query, query_l = tr_d, tr_l,  te_d, te_l
        if split == 'train' or split == 'test':
            if domain == 'Target':
                support,  support_l , query ,query_l=meta_tr_tasks_utils(tr_d, tr_l, self.n_way, self.shot, self.query,
                                                                         length=self.signal_size ,split='Target')

            elif domain == 'Source':
                support, support_l = meta_tr_tasks_utils(tr_d, tr_l, self.n_way, self.shot,
                                                         self.query, length=self.signal_size, split='Source')

        if split=='train':
            if domain == 'Source':
                train_loader = DataGenFn.data_folder(support, support_l)
                dataset = PU_train(train_loader,class_num=self.n_way)
        elif split=='test':
            test_loader = DataGenFn.data_folder(query ,query_l)
            dataset = PU_train(test_loader,class_num=self.n_way)
        elif split=='meta' and domain =='Target':
            num_meta_support = num_meta
            num_support = support.size()[1]- num_meta_support
            s_meta,s_meta_l,sup,sup_l = meta_tr_tasks_utils( support, support_l,self.n_way,num_meta_support,
                                                             num_support, length=self.signal_size ,split='Target')
            print('s_meta shape is {}'.format(s_meta.shape))
            print('sup shape is {}'.format(sup.shape))
            print('test shape is {}'.format(query.shape))
            meta_train_loader = DataGenFn.data_folder(s_meta,s_meta_l)
            train_loader = DataGenFn.data_folder(sup, sup_l)
            test_loader = DataGenFn.data_folder(query, query_l)
            meta_train_dataset = PU_train(meta_train_loader, class_num=self.n_way)
            train_dataset = PU_train(train_loader, class_num=self.n_way)
            test_dataset = PU_train(test_loader, class_num=self.n_way)
            meta_train_data_loader = DataLoader(meta_train_dataset, batch_size=10, shuffle=True)
            train_data_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
            test_data_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
            return meta_train_data_loader,train_data_loader,test_data_loader


        print("data_loader")
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

        #print(data_loader)
        return data_loader

class PU_train(Dataset):

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
    tr_d, tr_l, te_d, te_l = d.PU_5way(way=3, order=0, examples=200, split=20,
                                      normalize=False, data_len=1024, label=True)