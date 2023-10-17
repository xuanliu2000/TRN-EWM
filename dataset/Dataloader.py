import numpy as np
import torch
import random
import os
import pandas as pd
from scipy.io import loadmat

import copy
from dataset.PUdata_dir import PU_3way_1,PU_3way_2,PU_3way_3,PU_3way_4,PU_5way_1,PU_5way_2
from dataset.EB_dataset import EB_3way_1,EB_3way_2,EB_3way_3
from torch.utils.data import DataLoader, Dataset
from utils.training_utils import meta_tr_tasks_utils,im_tasks_utils
from utils.training_utils import my_normalization

normalization = my_normalization


def get_class(sample):
    return sample.split('\\')[-3]

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
        # PU data:
        self.case3 = [PU_3way_1,PU_3way_2,PU_3way_3,PU_3way_4]  # C01, C02...
        self.case5 = [EB_3way_1,EB_3way_2,EB_3way_3]
        self.case8 = [PU_5way_1,PU_5way_2]

    def PU_5way_imbalance(self, way, order, data_len=1024, shuffle=False,data_slice = [50,5,5],test_slice = [540,30,30],
                          meta_num = 10 , normalize=True, label=False):
        """
        1. examples each file <= 119 * 1024
        2. if examples>=119, the value of overlap should be True
        """
        print('PU_{}way load [{}] loading ……'.format(way, order))
        random.seed(2)
        file_dir = [self.case3[order]]
        print(file_dir)
        part = len(file_dir[0])
        print('part长度', part)
        train_files = []
        train_labels = []
        samples = dict()
        for i in range(part):
            temps1 = file_dir[0][i][0].split('N09', 2)[0]
            name0 = 'N09_M07_F10_'
            name1 = temps1.split('\\')[-2] + '_'
            temps = [os.path.join(temps1, name0 + name1 + str(x)) for x in range(1, 21)]
            # print(temps)
            samples[i] = random.sample(temps, len(temps))
            part = samples[i]

            for j in range(part.__len__()):
                temp = part[j]
                # print(temp)
                data0 = loadmat(temp)[temp.split('\\')[-1]][0][0][2][0][6][2][0]
                data1 = data0[:data0.size // 1024 * 1024].reshape(-1, 1024)
                # print(data1.shape)
                if j == 0:
                    file = data1
                else:
                    file = np.vstack(
                        [file, data1])
            #shuffle sampler
            file = sample_shuffle(file)
            # print(file)
            file1 = file[ :data_slice[i], :]
            file_t = file[ data_slice[i]:data_slice[i]+test_slice[i], :]
            file_m = file[ data_slice[i]+test_slice[i]:data_slice[i]+test_slice[i]+meta_num, :]
            print('label {} shape is {}'.format(i, file1.shape))
            if i ==0:
                file2 = file1
                file2_t = file_t
                file2_m = file_m
            else:
                file2 = np.vstack([file2,file1])
                file2_t = np.vstack([file2_t,file_t])
                file2_m = np.vstack([file2_m,file_m])
            label_slice = data_slice
            label_o = np.full([label_slice[i]],i)
            print(label_o)
            label_o = np.expand_dims(label_o, 1)
            label_t = np.full([test_slice[i]],i)
            label_t = np.expand_dims(label_t, 1)
            label_m = np.full([meta_num],i)
            label_m = np.expand_dims(label_m, 1)
            #print(label.shape)
            if i == 0:
                train_l = label_o
                test_l = label_t
                meta_l = label_m
            else:
                train_l = np.vstack([train_l,label_o])
                test_l = np.vstack([test_l, label_t])
                meta_l = np.vstack([meta_l, label_m])
        print('data.shape is',file2.shape)
        print('T_data.shape is', file2_t.shape)
        # file2 = sample_shuffle(file2)
        # file2_t = sample_shuffle(file2_t)
        # file2_m = sample_shuffle(file2_m)
        data_set = file2.reshape([file2.shape[0],1,data_len])
        data_set_t = file2_t.reshape([file2_t.shape[0], 1, data_len])
        data_set_m = file2_m.reshape([file2_m.shape[0], 1, data_len])
        print(data_set_m.shape)
        train_l = train_l.reshape(-1)
        test_l = test_l.reshape(-1)
        meta_l = meta_l.reshape(-1)
        print('label.shape is',train_l.shape)
        print('T_label.shape is', test_l.shape)
        print('m_label.shape is', meta_l.shape)

        if normalize:
            data_set = my_normalization(data_set)
            data_set_t = my_normalization(data_set_t)
            data_set_m = my_normalization(data_set_m)

        train_data, test_data , meta_data = data_set,data_set_t, data_set_m # 先shuffle
        train_data, test_data , meta_data = torch.from_numpy(train_data), torch.from_numpy(test_data), torch.from_numpy(meta_data)
        train_data, test_data , meta_data = train_data.float(), test_data.float(), meta_data.float()
        train_l, test_l, meta_l = torch.from_numpy(train_l), torch.from_numpy(test_l), torch.from_numpy(meta_l)

        if label:
            print(train_data.shape)
            print(test_data.shape)
            print(train_l.shape)
            print(test_l.shape)
            print(meta_data.shape)
            print(meta_l.shape)
            print('test_l is',test_l)
            return train_data, train_l, test_data, test_l ,meta_data, meta_l# [Nc,num_each_way,1,2048], [Nc, 50]
        else:
            return train_data, test_data  # [Nc, num_each_way, 1, 2048]

    def EB_5way_imbalance(self, way, order, data_len=1024, shuffle=False, data_slice=[50, 5, 5],
                          test_slice=[540, 30, 30],
                          meta_num=10, normalize=True, label=False):
        """
        1. examples each file <= 119 * 1024
        2. if examples>=119, the value of overlap should be True
        """
        file_dir = [self.case5[order]]
        print('EB_{}way load [{}] loading ……'.format(way, order))
        print(file_dir)
        part = len(file_dir[0])
        print('part长度', part)
        train_files = []
        train_labels = []
        samples = dict()
        n_file = len(file_dir)
        for i in range(part):
            num_each_file = data_slice[i]
            data_size = num_each_file * n_file
            data_ = np.zeros([num_each_file, data_len])
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
        n_way=tr_d.shape[1]
        examples=tr_d.shape[0]
        data_len=tr_d.shape[2]
        # print(n_way)
        # print(examples)
        # print(data_len)
        data_temp = tr_d
        label_temp = tr_l

        data_list_train = {}
        data_temp = data_temp.cuda().data.cpu().numpy()
        label_temp = label_temp.cuda().data.cpu().numpy()
        for j in range(5):
            data_list_train[j] = [data_temp[i] for i, label in enumerate(label_temp) if label == j]
            #print(data_list_train[j])
            #print('j label is', j)
            train_labels.append(j)
            # print('label is', train_labels)
            train_files.append(data_list_train[j])
        print('train label is', train_labels)
        print(train_files)
        train_folders = list(zip(train_files, train_labels))
        return train_folders

        # for i in range(n_way):
        #     data1=[]
        #     for j in range(examples):
        #         data1 = data_temp[j]
        #         label = label_temp[j]
        #         #print(label)
        #         # print(label)
        #         data1 = data1.cuda().data.cpu().numpy()
        #         label = label.cuda().data.cpu().numpy()
        #         # print(data1.shape)
        #         data1 = data1[np.newaxis,:, : ]
        #         # print(data1.shape)
        #         if j == 0:
        #             file = data1
        #             label1 = label
        #         else:
        #             file = np.vstack(
        #                 [file, data1])
        #             label1 = np.vstack(
        #                 [label1, label])
        #     train_labels.append(label1)
        #    # print('label is', train_labels)
        #     train_files.append(file)
        # train_folders = list(zip(train_files, train_labels))
        # return train_folders

class DataManager(object):
    @staticmethod
    def get_data_loader(self, data_file):
        pass

class SetDataManager2(DataManager):
    def __init__(self, signal_size=1024, n_shot=500,way= 3, n_eposide = 100,train_num = 800,examples = 1000,order = 1):
        super(SetDataManager2, self).__init__()
        self.signal_size = signal_size

        self.n_eposide = n_eposide
        self.train_num = train_num
        self.shot = n_shot
        self.examples = examples
        self.order = order
        self.n_way = way
        self.num_meta = 2


    def get_data_loader(self,shuffle=False,split='train',domain = 'Target',num_meta = 4): #parameters that would change on train/val set

        tr_d, tr_l,  te_d, te_l,me_d,me_l = DataGenFn().PU_5way_imbalance(way=self.n_way, order=self.order,
                             data_len=self.signal_size, shuffle=False,data_slice = [200, 10,10],meta_num = num_meta,
                                            test_slice = [200,200, 200],  normalize=False, label=True)
        print('train data shape',tr_d.shape)
        print('train label shape',tr_l.shape)
        print('meta label shape', me_l.shape)
        qu, qu_l,support, support_l =im_tasks_utils(tr_d, tr_l,5,1024,split = 'Target')
        su, su_l, query, query_l = im_tasks_utils(te_d, te_l, 5, 1024, split='Target')
        meta, meta_l =im_tasks_utils(me_d, me_l,2,1024,split = 'meta')
        if split=='train':
            if domain == 'Source':
                train_loader = DataGenFn.data_folder(support, support_l)
                dataset = PU_train(train_loader,class_num=self.n_way)
        elif split=='test':
            test_loader = DataGenFn.data_folder(query ,query_l)
            dataset = PU_train(test_loader,class_num=self.n_way)
        elif split=='meta' and domain =='Target':
            # num_meta_support = num_meta
            # num_support = support.size()[0]- num_meta_support
            # s_meta,s_meta_l,sup,sup_l = im_tasks_utils( support, support_l,self.n_way,num_meta_support,
            #                                                  num_support, length=self.signal_size ,split='Target')
            sup, sup_l, query, query_l, s_meta, s_meta_l = support, support_l, query, query_l,meta, meta_l
            print('s_meta shape is {}'.format(s_meta.shape))
            print('sup shape is {}'.format(sup.shape))
            print('test shape is {}'.format(query.shape))
            meta_train_loader = DataGenFn.data_folder(s_meta,s_meta_l)
            train_loader = DataGenFn.data_folder(sup, sup_l)
            test_loader = DataGenFn.data_folder(query, query_l)
            meta_train_dataset = PU_train(meta_train_loader, class_num=self.n_way)
            train_dataset = PU_train(train_loader, class_num=self.n_way)
            test_dataset = PU_train(test_loader, class_num=self.n_way)


            # data_list_train = {}
            # for j in range(self.n_way):
            #     data_list_train[j] = [i for i, label in enumerate(train_dataset.train_labels) if label == j]
            #     print('j label is',j)
            #
            # img_num_list = [self.num_meta ]*self.n_way
            # idx_to_meta = []
            # idx_to_train = []
            # print(img_num_list)
            #
            # for cls_idx, img_id_list in data_list_train.items():
            #     np.random.shuffle(img_id_list)
            #     print(cls_idx)
            #     img_num = img_num_list[int(cls_idx)]
            #     idx_to_meta.extend(img_id_list[:img_num])
            #     idx_to_train.extend(img_id_list[img_num:])
            # train_data = copy.deepcopy(train_dataset)
            # train_data_meta = copy.deepcopy(train_dataset)
            # train_data_meta.train_files = np.delete(train_dataset.train_files, idx_to_train, axis=0)
            # train_data_meta.train_labels = np.delete(train_dataset.train_labels, idx_to_train, axis=0)
            # train_data.train_files = np.delete(train_dataset.train_files, idx_to_meta, axis=0)
            # train_data.train_labels = np.delete(train_dataset.train_labels, idx_to_meta, axis=0)
            # print(train_data.train_files.shape)
            # print(train_data_meta.train_files.shape)

            meta_train_data_loader = DataLoader(meta_train_dataset, batch_size=4, shuffle=True)
            train_data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            test_data_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

            return meta_train_data_loader,train_data_loader,test_data_loader


        print("data_loader")
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

        #print(data_loader)
        return data_loader

class SetDataManager(DataManager):
    def __init__(self, signal_size=1024, n_shot=500, n_eposide = 100,train_num = 800,examples = 1000,order = 3):
        super(SetDataManager2, self).__init__()
        self.signal_size = signal_size

        self.n_eposide = n_eposide
        self.train_num = train_num
        self.shot = n_shot
        self.examples = examples
        self.order = order
        self.n_way = 3


    def get_data_loader(self,shuffle=False,split='train',domain = 'Target',num_meta = 2): #parameters that would change on train/val set

        tr_d, tr_l,  te_d, te_l,me_d,me_l = DataGenFn().PU_5way_imbalance(way=self.n_way, order=self.order,
                             data_len=self.signal_size, shuffle=False,data_slice = [5, 5, 5],meta_num = num_meta,
                                            test_slice = [200, 200, 200],  normalize=False, label=True)
        print('train data shape',tr_d.shape)
        print('train label shape',tr_l.shape)
        print('meta label shape', me_l.shape)
        support, support_l, qu, qu_l =im_tasks_utils(tr_d, tr_l,5,1024,split = 'Target')
        su, su_l, query, query_l = im_tasks_utils(te_d, te_l, 5, 1024, split='Target')
        meta, meta_l =im_tasks_utils(me_d, me_l,2,1024,split = 'meta')
        if split=='train':
            if domain == 'Source':
                train_loader = DataGenFn.data_folder(support, support_l)
                dataset = PU_train(train_loader,class_num=self.n_way)
        elif split=='test':
            test_loader = DataGenFn.data_folder(query ,query_l)
            dataset = PU_train(test_loader,class_num=self.n_way)
        elif split=='meta' and domain =='Target':
            # num_meta_support = num_meta
            # num_support = support.size()[0]- num_meta_support
            # s_meta,s_meta_l,sup,sup_l = im_tasks_utils( support, support_l,self.n_way,num_meta_support,
            #                                                  num_support, length=self.signal_size ,split='Target')
            sup, sup_l, query, query_l, s_meta, s_meta_l = support, support_l, query, query_l,meta, meta_l
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
            self.train_labels += [index for i in range(len(file))]
            index += 1
        np.array(self.train_labels)

    def __getitem__(self, idx):
       # print('idx is',idx)
        image = self.train_files[idx]
        label = self.train_labels[idx]
        return image, np.int64(label)

    def __len__(self):
        return len(self.train_files)


if __name__ == "__main__":
    d = DataGenFn()
    tr_d, tr_l, te_d, te_l,me_d,me_l = d.PU_5way_imbalance(way=3, order=0, meta_num = 10,
                                      normalize=False, data_len=1024, label=True)