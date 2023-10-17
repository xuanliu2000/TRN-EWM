import numpy as np
import pandas as pd
import torch
import random
import os
from scipy.io import loadmat
from dataset.PUdata_dir import PU_3way_1,PU_3way_2,PU_3way_3,PU_3way_4
from dataset.PUdata_dir import PU_5way_1,PU_5way_2,PU_8way_2
from utils.training_utils import my_normalization
from dataset.mat2csv import get_data_csv,mat2csv_cw
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset
from utils.training_utils import meta_tr_tasks_utils,im_tasks_utils


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
        # PU data:
        self.case3 = [PU_3way_1,PU_3way_2,PU_3way_3,PU_3way_4]  # C01, C02...
        self.case5 = [PU_5way_1,PU_5way_2]

    # balance data slice
    def PU_5way(self, way, order, examples=200, split=30, data_len=1024, shuffle=False,
                 normalize=True, label=False,task =1):
        print('PU_balance_{}way load [{}] loading ……'.format(way, order))
        random.seed(1)
        if way ==3:
            file_dir = [self.case3[order]]
        elif way ==5:
            file_dir = [self.case5[order]]
        else:
            print('way is no match case')
        print(file_dir)
        part=len(file_dir[0])
        print('part长度',part)
        train_files = []
        train_labels = []
        samples = dict()
        for i in range(part):
            temps1 = file_dir[0][i][0].split('N09', 2)[0]
            if task == 1:
                name0 = 'N15_M07_F10_'
            elif task ==2:
                name0 = 'N09_M07_F10_'
            elif task == 3:
                name0 = 'N15_M01_F10_'
            elif task == 4:
                name0 = 'N15_M07_F04_'
            name1 = temps1.split('\\')[-2] + '_'
            temps = [os.path.join(temps1,name0+name1+str(x)) for x in range(1, 21)]
            #print(temps)
            samples[i] = random.sample(temps, len(temps))
            part = samples[i]

            for j in range(part.__len__()):
                temp = part[j]
                data0 = loadmat(temp)[temp.split('\\')[-1]][0][0][2][0][6][2][0]
                data1 = data0[:data0.size // data_len * data_len].reshape(-1, data_len)
                if j == 0:
                    file = data1
                else:
                    file = np.vstack(
                        [file, data1])
                file = sample_shuffle(file)
                file1=np.expand_dims(file, 0)
            file1=file1[:,:2000,:]
            if i == 0:
                data_set = file1
            else:
                data_set = np.concatenate((data_set, file1), axis=0)
            print(file1.shape)
            # train_labels.append(i)
            # train_files.append(file1)
        #print(train_labels)
        #train_files=np.array(train_files).reshape(-1,data_len)
        #
        # data_set = data_set[:,:,np.newaxis,:]
        print(data_set.shape)
        data_set = data_set[:, :examples,  :]
        data_set =data_set.reshape(-1,data_len)
        if normalize:
            data_set = normalization(data_set)
        data_set=data_set.reshape([way,examples,1,data_len])
        print(data_set.shape)
        if shuffle:
            data_set = sample_shuffle(data_set)  # 数据少 不建议打乱 不利于训练和测试集有序分开
        train_data, test_data = data_set[:, :split], data_set[:, split:]  # 先shuffle
        train_data, test_data = torch.from_numpy(train_data), torch.from_numpy(test_data)
        train_data, test_data = train_data.float(), test_data.float()

        if label:
            label = torch.arange(way, dtype=torch.long).unsqueeze(1)#增加一个维度
            label = label.repeat(1, examples)  # [Nc, examples]
            train_lab, test_lab = label[:, :split], label[:, split:]
            print('PU shape:train_data{}|train_lab{}|test_data{}'.format(train_data.shape,train_lab.shape,test_data.shape))
            return train_data, train_lab, test_data, test_lab  # [Nc,num_each_way,1,2048], [Nc, 50]
        else:
            return train_data, test_data  # [Nc, num_each_way, 1, 2048]

    def PU_5way_imbalance(self, way, order, data_len=1024, shuffle=False,data_slice = [50,5,5],test_slice = [540,30,30],
                          meta_num = 10 , normalize=True, label=False,task = 1):
        """
        1. examples each file <= 119 * 1024
        2. if examples>=119, the value of overlap should be True
        """
        print('PU_imbalance_{}way load [{}] loading ……'.format(way, order))
        random.seed(2)
        if way ==3:
            file_dir = [self.case3[order]]
        elif way ==5:
            file_dir = [self.case5[order]]
        else:
            print('way is no match case')
        print(file_dir)
        part = len(file_dir[0])
        print('part长度', part)
        assert part == way
        train_files = []
        train_labels = []
        samples = dict()
        for i in range(part):
            temps1 = file_dir[0][i][0].split('N09', 2)[0]
            if task == 1:
                name0 = 'N15_M07_F10_'
            elif task == 2:
                name0 = 'N09_M07_F10_'
            elif task == 3:
                name0 = 'N15_M01_F10_'
            elif task == 4:
                name0 = 'N15_M07_F04_'
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

        data_set = file2.reshape([file2.shape[0],1,data_len])
        data_set_t = file2_t.reshape([file2_t.shape[0], 1, data_len])
        data_set_m = file2_m.reshape([file2_m.shape[0], 1, data_len])
        print(data_set_m.shape)
        train_l = train_l.reshape(-1)
        test_l = test_l.reshape(-1)
        meta_l = meta_l.reshape(-1)
        print('PU shape:label{}|T_label{}|m_label {}'.format(train_l.shape, test_l.shape, meta_l.shape))

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
            return train_data, train_l, test_data, test_l ,meta_data, meta_l# [Nc,num_each_way,1,2048], [Nc, 50]
        else:
            return train_data, test_data  # [Nc, num_each_way, 1, 2048]

    def data_folder(tr_d, tr_l,way,data_len):
        train_files = []
        train_labels = []
        n_way=way
        print('data_folder n_way is',n_way)
        data_temp = tr_d
        label_temp = tr_l
        data_temp = data_temp.cuda().data.cpu().numpy()
        label_temp = label_temp.cuda().data.cpu().numpy()
        print(data_temp.shape)
        print(label_temp.shape)
        data_temp = data_temp.reshape(-1,data_len)
        data_temp = data_temp[:,np.newaxis,:]
        label_temp = label_temp.reshape(-1)
        print(data_temp.shape)
        print(label_temp)

        # examples=tr_d.shape[1]
        # data_len=tr_d.shape[3]
        data_list_train = {}

        for j in range(n_way):
            data_list_train[j] = [data_temp[i] for i, label in enumerate(label_temp) if label == j]
            train_labels.append(j)
            train_files.append(data_list_train[j])
        print('train label is', train_labels)
        train_folders = list(zip(train_files, train_labels))
        return train_folders
        # for i in range(n_way):
        #     data1=[]
        #     for j in range(examples):
        #         data1 = data_temp[i][j]
        #         label = i
        #         data1 = data1.cuda().data.cpu().numpy()
        #         data1 = data1[np.newaxis,:, : ]
        #         # print(data1.shape)
        #         # print(label)
        #         if j == 0:
        #             file = data1
        #         else:
        #             file = np.vstack(
        #                 [file, data1])
        #     train_labels.append(label)
        #     train_files.append(file)
        # train_folders = list(zip(train_files, train_labels))
        # return train_folders

class DataManager(object):
    @staticmethod
    def get_data_loader(self, data_file):
        pass

class SetDataManager(DataManager): # set balance dataset into dataloader
    def __init__(self, signal_size=1024, n_way=5,  n_shot=1,train_num = 50,examples=200,batch_size = 4 ,order = 1,task = 1):
        super(SetDataManager, self).__init__()
        self.signal_size = signal_size
        self.n_way = n_way
        self.train_num = train_num
        self.examples = examples
        self.shot = n_shot
        self.batch_size = batch_size
        self.order = order
        self.task = task

    def get_data_loader(self,shuffle=False,split='train'): #parameters that would change on train/val set

        tr_d, tr_l,  te_d, te_l = DataGenFn().PU_5way(way=self.n_way, order=self.order, examples=self.examples,
                                                      split=self.train_num, data_len=self.signal_size, shuffle=False,
                                                      normalize=True, label=True,task=self.task)
        print(tr_d.shape)
        print(tr_l.shape)

        train_loader = DataGenFn.data_folder(tr_d, tr_l,self.n_way,self.signal_size)
        train_dataset = PU_train(train_loader, class_num=self.n_way)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        test_loader = DataGenFn.data_folder( te_d, te_l,self.n_way,self.signal_size)
        test_dataset = PU_train(test_loader, class_num=self.n_way)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        print("data_loader")

        if split == 'train' or split=='test':

            return train_loader,test_loader

        elif split =='meta':
            support, support_l,_, _ = meta_tr_tasks_utils(tr_d, tr_l,way=self.n_way,shot=self.shot,
                                                                     length=self.signal_size,split='Target')
            _, _, query, query_l = meta_tr_tasks_utils(te_d, te_l,way=self.n_way,shot=self.shot,
                                                                     length=self.signal_size,split='Target')
            train_loader = DataGenFn.data_folder(support, support_l,self.n_way,self.signal_size)
            train_dataset = PU_train(train_loader, class_num=self.n_way)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            test_loader = DataGenFn.data_folder(query, query_l,self.n_way,self.signal_size)
            test_dataset = PU_train(test_loader, class_num=self.n_way)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            return train_loader, test_loader

        else:
            raise ValueError('Unknown split')

class SetDataManager2(DataManager):# set imbalance dataset into dataloader
    def __init__(self, signal_size=1024, n_shot=50,way= 3,train_num = 80,examples = 1000,batch_size = 4,order = 1,
                 data_slice = [100, 5,5],test_slice = [200,200, 200],task = 1):
        super(SetDataManager2, self).__init__()
        self.signal_size = signal_size
        self.train_num = train_num
        self.shot = n_shot
        self.examples = examples
        self.order = order
        self.n_way = way
        self.d_slice = data_slice
        self.t_slice = test_slice
        self.task = task
        self.batch_size = batch_size

    def get_data_loader(self,shuffle=False,split='train',domain = 'Target',num_meta = 4): #parameters that would change on train/val set

        # if self.task == 1:
        #     self.d_slice = self.d_slice
        # elif self.task == 2:
        #     self.d_slice = [100, 10, 10]
        # elif self.task == 3:
        #     self.d_slice = [100, 20, 20]
        # elif self.task == 4:
        #     self.d_slice = [100, 30, 30]

        tr_d, tr_l,  te_d, te_l,me_d,me_l = DataGenFn().PU_5way_imbalance(way=self.n_way, order=self.order,
                             data_len=self.signal_size, shuffle=False,data_slice = self.d_slice,meta_num = num_meta,
                                            test_slice = self.t_slice,  normalize=False, label=True,task =self.task)
        print('im shape:train_data{}|train_label{}|meta label{}'.format(tr_d.shape,tr_l.shape,me_l.shape))
        # print('train data shape',tr_d.shape)
        # print('train label shape',tr_l.shape)
        # print('meta label shape', me_l.shape)
        # qu, qu_l,support, support_l =im_tasks_utils(tr_d, tr_l,5,1024,split = 'Target')
        # su, su_l, query, query_l = im_tasks_utils(te_d, te_l, 5, 1024, split='Target')
        support, support_l, query, query_l = tr_d, tr_l, te_d, te_l
        meta, meta_l =im_tasks_utils(me_d, me_l,num_meta,1024,split = 'meta')
        if split=='train' or split=='test':
            if domain == 'Source':
                train_loader = DataGenFn.data_folder(support, support_l,self.n_way,self.signal_size)
                train_dataset = PU_train(train_loader,class_num=self.n_way)
                test_loader = DataGenFn.data_folder(query ,query_l,self.n_way,self.signal_size)
                test_dataset = PU_train(test_loader,class_num=self.n_way)
                train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
                test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
                return  train_data_loader,test_data_loader
        elif split=='meta' and domain =='Target':
            # num_meta_support = num_meta
            # num_support = support.size()[0]- num_meta_support
            # s_meta,s_meta_l,sup,sup_l = im_tasks_utils( support, support_l,self.n_way,num_meta_support,
            #                                                  num_support, length=self.signal_size ,split='Target')
            sup, sup_l, query, query_l, s_meta, s_meta_l = support, support_l, query, query_l,meta, meta_l
            print('im shape:sup{}|test{}|s_meta label{}'.format(sup.shape, query.shape, s_meta.shape))
            # print('s_meta shape is {}'.format(s_meta.shape))
            # print('sup shape is {}'.format(sup.shape))
            # print('test shape is {}'.format(query.shape))
            meta_train_loader = DataGenFn.data_folder(s_meta,s_meta_l,self.n_way,self.signal_size)
            train_loader = DataGenFn.data_folder(sup, sup_l,self.n_way,self.signal_size)
            test_loader = DataGenFn.data_folder(query, query_l,self.n_way,self.signal_size)
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

            meta_train_data_loader = DataLoader(meta_train_dataset, batch_size=self.batch_size, shuffle=True)
            train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_data_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            return meta_train_data_loader,train_data_loader,test_data_loader

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
        image = self.train_files[idx]
        label = self.train_labels[idx]
        return image, np.int64(label)

    def __len__(self):
        return len(self.train_files)

if __name__ == "__main__":
    d = DataGenFn()
    tr_d, tr_l, te_d, te_l = d.PU_5way(way=5, order=0, examples=200, split=20,
                                      normalize=False, data_len=1024, label=True)
    meta_train_loader = DataGenFn.data_folder( tr_d, tr_l,way=5,data_len=1024)
