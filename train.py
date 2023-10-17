import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
from model import Resnet1d,resnet
import copy

import configs
import argparse
from model.CITN import BaseTrain,VNet
from utils.io_utils import model_dict, parse_args, get_resume_file, get_assigned_file,get_best_file
from utils.base_utils import adjust_learning_rate,accuracy
from dataset import EB_dataset,PU_dataset,Dataloader



# kwargs = {'num_workers': 1, 'pin_memory': True}
# use_cuda = not parse_args.no_cuda and torch.cuda.is_available()
# torch.manual_seed(parse_args.seed)
# device = torch.device("cuda" if use_cuda else "cpu")

#device = torch.device('cuda:0')
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")


def pre_train(base_loader, model, optimization, start_epoch, stop_epoch, params):
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    for epoch in range(start_epoch, stop_epoch):
        # print(epoch)
        model.train()
        model.train_loop(epoch, base_loader, optimizer)
        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

    return model

def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss +=F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    return accuracy


def train(train_loader,train_meta_loader,model, vnet,optimizer_model,optimizer_vnet,epoch):
    print('\nEpoch: %d' % epoch)

    train_loss = 0
    meta_loss = 0

    train_meta_loader_iter = iter(train_meta_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        #print('batch_idx is',batch_idx)
        model.train()
        inputs, targets = inputs.to(device), targets.to(device)
        meta_model = build_model2()
        meta_model.load_state_dict(model.state_dict())
        outputs = meta_model(inputs)
        #print(outputs)

        cost = F.cross_entropy(outputs, targets, reduce=False)  #计算meta-model交叉熵
        cost_v = torch.reshape(cost, (len(cost), 1))
        #print(cost_v)
        v_lambda = vnet(cost_v.data)
        #print(v_lambda)
        l_f_meta = torch.sum(cost_v * v_lambda)/len(cost_v)
        #print('lf_meta is {}'.format(l_f_meta))
        meta_model.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)#,allow_unused=True
        #print(grads)
        meta_lr = params.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 100)))   # For ResNet32
        meta_model.update_params(lr_inner=meta_lr, source_params=grads)
        del grads

        try:
            inputs_val, targets_val = next(train_meta_loader_iter)
        except StopIteration:
            train_meta_loader_iter = iter(train_meta_loader)
            inputs_val, targets_val = next(train_meta_loader_iter)
        inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
        y_g_hat = meta_model(inputs_val)
        l_g_meta = F.cross_entropy(y_g_hat, targets_val)
        prec_meta = accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0]


        optimizer_vnet.zero_grad()
        l_g_meta.backward()
        optimizer_vnet.step()

        outputs = model(inputs)
        cost_w = F.cross_entropy(outputs, targets, reduce=False)
        cost_v = torch.reshape(cost_w, (len(cost_w), 1))
        prec_train = accuracy(outputs.data, targets.data, topk=(1,))[0]

        with torch.no_grad():
            w_new = vnet(cost_v)

        loss = torch.sum(cost_v * w_new)/len(cost_v)

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()


        train_loss += loss.item() #item精度高
        meta_loss += l_g_meta.item()


        if (batch_idx + 1) % 1 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f' % (
                      (epoch + 1), params.epochs, batch_idx + 1, len(train_loader.dataset)/params.batch_size, (train_loss / (batch_idx + 1)),
                      (meta_loss / (batch_idx + 1)), prec_train, prec_meta))

def build_model():
    model = resnet.Resnet18()

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model

def build_model2():
    model = resnet.Resnet18()

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model

if __name__ == '__main__':
    print("HELLO")

    params = parse_args('train')
    print('params.method is {}'.format(params.method))
    np.random.seed(10)  # original was 10

    signal_size = 1024
    optimization = 'Adam'

    # train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
    # test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)

    # train dataset
    if params.dataset == "EB_dataset":
        print('dataset==EB_dataset')
        datamgr = EB_dataset.SetDataManager(n_way=params.train_n_way, n_shot=params.n_shot,signal_size=signal_size)
        # base_loader = datamgr.get_data_loader(aug=params.train_aug)
        base_loader = datamgr.get_data_loader(shuffle=False)
        print("loaded")
    elif params.dataset == "PU_dataset":
        print('dataset==PU_dataset')
        datamgr = PU_dataset.SetDataManager(n_way=params.train_n_way, n_shot=params.n_shot,signal_size=signal_size)
        # base_loader = datamgr.get_data_loader(aug=params.train_aug)
        base_loader = datamgr.get_data_loader(shuffle=False)
        print("loaded")
    else:
        raise ValueError('Unknown dataset')

    print(model_dict[params.model], params.method)

    #pre-train model
    model = BaseTrain(model_dict[params.model], params.num_classes)
    #print(model)
    model = model.cuda()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)

    save_dir = configs.save_dir
    print("WORKING")

    #set checkpoint
    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (save_dir, params.dataset, params.model, params.method)

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

        start_epoch = params.start_epoch
        stop_epoch = params.stop_epoch
        print(params.checkpoint_dir)

        #pre-train
        pre_model = pre_train(base_loader, model, optimization, start_epoch, stop_epoch, params)
    else:
        print('{} is already exist'.format(params.checkpoint_dir))

    ##########################################################################################


    #set imbanced data

    # datamgr = Dataloader.SetDataManager2(signal_size=signal_size)
    # train_meta_loader, train_loader, test_loader  = datamgr.get_data_loader(shuffle=False,split='meta', domain ='Target',num_meta=2)

    if params.test_dataset == "EB_dataset":
        print('test_dataset==EB_dataset')
        datamgr = Dataloader.SetDataManager(signal_size=signal_size)
        # base_loader = datamgr.get_data_loader(aug=params.train_aug)
        base_loader = datamgr.get_data_loader(shuffle=False)
    elif params.test_dataset == "PU_dataset":
        print('test_dataset==PU_dataset')
        datamgr = Dataloader.SetDataManager2(signal_size=signal_size)
        train_meta_loader, train_loader, test_loader = datamgr.get_data_loader(shuffle=False, split='meta',
                                                                               domain='Target', num_meta=2)
    else:
        raise ValueError('Unknown dataset')


    #check model state
    pretrained_dataset = params.dataset
    checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, pretrained_dataset, params.model, "baseline")
    if params.save_iter != -1:
        modelfile = get_assigned_file(checkpoint_dir, params.save_iter)
    else:
        modelfile = get_best_file(checkpoint_dir)

    print(modelfile)
    # create model ,load state

    if modelfile is not None:
        tmp = torch.load(modelfile)
        state = tmp['state']
        state_temp = copy.deepcopy(tmp)
        pretrained_model = build_model()
        print(pretrained_model.params())
        #print(pretrained_model)
        pretrained_model.load_state_dict(tmp,False)
        print('pretrained_model is loaded')
        print(pretrained_model.params())
        #state_temp = copy.deepcopy(state)

        # state_keys = list(state_temp.keys())
        # for _, key in enumerate(state_keys):
        #     if "feature." in key:
        #         newkey = key.replace("feature.",
        #                              "")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
        #         state_temp[newkey] = state_temp.pop(key)
        #     else:
        #         state_temp.pop(key)

        #pretrained_model.load_state_dict(state_temp)

    model = pretrained_model.cuda()
   # model = build_model().cuda()
    vnet = VNet(1, 100, 1).cuda()

    optimizer_model = torch.optim.SGD(model.params(), params.lr,momentum=params.momentum, weight_decay=params.weight_decay)
    optimizer_vnet = torch.optim.Adam(vnet.params(), 1e-3,weight_decay=1e-4)

    best_acc = 0
    for epoch in range(params.epochs):
        adjust_learning_rate(optimizer_model, epoch)
        train(train_loader,train_meta_loader,model, vnet,optimizer_model,optimizer_vnet,epoch)
        test_acc = test(model=model, test_loader=test_loader)
        if test_acc >= best_acc:
            best_acc = test_acc

    print('best accuracy:', best_acc)