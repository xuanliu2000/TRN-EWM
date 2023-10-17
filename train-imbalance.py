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
import matplotlib.pyplot as plt
import configs
import argparse
from model.CITN import BaseTrain,VNet,Classifier
from utils.io_utils import model_dict, parse_args, get_resume_file, get_assigned_file,get_best_file
from utils.base_utils import adjust_learning_rate,accuracy
from dataset import EB_dataset,PU_dataset,Dataloader,EB_dataset_imbalance,PU_dataset_imbalance
import utils.plot_utils as pl_u
from sklearn.metrics import precision_score,recall_score,f1_score
from collections import Iterable

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")


def pre_train(base_loader, model, optimization, start_epoch, stop_epoch, params):
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    if optimization == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), params.lr,momentum=params.momentum, weight_decay=params.weight_decay)
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
            y_t,p_t = targets.cuda().data.cpu().numpy(),predicted.cuda().data.cpu().numpy()
            output = outputs.cuda().data.cpu().numpy()
            if batch_idx ==0:
                y_t_o=y_t
                p_t_o=p_t
                output_o =output
            else:
                y_t_o=np.concatenate((y_t_o,y_t))
                p_t_o=np.concatenate((p_t_o,p_t))
                output_o = np.concatenate((output_o, output))
    print(y_t_o)
    print(p_t_o)

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    return accuracy,y_t_o,p_t_o,output_o


def train(train_loader, validation_loader,model, vnet,optimizer_a,optimizer_c,epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    meta_losses = AverageMeter()
    top1 = AverageMeter()
    meta_top1 = AverageMeter()
    model.train()


    for i, (input, target) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)


        meta_model = build_model()

        meta_model.load_state_dict(model.state_dict())

        y_f_hat = meta_model(input_var)
        cost = F.cross_entropy(y_f_hat, target_var, reduce=False)
        cost_v = torch.reshape(cost, (len(cost), 1))

        v_lambda = vnet(cost_v)

        norm_c = torch.sum(v_lambda)

        if norm_c != 0:
            v_lambda_norm = v_lambda / norm_c
        else:
            v_lambda_norm = v_lambda

        l_f_meta = torch.sum(cost_v * v_lambda_norm)
        meta_model.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        meta_lr = params.lr* ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 90)))
        meta_model.update_params(lr_inner=meta_lr, source_params=grads)
        del grads

        input_validation, target_validation = next(iter(validation_loader))
        input_validation_var = to_var(input_validation, requires_grad=False)
        target_validation_var = to_var(target_validation, requires_grad=False)
        y_g_hat = meta_model(input_validation_var)
        l_g_meta = F.cross_entropy(y_g_hat, target_validation_var)
        # l_g_meta.backward(retain_graph=True)
        prec_meta = accuracy(y_g_hat.data, target_validation_var.data, topk=(1,))[0]

        optimizer_c.zero_grad()
        l_g_meta.backward()
        # print(vnet.linear1.weight.grad)
        optimizer_c.step()

        y_f = model(input_var)
        cost_w = F.cross_entropy(y_f, target_var, reduce=False)
        cost_v = torch.reshape(cost_w, (len(cost_w), 1))
        prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]

        with torch.no_grad():
            w_new = vnet(cost_v)
        norm_v = torch.sum(w_new)

        if norm_v != 0:
            w_v = w_new / norm_v
        else:
            w_v = w_new

        l_f = torch.sum(cost_v * w_v)

        losses.update(l_f.item(), input.size(0))
        meta_losses.update(l_g_meta.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))
        meta_top1.update(prec_meta.item(), input.size(0))

        optimizer_a.zero_grad()
        l_f.backward()
        optimizer_a.step()

        if i % params.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Meta_Loss {meta_loss.val:.4f} ({meta_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'meta_Prec@1 {meta_top1.val:.3f} ({meta_top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses,meta_loss=meta_losses, top1=top1,meta_top1=meta_top1))

def build_model():
    model = resnet.Resnet18(out_channel=3)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model

def build_model2():
    model = resnet.Resnet18(out_channel=3)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_freeze_by_id(model, layer_num_last):
    for param in model.parameters():
        param.requires_grad = False
    child_list = list(model.children())[-layer_num_last:]
    if not isinstance(child_list, Iterable):
        child_list = list(child_list)
    for child in child_list:
        print(child)
        for param in child.parameters():
            param.requires_grad = True

if __name__ == '__main__':
    print("HELLO")

    params = parse_args('train')
    print('params.method is {}'.format(params.method))
    np.random.seed(10)  # original was 10

    signal_size = 1024
    # optimization = 'Adam'
    optimization = 'SGD'

    # train_few_shot_params = dict(n_way=params.train_n_way, n_support=params.n_shot)
    # test_few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)

    # train dataset
    if params.dataset == "EB_dataset":
        print('dataset==EB_dataset')
        datamgr = EB_dataset_imbalance.SetDataManager(n_way=params.train_n_way, n_shot=params.n_shot,
                                                      signal_size=signal_size,task = params.train_task)
        base_loader,_ = datamgr.get_data_loader(shuffle=False)
        print("train dataset loaded")
    elif params.dataset == "PU_dataset":
        print('dataset==PU_dataset')
        datamgr = PU_dataset_imbalance.SetDataManager(n_way=params.train_n_way, n_shot=params.n_shot
                                                      ,signal_size=signal_size,task = params.train_task)
        base_loader,_ = datamgr.get_data_loader(shuffle=False)
        print("train dataset loaded")
    else:
        raise ValueError('Unknown dataset')

    print(model_dict[params.model], params.method)

    #pre-train model
    model = BaseTrain(model_dict[params.model], params.num_classes)
    model = model.cuda()

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)

    save_dir = configs.save_dir
    print("WORKING")

    #set checkpoint
    params.checkpoint_dir = '%s/checkpoints/%s_%s/%s_%s_%s_way' % (save_dir, params.dataset, params.train_task,
                                                                   params.model, params.method, params.train_n_way)

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

    # test dataset
    if params.test_dataset == "EB_dataset":
        print('test_dataset==EB_dataset')
        datamgr = EB_dataset_imbalance.SetDataManager2(signal_size=signal_size,batch_size=params.batch_size,
                                             data_slice = [100, 30, 30],test_slice = [200,200, 200],task = params.test_task)
        train_meta_loader, train_loader, test_loader = datamgr.get_data_loader(shuffle=False, split='meta',
                                                                               domain='Target', num_meta=params.num_meta)
    elif params.test_dataset == "PU_dataset":
        print('test_dataset==PU_dataset')
        datamgr = PU_dataset_imbalance.SetDataManager2(signal_size=signal_size,batch_size=params.batch_size,
                                             data_slice = [100, 30, 30],test_slice = [200,200, 200],task = params.test_task)
        train_meta_loader, train_loader, test_loader = datamgr.get_data_loader(shuffle=False, split='meta',
                                                                               domain='Target', num_meta=params.num_meta)
    else:
        raise ValueError('Unknown dataset')


    #check model state
    pretrained_dataset = params.dataset
    checkpoint_dir = '%s/checkpoints/%s_%s/%s_%s_%s_way' % (configs.save_dir, params.dataset, params.train_task,
                                                                   params.model, 'baseline', params.train_n_way)
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
        pretrained_model.load_state_dict(tmp,False)
        print('pretrained_model is loaded')

    #model = pretrained_model.cuda()
    model = build_model().cuda()
    vnet = VNet(1, 10, 1).cuda()

    optimizer_model = torch.optim.SGD(model.params(), params.lr,
                                  momentum=params.momentum, nesterov=params.nesterov,
                                  weight_decay=params.weight_decay)
    optimizer_vnet = torch.optim.SGD(vnet.params(), 1e-5,
                                  momentum=params.momentum, nesterov=params.nesterov,
                                  weight_decay=params.weight_decay)

    # set_freeze_by_id(model, layer_num_last = 20)
    # para = list(filter(lambda p: p.requires_grad, model.parameters()))
    #
    # for param_group in optimizer_model.param_groups:
    #     param_group['params'] = para


    best_acc = 0
    for epoch in range(params.epochs):
        adjust_learning_rate(optimizer_model, epoch)
        train(train_loader,train_meta_loader,model, vnet,optimizer_model,optimizer_vnet,epoch)
        test_acc,y_t_o,p_t_o ,outputs= test(model=model, test_loader=test_loader)
        # if epoch == 3:
        #     print('best accuracy:', best_acc)
        #     print('precision_score:', precision_score(y_t_o,p_t_o, average='macro'))
        #     print('recall_score:', recall_score(y_t_o,p_t_o, average='macro'))
        #     print('f1_score:', f1_score(y_t_o,p_t_o, average='macro'))
        #     pl_u.plot_confusion_matrix(y_t_o,p_t_o)
        #     pl_u.plot_tsne(outputs,p_t_o)
        #     plt.show()

        if test_acc >= best_acc:
            best_acc = test_acc
            y_t_b, p_t_b = y_t_o,p_t_o
            outputs_b = outputs

    pl_u.plot_confusion_matrix(y_t_b, p_t_b)
    pl_u.plot_tsne(outputs, p_t_b)
    #plt.show()
    print('best accuracy:', best_acc)
    print('precision_score:',precision_score(y_t_b, p_t_b, average='macro'))
    print('recall_score:',recall_score(y_t_b, p_t_b, average='macro'))
    print('f1_score:',f1_score(y_t_b, p_t_b, average='macro'))

