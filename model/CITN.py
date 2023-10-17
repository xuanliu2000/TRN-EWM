import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init
import utils.base_utils as base_utils


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class VNet(MetaModule):
    def __init__(self, input, hidden1, output):
        super(VNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        # x = self.linear2(x)
        # x = self.relu1(x)
        out = self.linear2(x)
        return F.sigmoid(out)


class BaseTrain(nn.Module):
    def __init__(self, model_func, num_class, loss_type='softmax'):
        super(BaseTrain, self).__init__()
        self.feature = model_func()

        loss_type == 'softmax'
        self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
        self.classifier.bias.data.fill_(0)

        self.loss_type = loss_type  # 'softmax' #'dist'
        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.top1 = base_utils.AverageMeter()

    def forward(self, x):
        x = Variable(x.cuda())
        out = self.feature.forward(x)
        scores = self.classifier.forward(out)
        return scores

    def forward_loss(self, x, y):
        y = Variable(y.cuda())

        scores = self.forward(x)

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item() * 100 / (y.size(0) + 0.0), y.size(0))

        return self.loss_fn(scores, y)

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss = 0
        for i, (x, y) in enumerate(train_loader):
            # print(i)
            optimizer.zero_grad()
            loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()
            if i % print_freq == 0:
                # print(optimizer.state_dict()['param_groups'][0]['lr'])
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Top1 Val {:f} | Top1 Avg {:f}'.format(epoch, i,
                                                                                                        len(train_loader),
                                                                                                        avg_loss / float(
                                                                                                            i + 1),
                                                                                                        self.top1.val,
                                                                                                        self.top1.avg))

    def test_loop(self, val_loader):
        return -1  # no validation, just save model during iteration

def build_model():
    model = ResNet32(args.dataset == 'cifar10' and 10 or 100)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model

class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()

        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

