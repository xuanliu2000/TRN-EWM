import numpy as np
import os
import glob
import argparse
#import backbone
import model.Resnet1d as RE
import model.resnet as re

# model_dict = dict(ResNet18 = RE.resnet18)
model_dict = dict(ResNet18 = re.Resnet18)

def parse_args(script):
    parser = argparse.ArgumentParser(description="meta transfer learning")
    parser.add_argument('--dataset', default='PU_dataset', help='training base model')
    parser.add_argument('--test_dataset', default='EB_dataset', help='test dataset')
    parser.add_argument('--train_task', '-tr_t', default=2, type=int,
                        help='train_task (default: 1)')
    parser.add_argument('--test_task', '-te_t', default=1, type=int,
                        help='test_task (default: 1)')
    parser.add_argument('--num_meta', type=int, default=5, help='The number of meta data for each class.')
    parser.add_argument('--batch_size',type=int, default=5, help='train batch size')

    parser.add_argument('--model', default='ResNet18', help='backbone architecture')
    parser.add_argument('--method', default='baseline', help='baseline/protonet')

    #parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--train_n_way', default=3, type=int, help='class num to classify for training')
    parser.add_argument('--test_n_way', default=3, type=int, help='class num to classify for testing (validation) ')
    parser.add_argument('--n_shot', default=3, type=int, help='number of labeled data in each class, same as n_support')
    # parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ')
    #parser.add_argument('--both', action='store_true', help='use both tuned and untuned model ')
    #parser.add_argument('--freeze_backbone', action='store_true', help='Freeze the backbone network for finetuning')
    parser.add_argument('--save_iter', default=-1, type=int,
                        help='save feature from the model trained in x epoch, use the best model if x is -1')
    parser.add_argument('--models_to_use', '--names-list', nargs='+', default=['EB_dataset', 'PU_dataset'],
                        help='pretained model to use')
    parser.add_argument('--fine_tune_all_models', action='store_true',
                        help='fine-tune each model before selection')  # still required for save_features.py and test.py to find the model path correctly
    parser.add_argument('--fine_tune_epoch', default=100, type=int, help='number of epochs to finetune')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,help='weight decay (default: 5e-4)')

    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')

    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        help='print frequency (default: 10)')

    # parser.add_argument('--gen_examples', default=10, type=int,help ='number of examples to generate (data augmentation)')
    if script == 'train':
        parser.add_argument('--fine_tune'   , action='store_true',  help='fine tuning during training ') 
        parser.add_argument('--num_classes' , default=3, type=int, help='total number of classes in softmax, only used in baseline') #make it larger than the maximum label value in base class
        parser.add_argument('--save_freq'   , default=50, type=int, help='Save frequency')
        parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
        parser.add_argument('--stop_epoch'  , default=400, type=int, help ='Stopping epoch') # for meta-learning methods, each epoch contains 100 episodes
        parser.add_argument('--epochs', default=100, type=int, help='epoch')

    elif script == 'save_features':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        #parser.add_argument('--save_iter', default=-1, type=int,help ='save feature from the model trained in x epoch, use the best model if x is -1')
    elif script == 'test':
        parser.add_argument('--split'       , default='novel', help='base/val/novel') #default novel, but you can also test base/val class accuracy if you want 
        #parser.add_argument('--save_iter', default=-1, type=int,help ='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--adaptation'  , action='store_true', help='further adaptation in test time or not')
        parser.add_argument('--unsup'  , action='store_true', help='unsupervised learning or not')
        parser.add_argument('--unsup_cluster'  , action='store_true', help='unsupervised learning with clustering or not')
    else:
       raise ValueError('Unknown script')
        
    return parser.parse_args()

def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])  #读取epochs
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    return resume_file

def get_best_file(checkpoint_dir):    
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
