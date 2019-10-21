from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
sys.path.append('./torchFewShot')

from torchFewShot.models.net import Model
from torchFewShot.data_manager import DataManager
from torchFewShot.losses import CrossEntropyLoss
from torchFewShot.optimizers import init_optimizer

from torchFewShot.utils.iotools import save_checkpoint, check_isfile
from torchFewShot.utils.avgmeter import AverageMeter
from torchFewShot.utils.logger import Logger
from torchFewShot.utils.torchtools import one_hot, adjust_learning_rate

parser = argparse.ArgumentParser(description='Test image model with 5-way classification')
# Datasets
parser.add_argument('-d', '--dataset', type=str, default='miniImageNet_load')
parser.add_argument('--load', default=True)
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=84,
                    help="height of an image (default: 84)")
parser.add_argument('--width', type=int, default=84,
                    help="width of an image (default: 84)")
# Optimization options
parser.add_argument('--train-batch', default=4, type=int,
                    help="train batch size")
parser.add_argument('--test-batch', default=8, type=int,
                    help="test batch size")
# Architecture
parser.add_argument('--num_classes', type=int, default=64)
parser.add_argument('--scale_cls', type=int, default=7)
parser.add_argument('--save-dir', type=str, default='')
parser.add_argument('--resume', type=str, default='', metavar='PATH')
# FewShot settting
parser.add_argument('--nKnovel', type=int, default=5,
                    help='number of novel categories')
parser.add_argument('--nExemplars', type=int, default=1,
                    help='number of training examples per novel category.')
parser.add_argument('--train_nTestNovel', type=int, default=6 * 5,
                    help='number of test examples for all the novel category when training')
parser.add_argument('--train_epoch_size', type=int, default=1200,
                    help='number of episodes per epoch when training')
parser.add_argument('--nTestNovel', type=int, default=15 * 5,
                    help='number of test examples for all the novel category')
parser.add_argument('--epoch_size', type=int, default=2000,
                    help='number of batches per epoch')
# Miscs
parser.add_argument('--phase', default='test', type=str)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu-devices', default='1', type=str)

args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()

    sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")
        
    print('Initializing image data manager')
    dm = DataManager(args, use_gpu)
    trainloader, testloader = dm.return_dataloaders()

    model = Model(scale_cls=args.scale_cls, num_classes=args.num_classes)
    # load the model
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    print("Loaded checkpoint from '{}'".format(args.resume))

    if use_gpu:
        model = model.cuda()

    test(model, testloader, use_gpu)


def test(model, testloader, use_gpu):
    accs = AverageMeter()
    test_accuracies = []
    model.eval()

    with torch.no_grad():
        for batch_idx , (images_train, labels_train, images_test, labels_test) in enumerate(testloader):
            if use_gpu:
                images_train = images_train.cuda()
                images_test = images_test.cuda()

            end = time.time()

            batch_size, num_train_examples, channels, height, width = images_train.size()
            num_test_examples = images_test.size(1)

            labels_train_1hot = one_hot(labels_train).cuda()
            labels_test_1hot = one_hot(labels_test).cuda()

            cls_scores = model(images_train, images_test, labels_train_1hot, labels_test_1hot)
            cls_scores = cls_scores.view(batch_size * num_test_examples, -1)
            labels_test = labels_test.view(batch_size * num_test_examples)

            _, preds = torch.max(cls_scores.detach().cpu(), 1)
            acc = (torch.sum(preds == labels_test.detach().cpu()).float()) / labels_test.size(0)
            accs.update(acc.item(), labels_test.size(0))

            gt = (preds == labels_test.detach().cpu()).float()
            gt = gt.view(batch_size, num_test_examples).numpy() #[b, n]
            acc = np.sum(gt, 1) / num_test_examples
            acc = np.reshape(acc, (batch_size))
            test_accuracies.append(acc)

    accuracy = accs.avg
    test_accuracies = np.array(test_accuracies)
    test_accuracies = np.reshape(test_accuracies, -1)
    stds = np.std(test_accuracies, 0)
    ci95 = 1.96 * stds / np.sqrt(args.epoch_size)
    print('Accuracy: {:.2%}, std: :{:.2%}'.format(accuracy, ci95))

    return accuracy


if __name__ == '__main__':
    main()
