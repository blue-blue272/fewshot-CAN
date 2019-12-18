from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn


def open_all_layers(model):
    """
    Open all layers in model for training.
    """
    model.train()
    for p in model.parameters():
        p.requires_grad = True


def open_specified_layers(model, open_layers):
    """
    Open specified layers in model for training while keeping 
    other layers frozen.

    Args:
    - model (nn.Module): neural net model.
    - open_layers (list): list of layers names.
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    for layer in open_layers:
        assert hasattr(model, layer), "'{}' is not an attribute of the model, please provide the correct name".format(layer)

    for name, module in model.named_children():
        if name in open_layers:
            #print(module)
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False



def adjust_learning_rate(optimizer, iters, LUT):
    # decay learning rate by 'gamma' for every 'stepsize'
    for (stepvalue, base_lr) in LUT:
        if iters < stepvalue:
            lr = base_lr
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_lambda(iters, LUT):
    for (stepvalue, base_lambda) in LUT:
        if iters < stepvalue:
            lambda_xent = base_lambda
            break
    return lambda_xent


def set_bn_to_eval(m):
    # 1. no update for running mean and var
    # 2. scale and shift parameters are still trainable
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def count_num_param(model):
    num_param = sum(p.numel() for p in model.parameters()) / 1e+06
    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
        # we ignore the classifier because it is unused at test time
        num_param -= sum(p.numel() for p in model.classifier.parameters()) / 1e+06
    return num_param


def one_hot(labels_train):
    """
    Turn the labels_train to one-hot encoding.
    Args:
        labels_train: [batch_size, num_train_examples]
    Return:
        labels_train_1hot: [batch_size, num_train_examples, K]
    """
    labels_train = labels_train.cpu()
    nKnovel = 1 + labels_train.max()
    labels_train_1hot_size = list(labels_train.size()) + [nKnovel,]
    labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
    labels_train_1hot = torch.zeros(labels_train_1hot_size).scatter_(len(labels_train_1hot_size) - 1, labels_train_unsqueeze, 1)
    return labels_train_1hot
