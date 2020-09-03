import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet12 import resnet12
from cam import CAM


def one_hot(labels_train):
    labels_train = labels_train.cpu()
    nKnovel = 5
    labels_train_1hot_size = list(labels_train.size()) + [nKnovel,]
    labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
    labels_train_1hot = torch.zeros(labels_train_1hot_size).scatter_(len(labels_train_1hot_size) - 1, labels_train_unsqueeze, 1)
    return labels_train_1hot


class Model(nn.Module):
    def __init__(self, scale_cls, iter_num_prob=35.0/75, num_classes=64):
        super(Model, self).__init__()
        self.scale_cls = scale_cls
        self.iter_num_prob = iter_num_prob

        self.base = resnet12()
        self.cam = CAM()

        self.nFeat = self.base.nFeat
        self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1) 

    def test(self, ftrain, ftest):
        ftest = ftest.mean(4)
        ftest = ftest.mean(4)
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
        scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
        return scores

    def forward(self, xtrain, xtest, ytrain, ytest):
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)

        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)

        ftrain = f[:batch_size * num_train]
        ftrain = ftrain.view(batch_size, num_train, -1) 
        ftrain = torch.bmm(ytrain, ftrain)
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.view(batch_size, -1, *f.size()[1:])
        ftest = f[batch_size * num_train:]
        ftest = ftest.view(batch_size, num_test, *f.size()[1:]) 
        ftrain, ftest = self.cam(ftrain, ftest)
        ftrain = ftrain.mean(4)
        ftrain = ftrain.mean(4)

        if not self.training:
            return self.test(ftrain, ftest)

        ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
        ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
        ftrain_norm = ftrain_norm.unsqueeze(4)
        ftrain_norm = ftrain_norm.unsqueeze(5)
        cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)
        cls_scores = cls_scores.view(batch_size * num_test, *cls_scores.size()[2:])

        ftest = ftest.view(batch_size, num_test, K, -1)
        ftest = ftest.transpose(2, 3) 
        ytest = ytest.unsqueeze(3) 
        ftest = torch.matmul(ftest, ytest) 
        ftest = ftest.view(batch_size * num_test, -1, 6, 6)
        ytest = self.clasifier(ftest)
        return ytest, cls_scores

    def helper(self, ftrain, ftest, ytrain):
        b, n, c, h, w = ftrain.size()
        k = ytrain.size(2)

        ytrain_transposed = ytrain.transpose(1, 2)
        ftrain = torch.bmm(ytrain_transposed, ftrain.view(b, n, -1))
        ftrain = ftrain.div(ytrain_transposed.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.view(b, -1, c, h, w)

        ftrain, ftest = self.cam(ftrain, ftest)
        ftrain = ftrain.mean(-1).mean(-1)
        ftest = ftest.mean(-1).mean(-1)

        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
        scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
        return scores

    def test_transductive(self, xtrain, xtest, ytrain, ytest):
        iter_num_prob = self.iter_num_prob
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)

        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)

        ftrain = f[: batch_size*num_train].view(batch_size, num_train, *f.size()[1:])
        ftest = f[batch_size*num_train:].view(batch_size, num_test, *f.size()[1:])
        cls_scores = self.helper(ftrain, ftest, ytrain)

        num_images_per_iter = int(num_test * iter_num_prob)
        num_iter = num_test // num_images_per_iter

        for i in range(num_iter):
            max_scores, preds = torch.max(cls_scores, 2)
            chose_index = torch.argsort(max_scores.view(-1), descending=True)
            chose_index = chose_index[: num_images_per_iter * (i + 1)]

            ftest_iter = ftest[0, chose_index].unsqueeze(0)
            preds_iter = preds[0, chose_index].unsqueeze(0)
            preds_iter = one_hot(preds_iter).cuda()

            ftrain_iter = torch.cat((ftrain, ftest_iter), 1)
            ytrain_iter = torch.cat((ytrain, preds_iter), 1)
            cls_scores = self.helper(ftrain_iter, ftest, ytrain_iter)

        return cls_scores




if __name__ == '__main__':
    torch.manual_seed(0)

    net = Model(scale_cls=7)
    net.eval()

    x1 = torch.rand(1, 5, 3, 84, 84)
    x2 = torch.rand(1, 75, 3, 84, 84)
    y1 = torch.rand(1, 5, 5)
    y2 = torch.rand(1, 75, 5)

    y1 = net.test_transductive(x1, x2, y1, y2)
    print(y1.size())
