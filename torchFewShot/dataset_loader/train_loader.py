from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from PIL import Image
import numpy as np
import os.path as osp
import lmdb
import io
import random

import torch
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class FewShotDataset_train(Dataset):
    """Few shot epoish Dataset

    Returns a task (Xtrain, Ytrain, Xtest, Ytest, Ycls) to classify'
        Xtrain: [nKnovel*nExpemplars, c, h, w].
        Ytrain: [nKnovel*nExpemplars].
        Xtest:  [nTestNovel, c, h, w].
        Ytest:  [nTestNovel].
        Ycls: [nTestNovel].
    """

    def __init__(self,
                 dataset, # dataset of [(img_path, cats), ...].
                 labels2inds, # labels of index {(cats: index1, index2, ...)}.
                 labelIds, # train labels [0, 1, 2, 3, ...,].
                 nKnovel=5, # number of novel categories.
                 nExemplars=1, # number of training examples per novel category.
                 nTestNovel=6*5, # number of test examples for all the novel categories.
                 epoch_size=2000, # number of tasks per eooch.
                 transform=None,
                 load=False,
                 **kwargs
                 ):
        
        self.dataset = dataset
        self.labels2inds = labels2inds
        self.labelIds = labelIds
        self.nKnovel = nKnovel
        self.transform = transform

        self.nExemplars = nExemplars
        self.nTestNovel = nTestNovel
        self.epoch_size = epoch_size
        self.load = load

    def __len__(self):
        return self.epoch_size

    def _sample_episode(self):
        """sampels a training epoish indexs.
        Returns:
            Tnovel: a list of length 'nTestNovel' with 2-element tuples. (sample_index, label)
            Exemplars: a list of length 'nKnovel * nExemplars' with 2-element tuples. (sample_index, label)
        """

        Knovel = random.sample(self.labelIds, self.nKnovel)
        nKnovel = len(Knovel)
        assert((self.nTestNovel % nKnovel) == 0)
        nEvalExamplesPerClass = int(self.nTestNovel / nKnovel)

        Tnovel = []
        Exemplars = []
        for Knovel_idx in range(len(Knovel)):
            ids = (nEvalExamplesPerClass + self.nExemplars)
            img_ids = random.sample(self.labels2inds[Knovel[Knovel_idx]], ids) 

            imgs_tnovel = img_ids[:nEvalExamplesPerClass]
            imgs_emeplars = img_ids[nEvalExamplesPerClass:]

            Tnovel += [(img_id, Knovel_idx) for img_id in imgs_tnovel]
            Exemplars += [(img_id, Knovel_idx) for img_id in imgs_emeplars]
        assert(len(Tnovel) == self.nTestNovel)
        assert(len(Exemplars) == nKnovel * self.nExemplars)
        random.shuffle(Exemplars)
        random.shuffle(Tnovel)

        return Tnovel, Exemplars

    def _creatExamplesTensorData(self, examples):
        """
        Creats the examples image label tensor data.

        Args:
            examples: a list of 2-element tuples. (sample_index, label).

        Returns:
            images: a tensor [nExemplars, c, h, w]
            labels: a tensor [nExemplars]
            cls: a tensor [nExemplars]
        """

        images = []
        labels = []
        cls = []
        for (img_idx, label) in examples:
            img, ids = self.dataset[img_idx]
            if self.load:
                img = Image.fromarray(img)
            else:
                img = read_image(img)
            if self.transform is not None:
                img = self.transform(img)
            images.append(img)
            labels.append(label)
            cls.append(ids)
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)
        cls = torch.LongTensor(cls)
        return images, labels, cls


    def __getitem__(self, index):
        Tnovel, Exemplars = self._sample_episode()
        Xt, Yt, Ytc = self._creatExamplesTensorData(Exemplars)
        Xe, Ye, Yec = self._creatExamplesTensorData(Tnovel)
        return Xt, Yt, Xe, Ye, Yec
