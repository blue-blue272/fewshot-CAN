from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import pickle


def load_data(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo)
    return data


def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


class miniImageNet_load(object):
    """
    Dataset statistics:
    # 64 * 600 (train) + 16 * 600 (val) + 20 * 600 (test)
    """
    dataset_dir = '/home/houruibing/code/few-shot/MiniImagenet/'

    def __init__(self, **kwargs):
        super(miniImageNet_load, self).__init__()
        self.train_dir = os.path.join(self.dataset_dir, 'miniImageNet_category_split_train_phase_train.pickle')
        self.val_dir = os.path.join(self.dataset_dir, 'miniImageNet_category_split_val.pickle')
        self.test_dir = os.path.join(self.dataset_dir, 'miniImageNet_category_split_test.pickle')

        self.train, self.train_labels2inds, self.train_labelIds = self._process_dir(self.train_dir)
        self.val, self.val_labels2inds, self.val_labelIds = self._process_dir(self.val_dir)
        self.test, self.test_labels2inds, self.test_labelIds = self._process_dir(self.test_dir)

        self.num_train_cats = len(self.train_labelIds)
        num_total_cats = len(self.train_labelIds) + len(self.val_labelIds) + len(self.test_labelIds)
        num_total_imgs = len(self.train + self.val + self.test)

        print("=> MiniImageNet loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # cats | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(len(self.train_labelIds), len(self.train)))
        print("  val      | {:5d} | {:8d}".format(len(self.val_labelIds),   len(self.val)))
        print("  test     | {:5d} | {:8d}".format(len(self.test_labelIds),  len(self.test)))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_cats, num_total_imgs))
        print("  ------------------------------")

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _get_pair(self, data, labels):
        assert (data.shape[0] == len(labels))
        data_pair = []
        for i in range(data.shape[0]):
            data_pair.append((data[i], labels[i]))
        return data_pair

    def _process_dir(self, file_path):
        dataset = load_data(file_path)
        data = dataset['data']
        print(data.shape)
        labels = dataset['labels']
        data_pair = self._get_pair(data, labels)
        labels2inds = buildLabelIndex(labels)
        labelIds = sorted(labels2inds.keys())
        return data_pair, labels2inds, labelIds

if __name__ == '__main__':
    miniImageNet_load()
