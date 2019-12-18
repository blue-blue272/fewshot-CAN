from __future__ import absolute_import

from .train_loader import FewShotDataset_train
from .test_loader import FewShotDataset_test


__loader_factory = {
        'train_loader': FewShotDataset_train,
        'test_loader': FewShotDataset_test,
}



def get_names():
    return list(__loader_factory.keys()) 


def init_loader(name, *args, **kwargs):
    if name not in list(__loader_factory.keys()):
        raise KeyError("Unknown model: {}".format(name))
    return __loader_factory[name](*args, **kwargs)

