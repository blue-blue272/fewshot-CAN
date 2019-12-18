import argparse
import torchFewShot

def argument_parser():

    parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
    # ************************************************************
    # Datasets (general)
    # ************************************************************
    parser.add_argument('-d', '--dataset', type=str, default='tieredImageNet')
    parser.add_argument('--load', default=False)

    parser.add_argument('-j', '--workers', default=4, type=int,
                        help="number of data loading workers (default: 4)")
    parser.add_argument('--height', type=int, default=84,
                        help="height of an image (default: 84)")
    parser.add_argument('--width', type=int, default=84,
                        help="width of an image (default: 84)")

    # ************************************************************
    # Optimization options
    # ************************************************************
    parser.add_argument('--optim', type=str, default='sgd',
                        help="optimization algorithm (see optimizers.py)")
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help="initial learning rate")
    parser.add_argument('--weight-decay', default=5e-04, type=float,
                        help="weight decay (default: 5e-04)")

    parser.add_argument('--max-epoch', default=80, type=int,
                        help="maximum epochs to run")
    parser.add_argument('--start-epoch', default=0, type=int,
                        help="manual epoch number (useful on restarts)")
    parser.add_argument('--stepsize', default=[60], nargs='+', type=int,
                        help="stepsize to decay learning rate")
    parser.add_argument('--LUT_lr', default=[(20, 0.1), (40, 0.01), (60, 0.001), (80, 0.0001)],
                        help="multistep to decay learning rate")

    parser.add_argument('--train-batch', default=4, type=int,
                        help="train batch size")
    parser.add_argument('--test-batch', default=8, type=int,
                        help="test batch size")


    # ************************************************************
    # Architecture settings
    # ************************************************************
    parser.add_argument('--num_classes', type=int, default=351)
    parser.add_argument('--scale_cls', type=int, default=7)

    # ************************************************************
    # Miscs
    # ************************************************************
    parser.add_argument('--save-dir', type=str, default='./result/tieredImageNet/CAM/5-shot-v2/')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--gpu-devices', default='0', type=str)

    # ************************************************************
    # FewShot settting
    # ************************************************************
    parser.add_argument('--nKnovel', type=int, default=5,
                        help='number of novel categories')
    parser.add_argument('--nExemplars', type=int, default=5,
                        help='number of training examples per novel category.')

    parser.add_argument('--train_nTestNovel', type=int, default=6 * 5,
                        help='number of test examples for all the novel category when training')
    parser.add_argument('--train_epoch_size', type=int, default=13980,
                        help='number of batches per epoch when training')
    parser.add_argument('--nTestNovel', type=int, default=15 * 5,
                        help='number of test examples for all the novel category')
    parser.add_argument('--epoch_size', type=int, default=2000,
                        help='number of batches per epoch')

    parser.add_argument('--phase', default='test', type=str,
                        help='use test or val dataset to early stop')
    parser.add_argument('--seed', type=int, default=1)

    return parser

