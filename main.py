import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset, PadCollate
import base_model
from train import train
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='tmp')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--init_from', type=str, default=None, help='init the model')
    #args = parser.parse_args()
    return parser


if __name__ == '__main__':
    args = parse_args().parse_args()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary)
    eval_dset = VQAFeatureDataset('val', dictionary)
    # train_dset = eval_dset
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    
    model = nn.DataParallel(model).cuda()

    if args.init_from is not None:
        print('Init from: ' + args.init_from)
        init_model = torch.load(args.init_from)
        model.load_state_dict(init_model)

    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=4, collate_fn=PadCollate(dim=0))
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=4, collate_fn=PadCollate(dim=0))
    train(model, train_loader, eval_loader, args.epochs, args.output)
