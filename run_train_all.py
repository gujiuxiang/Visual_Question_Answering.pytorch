import os
from main import parse_args
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dataset import Dictionary, VQAFeatureDataset, PadCollate
import base_model
import utils

from train_all import train, evaluate

parser = parse_args()
args = parser.parse_args([])
args.epochs = 12

dictionary = Dictionary.load_from_file('data/dictionary.pkl')
train_dset = VQAFeatureDataset('train', dictionary)
eval_dset = VQAFeatureDataset('val', dictionary)
batch_size = args.batch_size
train_loader =  DataLoader(train_dset + eval_dset, batch_size, shuffle=True, num_workers=4, collate_fn=PadCollate(dim=0))

import random
n_models = 6
exp_id_st = 24
for idx in range(exp_id_st, exp_id_st + n_models):
    print("%d/%d"%(idx, exp_id_st + n_models))
    args.seed = random.randint(0, 1000000)
    print(args.seed)
    args.output = 'saved_models_trainall/exp%02d'%idx
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    model = nn.DataParallel(model).cuda()

    train(model, train_loader, None, args.epochs, args.output)
