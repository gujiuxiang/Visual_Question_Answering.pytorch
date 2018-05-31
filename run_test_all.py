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

dictionary = Dictionary.load_from_file('data/dictionary.pkl')
test_dset = VQAFeatureDataset('test', dictionary)
batch_size = args.batch_size
test_loader =  DataLoader(test_dset, batch_size, shuffle=False, num_workers=4, collate_fn=PadCollate(dim=0))

import numpy as np
from tqdm import trange
import os
n_models = 18
pred_list_sum = 0

models_root_dir = 'saved_models_trainall'

for idx in trange(n_models):
    print(idx)
    args.seed = idx
    args.output = '%s/exp%02d'%(models_root_dir, idx)
    args.init_from = os.path.join(args.output, 'model.pth')

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(test_dset, args.num_hid).cuda()
    model.w_emb.init_embedding('data/glove6b_init_300d.npy')
    model = nn.DataParallel(model).cuda()

    print('Init from: ' + args.init_from)
    init_model = torch.load(args.init_from)
    model.load_state_dict(init_model)
    
    q_id_list, pred_list = evaluate(model, test_loader)
    pred_list_sum += pred_list

out_prob = '%s/test2015' % models_root_dir
np.save(out_prob, pred_list)
out_prob = '%s/test2015_sum' % models_root_dir
np.save(out_prob, pred_list_sum)

import cPickle as pkl
from tqdm import tqdm
import json
label2ans = pkl.load(open('data/cache/trainval_label2ans.pkl'))
results = []
for x, y in zip(q_id_list, np.argmax(pred_list_sum, 1)):
    y = label2ans[y]
    results.append({'question_id': x, 'answer': y})
json.dump(results, open('%s/test2015_predictions.json' % models_root_dir, 'w'))

q_id2result = {x['question_id']:x for x in results}
test_dev_set = json.load(open('data/v2_OpenEnded_mscoco_test-dev2015_questions.json'))['questions']
test_dev_q_ids = [x['question_id'] for x in test_dev_set]
results_test_dev = [q_id2result[x] for x in test_dev_q_ids]
json.dump(results_test_dev, open('%s/test-dev_predictions.json' % models_root_dir, 'w'))
