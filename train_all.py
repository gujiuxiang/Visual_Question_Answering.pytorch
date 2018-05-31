import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
from tqdm import tqdm
import json
import cPickle as pkl
import numpy as np

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, num_epochs, output):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    
    for epoch in range(num_epochs):
        print('epoch %d/%d'%(epoch, num_epochs))
        for i, (v, cls, attr, b, q, a, _) in enumerate(tqdm(train_loader)):
            v = Variable(v).cuda()
            cls = Variable(cls).cuda()
            attr = Variable(attr).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()

            pred = model(v, cls, attr, b, q, a)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

    model_path = os.path.join(output, 'model.pth')
    torch.save(model.state_dict(), model_path)


def evaluate(model, dataloader):
    model.train(False)
    pred_list = []
    q_id_list = []
    for v, cls, attr, b, q, a, q_id in dataloader:
        v = Variable(v, volatile=True).cuda()
        cls = Variable(cls, volatile=True).cuda()
        attr = Variable(attr, volatile=True).cuda()
        b = Variable(b, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        pred = model(v, cls, attr, b, q, None)
        pred = nn.functional.sigmoid(pred)
        pred_list.append(pred.data.cpu().numpy())
        q_id_list.extend(q_id)
        

    pred_list = np.concatenate(pred_list, axis=0)
    model.train(True)
    return q_id_list, pred_list
