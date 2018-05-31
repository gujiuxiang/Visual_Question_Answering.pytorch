from __future__ import print_function
import os
import json
import cPickle
import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
import collections
from tqdm import tqdm


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    answer.pop('image_id')
    answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer}
    return entry


def _load_dataset(dataroot, name, img_id2val):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    if name == 'test':
        question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s2015_questions.json' % name)
        questions = sorted(json.load(open(question_path))['questions'],
                           key=lambda x: x['question_id'])

        entries = []
        for question in questions:
            answer = {'image_id':'', 'question_id':'', 'labels':[], 'scores':[]}
            img_id = question['image_id']
            entries.append(_create_entry(img_id2val[img_id], question, answer))
        
        return entries
        
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s2014_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        entries.append(_create_entry(img_id2val[img_id], question, answer))

    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val', 'test']

        dataroot='data'
        
        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary

        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s36_imgid2idx.pkl' % name)))
        #print('loading features from h5 file')
        #h5_path = os.path.join(dataroot, '%s36.hdf5' % name)
        #with h5py.File(h5_path, 'r') as hf:
        #    self.features = np.array(hf.get('image_features'))
        #    self.spatials = np.array(hf.get('spatial_features'))
        

        self.entries = _load_dataset(dataroot, name, self.img_id2idx)
        
        print('loading features from numpy files')
        if name == 'test':
            self.image_feat_dir = 'bottom-up-adaptK/test2015'
        else:
            self.image_feat_dir = 'bottom-up-adaptK/trainval'
  
        self.tokenize()
        self.tensorize()
        self.f_dim = 2048
        self.cls_dim = 1601
        self.attr_dim = 401
        self.v_dim = self.f_dim #self.features.size(2)
        self.s_dim = 6 #self.spatials.size(2)

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = padding + tokens
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        #self.features = torch.from_numpy(self.features)
        #self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        #features = self.features[entry['image']]
        #spatials = self.spatials[entry['image']]
        
        features = torch.zeros((1, self.f_dim))
        cls_prob = torch.zeros((1, self.cls_dim))
        attr_prob = torch.zeros((1, self.attr_dim))
        spatials = torch.from_numpy(np.array([0]))

        image_id = entry['image_id']
        image_feat_path = os.path.join(self.image_feat_dir, '%d.npy'%image_id)
        if os.path.exists(image_feat_path):
            data = np.load(image_feat_path)[()]
            if 'features' in data.keys():
                features = torch.from_numpy(data['features'].astype(np.float32))
                cls_prob = torch.from_numpy(data['cls_prob'].astype(np.float32))
                attr_prob = torch.from_numpy(data['attr_prob'].astype(np.float32))
            else:
                pass
                #print('no features:%d'%image_id)
        else:
            pass
            #print('Not exist: %s'%image_feat_path)

        question_id = entry['question_id']
        question = entry['q_token']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)

        return features, cls_prob, attr_prob, spatials, question, target, question_id

    def __len__(self):
        return len(self.entries)

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size).type_as(vec)], dim=dim)

class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        assert isinstance(batch[0], collections.Sequence)
        transposed = zip(*batch)
        pad_batch = []
        for samples in transposed:
            if type(samples[0]) is int:
                pad_batch.append(samples)
                continue
            pad_dim = self.dim
            max_len = max(map(lambda x: x.shape[pad_dim], samples))
            samples = map(lambda x: pad_tensor(x, pad=max_len, dim=pad_dim), samples)
            samples = torch.stack(samples, dim=0)
            pad_batch.append(samples)
        return pad_batch

    def __call__(self, batch):
        return self.pad_collate(batch)