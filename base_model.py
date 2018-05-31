import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
from compact_bilinear_pooling import CompactBilinearPooling


class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, cls_att, attr_att, q_net, v_net, cls_net, attr_net, classifier, mcb):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.cls_att = cls_att
        self.attr_att = attr_att
        self.q_net = q_net
        self.v_net = v_net
        self.cls_net = cls_net
        self.attr_net = attr_net
        self.classifier = classifier
        self.mcb = mcb

    def forward(self, v, cls, attr, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        v_emb = (self.v_att(v, q_emb) * v).sum(1) # [batch, v_dim]
        cls_emb = (self.cls_att(cls, q_emb) * cls).sum(1) # [batch, cls_dim]
        attr_emb = (self.attr_att(attr, q_emb) * attr).sum(1) # [batch, cls_dim]
        
        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        cls_repr = self.cls_net(cls_emb)
        attr_repr = self.attr_net(attr_emb)

        #merged_feat = torch.cat([q_repr * v_repr, q_repr * cls_repr, q_repr * attr_repr], dim=1)
        #merged_feat = q_repr * (v_repr + cls_repr + attr_repr)
        merged_feat = self.mcb(q_repr, v_repr + cls_repr + attr_repr)
        logits = self.classifier(merged_feat)

        return logits


def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    cls_att = NewAttention(dataset.cls_dim, q_emb.num_hid, num_hid)
    attr_att = NewAttention(dataset.attr_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    cls_net = FCNet([dataset.cls_dim, num_hid])
    attr_net = FCNet([dataset.attr_dim, num_hid])

    fusion_dim = 16000
    mcb = CompactBilinearPooling(num_hid, num_hid, fusion_dim)
    classifier = SimpleClassifier(fusion_dim, num_hid * 2, dataset.num_ans_candidates, 0.5)
    
    return BaseModel(w_emb, q_emb, v_att, cls_att, attr_att, q_net, v_net, cls_net, attr_net, classifier, mcb)
