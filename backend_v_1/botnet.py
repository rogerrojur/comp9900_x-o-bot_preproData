# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 20:44:18 2019

@author: 63184
"""

from matplotlib.pylab import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from utils import *
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def logit2pred(logit):
    return torch.argmax(logit, -1)

def raw2gold_pred(pred, cls_idx, sep_decoder):
    pred = pred.tolist()
    gold_pred = []
    for i, pred1 in enumerate(pred):
        ed = cls_idx[i] - 1
        st = sep_decoder[i]
        gold_pred.append(pred1[st: ed])
        #gold_pred.append(pred1[: cls_idx[i] + 1])
    return gold_pred

def raw2gold_truth(truth, cls_idx, sep_decoder):
    truth = truth.tolist()
    gold_truth = []
    for i, truth1 in enumerate(truth):
        ed = cls_idx[i] - 1
        st = sep_decoder[i]
        gold_truth.append(truth1[st: ed])
    return gold_truth

def n_gram_acc(pred, truth, n, elip=1e-13):
    #print('pred:', pred)
    #print('truth:', truth)
    if len(pred) < n and len(truth) < n:
        return 1.0 - elip
    elif len(pred) < n:
        return 0.0 + elip
    elif len(truth) < n:
        return 1.0 - elip
    else:
        pred_list = []
        for st in range(len(pred) - n + 1):
            pred_list.append(tuple(pred[st: st + n]))
        truth_list = []
        for st in range(len(truth) - n + 1):
            truth_list.append(tuple(truth[st: st + n]))
        cnt_pred = defaultdict(int)
        cnt_truth = defaultdict(int)
        for token in pred_list:
            cnt_pred[token] += 1
        for token in cnt_pred:
            cnt_truth[token] += truth_list.count(token)
        sum_pred = sum([v for _, v in cnt_pred.items()])
        sum_truth = sum([v for _, v in cnt_truth.items()])
        res = (sum_truth + elip) / (sum_pred + elip)
        return res
    
def BP(pred, truth, elip=1e-13):
    if len(pred) >= len(truth):
        return 1.0 - elip
    else:
        return math.exp(1.0 - elip  - len(truth) / len(pred))

def belu(y_pred, y_truth, cls_idx, n=4):
    #print(y_pred)
    #print(y_truth)
    #y_pred = y_pred.tolist()
    #y_truth = y_truth.tolist()
    belu_score = []
    for ib, y_pred1 in enumerate(y_pred):
        y_truth1 = y_truth[ib]
        cls_idx1 = cls_idx[ib]
        y_truth1 = y_truth1[:cls_idx1 + 1]
        y_pred1 = y_pred1[:cls_idx1 + 1]
        n_gram_score_ave = 0
        for i in range(1, n + 1):
            n_gram_score_ave += math.log(n_gram_acc(y_pred1, y_truth1, i))
        n_gram_score_ave /= n
        belu_score1 = BP(y_pred1, y_truth1) * math.exp(n_gram_score_ave)
        belu_score.append(belu_score1)
    return belu_score#[bs]    

def gelu_new(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def word2sig(word_size, word_id):
    if word_id < word_size:
        return 0
    else:
        return word_id - word_size + 1#word: 0; pad: 1; sep: 2; cls: 3; eof: 4
    
def word2sig_batch(word_size, word_id_batch, class_num=5):
    res = torch.zeros(*word_id_batch.size(), class_num).to(device)
    label = F.relu(word_id_batch - torch.tensor(word_size - 1).to(device))[:, :, None]
    res = res.scatter_(2, label, 1).to(device)
    return res

def construct_loss_mask(logit, first_sep, cls_location, max_len=512):
    loss_mask = []
    for i in range(len(first_sep)):
        loss_mask1 = torch.zeros(max_len).to(device)
        loss_mask1[first_sep[i]: cls_location[i] - 1] = 1.0
        loss_mask.append(loss_mask1)
    loss_mask = torch.stack(loss_mask, 0).to(device)
    return loss_mask[:, :, None].expand_as(logit)

def construct_qkv_mask(cls_location, max_len=512):
    qkv_mask = []
    for cls_idx in cls_location:
        length = cls_idx + 1
        left = torch.ones(length)
        right = torch.zeros(max_len - length)
        qkv_mask1 = torch.cat([left, right], 0)
        qkv_mask.append(qkv_mask1)
    return torch.stack(qkv_mask, 0).to(device)

def construct_a_mask(first_sep, cls_location, max_len=512):#first not kl's sep
    a_mask = []
    for idx, cls_idx in zip(first_sep, cls_location):
        length = cls_idx + 1
        right_up = torch.ones(length, max_len - length)
        down = torch.ones(max_len - length, max_len)
        if idx + 1 != cls_idx:
            matrix1 = torch.zeros(length, idx + 1)
            matrix2 = torch.ones(idx + 1, length - (idx + 1))
            matrix3 = torch.triu(torch.ones(length - (idx + 1), length - (idx + 1)), diagonal=1)# not include I
            matrix23 = torch.cat([matrix2, matrix3], 0)
            left_up = torch.cat([matrix1, matrix23], 1)
        else:
            left_up = torch.zeros(length, length)
        up = torch.cat([left_up, right_up], 1)
        a_mask1 = torch.cat([up, down], 0)
        penalty = torch.ones_like(a_mask1) * 1e13
        a_mask1 = torch.einsum('hw,hw->hw', penalty, a_mask1)
        a_mask.append(a_mask1)
    return torch.stack(a_mask, 0).to(device)

def schedule_hpu(nlu_hpu, l_hs):
    decode_hpu = []
    st = 0
    for l_hs1 in l_hs:
        decode_hpu.append(nlu_hpu[st: st + l_hs1])
        st += l_hs1
    batch_location = [[] for _ in range(len(l_hs))]
    batch_data = []
    l_hs_matrix = [list(range(l_hs1)) for l_hs1 in l_hs]
    max_l_hs = max(l_hs)
    for i in range(len(l_hs_matrix)):
        if len(l_hs_matrix[i]) < max_l_hs:
            l_hs_matrix[i] += [-1 for _ in range(max_l_hs - len(l_hs_matrix[i]))]
    for i in range(max_l_hs):
        batch_data1 = []
        for j in range(len(l_hs)):
            if l_hs_matrix[j][i] != -1:
                batch_location[j].append((i, len(batch_data1)))
                batch_data1.append(decode_hpu[j][i])
        batch_data.append(batch_data1)
    return batch_data, batch_location

def Botnet_loss(logit, label, cls_idx, sep_decoder_idx):
    #print(label.size())
    loss_mask = construct_loss_mask(logit, sep_decoder_idx, cls_idx)
    loss = 0
    for i in range(logit.size()[1]):
        logit1 = logit[:, i, :].squeeze(-2)
        loss_mask1 = loss_mask[:, i, :].squeeze(-2)
        logit_masked1 = torch.einsum('bw,bw->bw', logit1, loss_mask1)
        label1 = label[:, i].squeeze(-1)
        '''
        print(logit1.size())
        print(loss_mask1.size())
        print(label1.size())
        '''
        loss_temp = F.cross_entropy(logit_masked1, label1)
        loss += loss_temp
        
    return loss

class Botnet(nn.Module):
    def __init__(self, word_emb, w2id, course2id, tf_emb_size, lS_lstm, dr, nb_tf_en, hidden_tf_en, nb_tf_de, hidden_tf_de, heads, max_len=512, sig_class=5):
        super(Botnet, self).__init__()
        self.emb_layer = Embedding_layer(word_emb, w2id, course2id, tf_emb_size, lS_lstm, dr, max_len, sig_class)
        self.tf_encoder = Transformer_encoder(nb_tf_en, tf_emb_size, hidden_tf_en, dr, heads, tf_emb_size // heads, tf_emb_size // heads)
        self.tf_decoder = Transformer_decoder(nb_tf_de, tf_emb_size, hidden_tf_de, dr, heads, tf_emb_size // heads, tf_emb_size // heads)
        self.classify_decoder = Classify_decoder(tf_emb_size, len(w2id))
        
    def forward(self, inputs, target, nlu_hpu, l_hs, course, cls_idx, sep_idx_encoder, sep_idx_decoder):
        x, label = self.emb_layer(inputs, target, nlu_hpu, l_hs, course)
        encode_x = self.tf_encoder(x, cls_idx, sep_idx_encoder)
        decode_x = self.tf_decoder(encode_x, x, cls_idx, sep_idx_decoder)
        y = self.classify_decoder(decode_x)
        
        return y, label

class MultiHeaderAttention(nn.Module):
    def __init__(self, iS, heads, out_size, key_size=None):
        super(MultiHeaderAttention, self).__init__()
        self.heads = heads
        self.out_size = out_size
        self.out_dim = heads * out_size
        self.key_size = key_size if key_size else out_size
        self.iS = iS
        
        self.q_dense = nn.Linear(self.iS, self.key_size * self.heads)
        self.k_dense = nn.Linear(self.iS, self.key_size * self.heads)
        self.v_dense = nn.Linear(self.iS, self.out_dim)
        self.o_dense = nn.Linear(self.out_dim, self.out_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, inputs, q_mask, kv_mask, a_mask):
        '''
        words, from 0 to w - 1, and then w is pad, w + 1 is sep, w + 2 is cls, w + 3 is eof
        
        previous cls will be the next of kl
        
        n tokens question
        m tokens answers
        k course title
        [kl, sep, 1, 2, 3, ..., n, sep, 1, 2, 3, ..., m, eof, sep, cls, pad, pad, ...], input no second eof, target will have it but no kl
        input: [kl, sep, 1, 2, 3, ..., n, sep, 1, 2, 3, ..., m, sep, cls, pad, pad, ...]
        target: [sep, 1, 2, 3, ..., n, sep, 1, 2, 3, ..., m, eof, sep, cls, pad, pad, ...]
        leading: [1, 2, 3, ..., k, sep, cls, pad, pad, ...]
        benefit is eof will only occur at target.
        q_mask: padding is zeros
        kv_mask: padding is zeros
        example:[I, am, a, student, !, [pad], [pad], [pad], [pad], [pad]] = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        
        a_mask: language model mask with big penalty value
        '''
        q, k, v = inputs
        qw = self.q_dense(q)
        qw *= q_mask[:, :, None].expand_as(qw)#(bs, seq_len) -> (bs, seq_len, dim)
        kw = self.k_dense(k)
        kw *= kv_mask[:, :, None].expand_as(kw)
        vw = self.v_dense(v)
        vw *= kv_mask[:, :, None].expand_as(vw)
        
        qw = qw.view(*qw.size()[:-1], self.heads, self.key_size).permute(0, 2, 1, 3)#[bs, head, seq_len, dim]
        kw = kw.view(*kw.size()[:-1], self.heads, self.key_size).permute(0, 2, 1, 3)
        vw = vw.view(*vw.size()[:-1], self.heads, self.out_size).permute(0, 2, 1, 3)
        
        a = torch.einsum('bhqk,bhvk->bhqv', qw, kw) / torch.sqrt(torch.tensor(self.key_size).float().to(device))
        
        a -= a_mask[:, None, :, :].expand_as(a)
        
        a = self.softmax(a)
        
        o = torch.einsum('bhqv,bhvo->bhqo', a, vw)
        
        o = self.o_dense(o.permute(0, 2, 1, 3).contiguous().view(*q.size()[:-1], self.out_dim))#[bs, h, seq_len, dim] -> [bs, seq_len, h, dim] -> [bs, seq_len, h*dim]
        o *= kv_mask[:, :, None].expand_as(o)
        #print(q.size())
        #print(v.size())
        #print(o.size())
        
        return o
    
class FFN(nn.Module):
    def __init__(self, o_dim, hS, dr):
        super(FFN, self).__init__()
        self.out_dim = o_dim
        self.hS = hS
        self.dr = dr
        
        self.dropout = nn.Dropout(self.dr)
        self.layerNorm = nn.LayerNorm(self.out_dim)
        self.fc1 = nn.Linear(self.out_dim, self.hS)
        self.fc2 = nn.Linear(self.hS, self.out_dim)
        
    def forward(self, q, o):
        o = self.layerNorm(self.dropout(o) + q)
        q = o
        o = self.fc1(o)
        o = gelu_new(o)
        o = self.fc2(o)
        o = self.layerNorm(self.dropout(o) + q)
        
        return o
    
class Transformer_block(nn.Module):
    def __init__(self, iS, hS, dr, heads, out_size, key_size=None):
        super(Transformer_block, self).__init__()
        self.atten = MultiHeaderAttention(iS, heads, out_size, key_size)
        self.ffn = FFN(heads * out_size, hS, dr)
        
    def forward(self, inputs, q_mask, kv_mask, a_mask):
        q, _, _ = inputs
        o = self.atten(inputs, q_mask, kv_mask, a_mask)
        o = self.ffn(q, o)
        
        return o# this is the next q for decoder, next qkv for encoder and next qkv for kl
    
class Transformer_encoder(nn.Module):#share parameter
    def __init__(self, lS, iS, hS, dr, heads, out_size, key_size=None):
        super(Transformer_encoder, self).__init__()
        self.tf_block = Transformer_block(iS, hS, dr, heads, out_size, key_size)
        self.lS = lS
        
    def forward(self, inputs, cls_idx, sep_idx):
        qkv_mask = construct_qkv_mask(cls_idx)
        a_mask = construct_a_mask(sep_idx, cls_idx)
        x = inputs
        for i in range(self.lS):
            x = self.tf_block([x, x, x], qkv_mask, qkv_mask, a_mask)
        
        return x

class Transformer_decoder(nn.Module):#share parameter
    def __init__(self, lS, iS, hS, dr, heads, out_size, key_size=None):
        super(Transformer_decoder, self).__init__()
        self.tf_block = Transformer_block(iS, hS, dr, heads, out_size, key_size)
        self.lS = lS
        
    def forward(self, inputs, target, cls_idx, sep_idx):
        qkv_mask = construct_qkv_mask(cls_idx)
        a_mask = construct_a_mask(sep_idx, cls_idx)
        x2 = inputs#from encoder
        x1 = target
        for i in range(self.lS):
            x1 = self.tf_block([x1, x2, x2], qkv_mask, qkv_mask, a_mask)
        
        return x1
    
class Classify_decoder(nn.Module):
    def __init__(self, tf_emb_size, word_size):
        super(Classify_decoder, self).__init__()
        self.classify_layer = nn.Linear(tf_emb_size, word_size)
        
    def forward(self, target):
        logit = self.classify_layer(target)
        
        return logit#[bs, seq_len, word_size]
    
class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0, device=device) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq_len, bsz=None):
        pos_seq = torch.arange(0, pos_seq_len, 1.0, device=device).float()
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[None,:,:].expand(bsz, -1, -1).to(device)
        else:
            return pos_emb[None,:,:].to(device)
        
class Embedding_layer(nn.Module):
    def __init__(self, word_emb, w2id, course2id, tf_emb_size, lS, dr, max_len=512, sig_class=5):#word_emb is from 0 to word_size - 1
        super(Embedding_layer, self).__init__()
        self.word_size, self.emb_size = word_emb.size()
        self.w2id = w2id
        self.course2id = course2id
        course_id_set = set([v for _, v in course2id.items()])
        self.course_nb = len(course_id_set)
        reverse_dict = dict()
        for w, w_id in w2id.items():
            reverse_dict[w_id] = w
        self.word_emb = []
        for w_id in range(self.word_size):
            tmp = reverse_dict[w_id]
            tmp = self.w2id[tmp]
            #print(tmp)
            tmp = word_emb[tmp]
            #print(tmp)
            self.word_emb.append(tmp)
        self.word_emb = torch.stack(self.word_emb, 0).to(device)
        self.tf_emb_size = tf_emb_size
        self.max_len = max_len
        self.sig_class = sig_class
        self.lS = lS
        self.dr = dr
        
        self.pos_emb = PositionalEmbedding(tf_emb_size)
        
        self.course_fc = nn.Linear(self.tf_emb_size + self.course_nb, self.tf_emb_size)
        
        self.sig_fc = nn.Linear(self.sig_class, self.tf_emb_size)
        self.emb_fc = nn.Linear(self.emb_size, self.tf_emb_size)
        self.enc_kl = nn.LSTM(input_size=self.emb_size, hidden_size=int(self.tf_emb_size / self.lS),
                             num_layers=self.lS, batch_first=True,
                             dropout=self.dr, bidirectional=True)
        self.cell_fc = nn.Linear(self.tf_emb_size, self.tf_emb_size * self.lS)
        self.hidden_fc = nn.Linear(self.tf_emb_size, self.tf_emb_size * self.lS)
        
    def sentence2id(self, sentence):
        return [self.w2id[w] for w in sentence]
    
    def course_emb(self, course, seq_len, class_num=42):
        return torch.zeros(course.size()[0], class_num).scatter_(1, course[:, None], 1)[:, None, :].expand(-1, seq_len, -1).to(device)
        
    def forward(self, inputs, target, nlu_hpu, l_hs, course):
        for i in range(len(nlu_hpu)):
            nlu_hpu[i] = self.sentence2id(nlu_hpu[i])
        for i in range(len(inputs)):
            inputs[i] = self.sentence2id(inputs[i])
        for i in range(len(target)):
            target[i] = self.sentence2id(target[i])
        batch_data, batch_location = schedule_hpu(nlu_hpu, l_hs)
        batch_hs = []
        for b in range(len(batch_data)):
            batch_hs1 = []
            for ib in range(len(batch_data[b])):
                batch_hs1.append(len(batch_data[b][ib]))
                batch_data[b][ib] = self.word_emb[batch_data[b][ib]]
            batch_hs.append(batch_hs1)
            max_cur = max(batch_hs1)
            for ib in range(len(batch_data[b])):
                cur_len, cur_dim = batch_data[b][ib].size()
                if cur_len < max_cur:
                    pad = torch.zeros(max_cur - cur_len, cur_dim).to(device)
                    batch_data[b][ib] = torch.cat([batch_data[b][ib], pad], 0)
            batch_data[b] = torch.stack(batch_data[b], 0).to(device)
        kl = encode(self.enc_kl, batch_data[0], batch_hs[0], return_hidden=False, hc0=None, last_only=True).squeeze(1)
        #print(kl.size())
        if len(batch_data) > 1:
            for b in range(1, len(batch_data)):
                tmp_h = []
                tmp_c = []
                for j, batch_location1 in enumerate(batch_location):
                    if b < len(batch_location1):
                        tmp_h.append(kl[j])
                        tmp_c.append(kl[j])
                tmp_h = torch.stack(tmp_h, 0).to(device)
                tmp_c = torch.stack(tmp_c, 0).to(device)
                hidden = self.hidden_fc(tmp_h)
                hidden = hidden.view(hidden.size()[0], self.lS * 2, self.tf_emb_size // 2).transpose(0, 1).contiguous()
                cell = self.cell_fc(tmp_c)
                cell = cell.view(cell.size()[0], self.lS * 2, self.tf_emb_size // 2).transpose(0, 1).contiguous()
                tmp_kl = encode(self.enc_kl, batch_data[b], batch_hs[b], return_hidden=False, hc0=(hidden, cell), last_only=True).squeeze(1)
                for j, batch_location1 in enumerate(batch_location):
                    if b < len(batch_location1):
                        kl[j] = tmp_kl[batch_location1[b][1]]
        
        inputs = torch.tensor(inputs).to(device)
        #print(inputs.size())
        target_label = torch.tensor(target).to(device)
        
        inputs_sig = self.sig_fc(word2sig_batch(self.word_size, inputs).to(device))
        #print(inputs_sig.size())
        #target_sig = word2sig_batch(self.word_size, target_label).to(device)
        
        inputs = self.emb_fc(self.word_emb[inputs].to(device))
        #print(inputs.size())
        #target = self.word_emb[target].to(device)
        
        inputs += inputs_sig
        #target += target_sig
        
        course = [self.course2id[e] for e in course]
        
        #print(course)
        
        c_emb = self.course_emb(torch.tensor(course), self.max_len)
        #print(c_emb.size())
        
        kl = kl[:, None, :]
        
        inputs = torch.cat([kl, inputs], 1)
        
        #print(inputs.size())
        
        p_emb_i = self.pos_emb(inputs.size()[1], inputs.size()[0])
        #p_emb_t = self.pos_emb(target.size()[1], target.size()[0])
        
        inputs += p_emb_i
        #target += p_emb_t
        
        inputs = self.course_fc(torch.cat([inputs, c_emb], -1))
        #target = self.course_fc(torch.cat([target, c_emb], -1))
        
        return inputs, target_label