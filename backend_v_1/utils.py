# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 21:50:10 2019

@author: 63184
"""

import json
from matplotlib.pylab import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dict(path):
    myDict = dict()
    with open(path) as f:
        myDict = json.load(f)
    for key in myDict:
        cur_dict = myDict[key]
        if 'parent_id' in cur_dict:
            myDict[str(cur_dict['parent_id'])]['children_id'] = cur_dict['mesg_id']
    return myDict

def generate_perm_inv(perm):
    # Definitly correct.
    perm_inv = zeros(len(perm), dtype=int32)
    for i, p in enumerate(perm):
        perm_inv[int(p)] = i

    return perm_inv

def get_word_set(path):
    word_dict = dict()
    with open(path) as f:
        word_dict = json.load(f)
    word_set = set([key for key in word_dict])
    return word_set

def get_sig(path):
    sig = dict()
    with open(path) as f:
        sig = json.load(f)
    sig = set([key for key in sig])
    return sig

def tokenize(sentence, word_set, sig):
    if not sentence:
        return []
    tokens = sentence.lower().split()
    tokens = [token for token in tokens if token]
    #print(tokens)
    words = []
    for token in tokens:
        if token in word_set:
            words.append(token)
        else:
            tmp = []
            tmp_str = ''
            for c in token:
                if c in sig:
                    if tmp_str:
                        tmp.append(tmp_str)
                        tmp_str = ''
                        tmp.append(c)
                else:
                    tmp_str += c
            if tmp_str:
                tmp.append(tmp_str)
            for w in tmp:
                if w and w in word_set:
                    words.append(w)
    return words

def load_course(path):
    course_dict = dict()
    with open(path) as f:
        course_dict = json.load(f)
    result_dict = dict()
    for k, v in course_dict.items():
        if v > 40:
            result_dict[k] = 41#other
        else:
            result_dict[k] = v
    return result_dict

def load_w2id(path):
    w2id = dict()
    with open(path) as f:
        w2id = json.load(f)
    return w2id

def load_emb(path, word2id, emb_dim=300):
    emb = []
    word_dict = dict()
    reverse_dict = dict()
    word_size = len(word2id)
    for w, w_id in word2id.items():
        reverse_dict[w_id] = w
    with open(path) as f:
        word_dict = json.load(f)
    for i in range(word_size):
        v = word_dict[reverse_dict[i]]
        emb_dim = len(v)
        emb.append(torch.tensor([float(v1) for v1 in v]).float())
    #print(emb[3].size())
    for i in range(4):
        tmp = torch.ones(emb_dim) * i
        tmp = tmp.float()
        #print(tmp.size())
        emb.append(tmp)#pad is 0, sep is 1, cls is 2, eof is 3
    word2id['[pad]'] = word_size
    word2id['[sep]'] = word_size + 1
    word2id['[cls]'] = word_size + 2
    word2id['[eof]'] = word_size + 3
    word_set = set([w for w in word2id])
    emb = torch.stack(emb, 0).to(device)
    
    return emb, word2id, word_set

def encode(lstm, wemb_l, l, return_hidden=False, hc0=None, last_only=False):
    """ [batch_size, max token length, dim_emb]
    """
    bS, mL, eS = wemb_l.shape


    # sort before packking
    l = array(l)#l is the list of how many tokens in this question, so it is a list of int
    perm_idx = argsort(-l)#sort the indices from large to small
    perm_idx_inv = generate_perm_inv(perm_idx)#so now the largest element is in the position when the value in this list is 0

    # pack sequence
    #reconstruct the order of wemb_l and l from large to small length and then pack sequence

    packed_wemb_l = nn.utils.rnn.pack_padded_sequence(wemb_l[perm_idx, :, :],
                                                      l[perm_idx],
                                                      batch_first=True)
    # Time to encode
    if hc0 is not None:
        hc0 = (hc0[0][:, perm_idx], hc0[1][:, perm_idx])

    # ipdb.set_trace()
    packed_wemb_l = packed_wemb_l.float() # I don't know why..
    packed_wenc, hc_out = lstm(packed_wemb_l, hc0)#packed_wenc is (seq_length, batch, hiddenSize * nbDirection)
    hout, cout = hc_out

    # unpack
    wenc, _l = nn.utils.rnn.pad_packed_sequence(packed_wenc, batch_first=True)

    if last_only:
        # Take only final outputs for each columns.
        wenc = wenc[tuple(range(bS)), l[perm_idx] - 1]  # [batch_size, dim_emb]
        wenc.unsqueeze_(1)  # [batch_size, 1, dim_emb]

    wenc = wenc[perm_idx_inv]



    if return_hidden:
        # hout.shape = [batch, seq_len, num_of_layer * number_of_direction ] w/ batch_first.. w/o batch_first? I need to see.
        hout = hout[:, perm_idx_inv].to(device)
        cout = cout[:, perm_idx_inv].to(device)  # Is this correct operation?

        return wenc, hout, cout
    else:
        return wenc

def encode_hpu(lstm, wemb_hpu, l_hpu, l_hs):
    wenc_hpu, hout, cout = encode( lstm,
                                   wemb_hpu,
                                   l_hpu,
                                   return_hidden=True,
                                   hc0=None,
                                   last_only=True )

    wenc_hpu = wenc_hpu.squeeze(1)
    bS_hpu, mL_hpu, eS = wemb_hpu.shape
    hS = wenc_hpu.size(-1)

    wenc_hs = wenc_hpu.new_zeros(len(l_hs), max(l_hs), hS)
    wenc_hs = wenc_hs.to(device)

    # Re-pack according to batch.
    # ret = [B_NLq, max_len_headers_all, dim_lstm]
    # sum(hs) = len(l_hpu) so l_hpu 是展开来的col 长度列表 也即是 每一个col多少字， l_hs是这个玩意的大小是有多少个col
    st = 0
    #print('l_hpu: ', len(l_hpu), sum(l_hs), '; wenc_hs: ', wenc_hs.size(), '; wenc_hpu: ', wenc_hpu.size(), '; wemb_hpu: ', wemb_hpu.size())
    for i, l_hs1 in enumerate(l_hs):#l_hs记录的每个batch的长度，长度的意思是有多少句
        wenc_hs[i, :l_hs1] = wenc_hpu[st:(st + l_hs1)]
        st += l_hs1

    return wenc_hs

def get_part(part_list, data_dict, word_set, sig, max_part=224):
    part_raw = []
    for part in part_list:
        token_list = tokenize(data_dict[part]['message_body'], word_set, sig)
        if len(token_list) == 0:
            token_list = tokenize("I don't know.", word_set, sig)
        part_raw += token_list
    part_result = []
    if len(part_raw) > max_part:
        first = max_part // 2
        last = max_part - first
        part_result += part_raw[:first]
        part_result += part_raw[-last:]
    else:
        part_result = part_raw
    return part_result

def generate_inputs(t, data_dict, word_set, sig, max_part=224, max_len=512):
    l_hs = []
    nlu_hpu = []
    seq_input = []
    seq_target = []
    course = []
    first_sep_encoder = []
    first_sep_decoder = []
    cls_idx = []
    max_input_len = max_len - 1
    max_target_len = max_len
    for t1 in t:
        l_hs.append(len(t1['previous']) + 1)
        course.append(data_dict[t1['pair'][0][0]]['course'])
        nlu_hpu.append(tokenize(data_dict[t1['pair'][0][0]]['course'], word_set, sig))
        if t1['previous']:
            for tid in t1['previous']:
                token_list = tokenize(data_dict[tid]['message_body'], word_set, sig)
                if len(token_list) == 0:
                    token_list = tokenize("I don't know.", word_set, sig)
                nlu_hpu.append(token_list)
        first_part = get_part(t1['pair'][0], data_dict, word_set, sig, max_part)
        second_part = get_part(t1['pair'][1], data_dict, word_set, sig, max_part)
        seq_input1 = []
        seq_target1 = []
        seq_input1.append('[sep]')
        seq_target1.append('[sep]')
        seq_input1 += first_part
        seq_target1 += first_part
        seq_input1.append('[sep]')
        seq_target1.append('[sep]')
        first_sep_decoder.append(len(seq_input1))
        seq_input1 += second_part
        seq_target1 += second_part
        
        seq_target1.append('[eof]')
        
        seq_input1.append('[sep]')
        seq_target1.append('[sep]')
        first_sep_encoder.append(len(seq_input1))
        seq_input1.append('[cls]')
        seq_target1.append('[cls]')
        cls_idx.append(len(seq_input1))
        
        seq_input1 += ['[pad]' for _ in range(max_input_len - len(seq_input1))]
        seq_target1 += ['[pad]' for _ in range(max_target_len - len(seq_target1))]
        seq_input.append(seq_input1)
        seq_target.append(seq_target1)
    
    return l_hs, nlu_hpu, seq_input, seq_target, course, cls_idx, first_sep_encoder, first_sep_decoder

def get_loader(train_path, dev_path, bS, shuffle_train=True, shuffle_dev=False):
    train_dict = dict()
    dev_dict = dict()
    with open(train_path) as f:
        train_dict = json.load(f)
    with open(dev_path) as f:
        dev_dict = json.load(f)
    data_train = [v for _, v in train_dict.items()]
    data_dev = [v for _, v in dev_dict.items()]
    train_loader = DataLoader(
        batch_size=bS,
        dataset=data_train,
        shuffle=shuffle_train,
        num_workers=0,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    dev_loader = DataLoader(
        batch_size=bS,
        dataset=data_dev,
        shuffle=shuffle_dev,
        num_workers=0,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    return train_loader, dev_loader        

if __name__ ==  '__main__':
    word_set = get_word_set('words.json')
    sig = get_sig('sig.json')
    print(tokenize('COMP1521 18s2 Computer Systems Fundamentals', word_set, sig))