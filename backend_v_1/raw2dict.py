# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:50:55 2019

@author: 63184
"""

import json
import re
from collections import defaultdict
from utils import *
import numpy as np

def load_json_list(path):
    myDict = dict()
    with open(path, encoding='utf-8') as f:
        cur_obj = ''
        cnt = 0
        for line in f:
            clean_line = line.strip()
            if cur_obj == '' and line[:-1] == ']':
                break
            elif cur_obj == '' and line[:-1] == '[':
                continue
            elif cur_obj == '' and line[:-1] == '{':
                cur_obj = '{'
            elif line[:-1] == '},' and '\"resource_id\"' in cur_obj:
                idx = cur_obj.find('Traceback (most recent call last)')
                if idx != -1:
                    cur_obj1 = cur_obj[:idx]
                    cur_obj2 = cur_obj[idx:]
                    cur_obj_rest_idx = cur_obj2.find('\",\"resource_id\"')
                    cur_obj2 = cur_obj2[cur_obj_rest_idx:]
                    cur_obj = cur_obj1 + cur_obj2
                cur_obj += '}'
                cur_obj = cur_obj.replace('image2=imread(\'C:\\Users\\ABCD\\threads.png\',0)\\n\\nshould read image2=cv2.imread(\'C:\\Users\\ABCD\\threads.png\',0)', ' ')
                cur_obj = cur_obj.replace('File \\\"C:\\Users\\user\\Desktop\\assignment2\\radio_index2xy.py\\\"', ' ')
                for i in range(10):
                    cur_obj = cur_obj.replace('\\' + str(i), ' ')
                for i in range(ord('a'), ord('z') + 1):
                    cur_obj = cur_obj.replace('\\' + chr(i), ' ')
                for i in range(ord('A'), ord('Z') + 1):
                    cur_obj = cur_obj.replace('\\' + chr(i), ' ')
                #print()
                cur_obj = cur_obj.replace('fA\\B?', ' ').replace('\\ ', ' ').replace(' \\ ', ' ')
                #print(cur_obj)
                #print()
                #print(cur_obj[643:647] == '] \ ')
                #print(cur_obj[643:647])
                cur_dict = json.loads(cur_obj, strict=False)
                cur_dict['mesg_id'] = int(cur_dict['mesg_id'])
                if 'parent_id' in cur_dict:
                    cur_dict['parent_id'] = int(cur_dict['parent_id'])
                cur_dict['posted_by'] = int(cur_dict['posted_by'])
                cur_dict['resource_id'] = int(cur_dict['resource_id'])
                key = cur_dict['mesg_id']
                myDict[key] = cur_dict
                cnt += 1
                #for k in cur_dict:
                #    print(k, ':', cur_dict[k])
                cur_obj = ''
            elif cur_obj != '' and clean_line == '':
                cur_obj += ' '
            else:
                good_line = clean_line.replace('\\0', ' ').replace('\\x', ' ').replace(' =\\', ' ').replace('\\s', ' ').replace('\\[', ' ').replace('\\]', ' ').replace('\\\\\"%s\\\\\"', ' ').replace('\\ ', ' ').replace('\\i', ' ').replace('\\df', ' ').replace('\\ef', ' ').replace('\\d', ' ').replace('¯\\_(ツ)_/¯', ' ').replace('\\png', ' ').replace('\\ge', ' ').replace('V\\', ' ').replace('\\{0}', ' ').replace('\'\\\'', ' ').replace('D:\\UNSW\\S2\\COMP9024\\Assignment', ' ').replace('\\.', ' ').replace('B\\A', ' ').replace('A\\B', ' ').replace('\\argmax', 'argmax').replace('\\prod', 'prod').replace('\\Sigma', 'Sigma').replace('\\math', 'math').replace('\\begin', 'begin').replace('\\the', 'the').replace('\\log', 'log').replace('\\mu', 'mu').replace('\\c', 'c').replace('\\end', 'end').replace('\\left', 'left').replace('\\right', 'right').replace('\\lam', 'lam').replace('\\[UNK]', ' ').replace('\\pi', 'pi').replace('\\gamma', 'gamma').replace('\\{', '{').replace('\\}', '}').replace('\\frac', 'frac').replace('\\alpha', 'alpha').replace('\\beta', 'beta').replace(' _/\\_', ' ')
                cur_obj += good_line
        print(cnt, 'number of data have loaded to the dictionary!')
    return myDict

def save_obj_dict(path, myDict):
    jsObj = json.dumps(myDict)
    fileObj = open(path, 'w')
    fileObj.write(jsObj)
    fileObj.close()
    print('data have been saved to', path)

def get_all_head(myDict):#the leading sentence
    heads = []
    for key in myDict:
        cur_dict = myDict[key]
        if 'parent_id' not in cur_dict:
            heads.append(key)
    return heads

def print_chain(myDict, h_id, word_set, sig):
    h = h_id
    end = False
    chain = []
    while not end:
        chain.append((h, tokenize(myDict[h]['message_body'], word_set, sig)))
        if 'children_id' in myDict[h]:
            h = str(myDict[h]['children_id'])
        else:
            end = True
    pre = '-1'
    new_chain = []
    for e in chain:
        if myDict[e[0]]['posted_by'] != pre:
            new_chain.append([e])
            pre = myDict[e[0]]['posted_by']
        else:
            new_chain[-1].append(e)
    for e in new_chain:
        print(e)
        
def cat_chain(chain, myDict):
    pre = '-1'
    new_chain = []
    for e in chain:
        if myDict[e]['posted_by'] != pre:
            new_chain.append([e])
            pre = myDict[e]['posted_by']
        else:
            new_chain[-1].append(e)
    return new_chain
            

def generate_dataset(myDict):
    head_list = get_all_head(myDict)
    chains = []
    for h in head_list:
        cur = [h]
        while 'children_id' in myDict[h]:
            h = str(myDict[h]['children_id'])
            cur.append(h)
        cur = cat_chain(cur, myDict)
        chains.append(cur)
    dataset = []
    sep_dataset = []
    for chain in chains:
        if len(chain) == 1:
            continue
        cur = []
        for i in range(len(chain) - 1):
            pairs = (chain[i], chain[i + 1])
            cur.append(pairs)
            sep_dataset.append(pairs)
        dataset.append(cur)
    return dataset, sep_dataset

def get_all_words(path, myDict, sig_path):
    #sig = get_sig(sig_path)
    words = set()
    for key in myDict:
        cur_mesg = myDict[key]['message_body'].lower()
        cur_course = myDict[key]['course'].lower()
        L = re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', cur_mesg)
        #L = tokenize_raw(cur_mesg, sig)
        L = set([e for e in L if e and len(e) < 100])
        words |= L
        L = re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', cur_course)
        #L = tokenize_raw(cur_course, sig)
        L = set([e for e in L if e and len(e) < 100])
        words |= L
    w_dict = dict()
    for w in words:
        w_dict[w] = w
    jsObj = json.dumps(w_dict)
    fileObj = open(path, 'w')
    fileObj.write(jsObj)
    fileObj.close()
    print(len(w_dict), 'number of words have been saved to', path)
    
def get_str_digit():
    my_set = set()
    for i in range(ord('a'), ord('z') + 1):
        my_set.add(chr(i))
    for i in range(ord('A'), ord('Z') + 1):
        my_set.add(chr(i))
    for i in range(10):
        my_set.add(str(i))
    return my_set

def check_not_str_digit(my_str, str_digit_set):
    if all([e not in str_digit_set for e in my_str]):
        return True
    else:
        return False

def tokenize_raw(sentence, sig):
    if not sentence:
        return []
    tokens = sentence.lower().split()
    tokens = [token for token in tokens if token]
    #print(tokens)
    words = []
    for token in tokens:
        tmp = []
        tmp_str = ''
        for c in token:
            if c in sig:
                if tmp_str:
                    tmp.append(tmp_str)
                    tmp_str = ''
                    tmp.append(c)
                else:
                    tmp.append(c)
            else:
                tmp_str += c
        if tmp_str:
            tmp.append(tmp_str)
        for w in tmp:
            words.append(w)
    return words
    
def load_w_dict(w_path, glove_path, target_path, new_w_path, sig_path):
    w_dict = dict()
    with open(w_path) as f:
        w_dict = json.load(f)
    glove_dict = dict()
    #str_digit_set = get_str_digit()
    sig = get_sig(sig_path)
    cnt = 0
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                L = line.split()
                key = L[0]
                if key in w_dict or key in sig:
                    cnt += 1
                    value = L[1:]
                    value = [float(value1) for value1 in value]
                    glove_dict[key] = value
    jsObj = json.dumps(glove_dict)
    fileObj = open(target_path, 'w')
    fileObj.write(jsObj)
    fileObj.close()
    new_w_dict = dict()
    for i, key in enumerate(glove_dict):
        new_w_dict[key] = i
    jsObj = json.dumps(new_w_dict)
    fileObj = open(new_w_path, 'w')
    fileObj.write(jsObj)
    fileObj.close()
    print(cnt, 'embedding have been saved to', target_path)
    print(cnt, 'words have been saved to', new_w_path)
    
def create_sig_dict(w_path, sig_path):
    w_dict = dict()
    with open(w_path) as f:
        w_dict = json.load(f)
    sd_set = get_str_digit()
    sig_dict = dict()
    cnt = 0
    for key in w_dict:
        if len(key) == 1 and check_not_str_digit(key, sd_set):
            cnt += 1
            sig_dict[key] = key
    jsObj = json.dumps(sig_dict)
    fileObj = open(sig_path, 'w')
    fileObj.write(jsObj)
    fileObj.close()
    print(cnt, 'done!')
    
def cnt_dataset(dataset, myDict, course_path):
    cnt_dict = defaultdict(int)
    for e in dataset:
        cnt_dict[len(e)] += 1
        if len(e) == 14:
            print(e)
    L = []
    for kv in cnt_dict.items():
        L.append(kv)
    L.sort(key=lambda x: x[0])
    for e in L:
        print(e)
    course_dict = defaultdict(int)
    for e in dataset:
        course_dict[myDict[e[0][0][0]]['course']] += 1
    print(len(course_dict), 'of courses')
    L = []
    for kv in course_dict.items():
        L.append(kv)
    L.sort(key=lambda x: x[1], reverse=True)
    course_dict = dict()
    for i, e in enumerate(L):
        course_dict[e[0]] = i
    jsObj = json.dumps(course_dict)
    fileObj = open(course_path, 'w')
    fileObj.write(jsObj)
    fileObj.close()
    print('courses have been saved to', course_path)
    for e in L:
        print(e)
        
def print_dup(sep_dataset):
    cnt_dict = defaultdict(int)
    for e in sep_dataset:
        for ee in e:
            cnt_dict[len(ee)] += 1
    L = []
    for kv in cnt_dict.items():
        L.append(kv)
    L.sort(key=lambda x: x[1], reverse=True)
    for e in L:
        print(e)
        
def data_length(myDict, word_set, sig):
    max_len = -1
    len_cnt_dict = defaultdict(int)
    for key in myDict:
        cur_len = len(tokenize(myDict[key]['message_body'], word_set, sig))
        len_cnt_dict[cur_len] += 1
        if cur_len > max_len:
            max_len = cur_len
    L = []
    for kv in len_cnt_dict.items():
        L.append(kv)
    L.sort(key=lambda x: x[1], reverse=True)
    for e in L[:50]:
        print(e)
    print('max length:', max_len)
    
def split_data(dataset, ftrain, fdev, train_percentage=0.9):
    total_size = len(dataset)
    arr = np.arange(total_size)
    np.random.shuffle(arr)
    train_size = int(total_size * train_percentage)
    dev_size = total_size - train_size
    train_dataset = []
    dev_dataset = []
    for i, e in enumerate(arr):
        if i < train_size:
            train_dataset.append(dataset[e])
        else:
            dev_dataset.append(dataset[e])
    print(train_size, 'numbers of train data and', dev_size, 'numbers of dev data.')
    train_dict = dict()
    for i, e in enumerate(train_dataset):
        train_dict[i] = e
    dev_dict = dict()
    for i, e in enumerate(dev_dataset):
        dev_dict[i] = e
    jsObj = json.dumps(train_dict)
    fileObj = open(ftrain, 'w')
    fileObj.write(jsObj)
    fileObj.close()
    jsObj = json.dumps(dev_dict)
    fileObj = open(fdev, 'w')
    fileObj.write(jsObj)
    fileObj.close()
    
def reconstruct_dataset(fname):
    dataset = dict()
    with open(fname) as f:
        dataset = json.load(f)
    dataset_list = []
    for _, v in dataset.items():
        for i, e in enumerate(v):
            cur_dict = dict()
            cur_dict['previous'] = [mid[0] for mid in v[:i]]
            if cur_dict['previous']:
                cur_dict['previous'] = [q for p in cur_dict['previous'] for q in p]
            cur_dict['pair'] = e
            dataset_list.append(cur_dict)
    dataset = dict()
    for i, e in enumerate(dataset_list):
        dataset[i] = e
    jsObj = json.dumps(dataset)
    fileObj = open(fname, 'w')
    fileObj.write(jsObj)
    fileObj.close()
    print(fname, 'has been reconstructed! There are', len(dataset), 'numbers of data!')

if __name__ == '__main__':
    save_obj_dict('id2content.json', load_json_list('webcms3json/messages.json'))
    data = load_dict('id2content.json')
    get_all_words('raw_words.json', data, 'sig.json')
    load_w_dict('raw_words.json', 'glove.6B/glove.6B.300d.txt', 'glove_emb.json', 'words.json', 'sig.json')
    dataset, sep_dataset = generate_dataset(data)
    print('dataset length:', len(dataset))
    print('sep_dataset length:', len(sep_dataset))
    cnt_dataset(dataset, data, 'course.json')
    print_dup(sep_dataset)
    word_set = get_word_set('words.json')
    sig = get_sig('sig.json')
    print_chain(data, '2690981', word_set, sig)
    data_length(data, word_set, sig)
    split_data(dataset, 'train.json', 'dev.json')
    reconstruct_dataset('train.json')
    reconstruct_dataset('dev.json')