# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:50:55 2019

@author: 63184
"""

import json

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
                cur_obj = cur_obj.replace('image2=imread(\'C:\\Users\\ABCD\\threads.png\',0)\\n\\nshould read image2=cv2.imread(\'C:\\Users\\ABCD\\threads.png\',0)', '')
                cur_obj = cur_obj.replace('File \\\"C:\\Users\\user\\Desktop\\assignment2\\radio_index2xy.py\\\"', '')
                for i in range(10):
                    cur_obj = cur_obj.replace('\\' + str(i), '[NUM]')
                #print()
                #print(cur_obj)
                #print()
                #print(cur_obj[824:827] == 'A\\B')
                #print(cur_obj[473:476])
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
                cur_obj += '\\n'
            else:
                good_line = clean_line.replace('\\0', '[NUL]').replace('\\x', '\\u00').replace(' =\\', '').replace('\\s', '[STR]').replace('\\[', '[OPB]').replace('\\]', '[CLB]').replace('\\\\\"%s\\\\\"', '[UNK]').replace('\\ ', '[UNK]').replace('\\i', '[UNK]').replace('\\df', '[UNK]').replace('\\ef', '[UNK]').replace('\\d', '[UNK]').replace('¯\\_(ツ)_/¯', '[UNK]').replace('\\png', '[PNG]').replace('\\ge', '[UNK]').replace('V\\', '').replace('\\{0}', '-{0}').replace('\'\\\'', '[SLA]').replace('D:\\UNSW\\S2\\COMP9024\\Assignment', '[PAT]').replace('\\.', '').replace('B\\A', 'B-A').replace('A\\B', 'A-B').replace('\\argmax', 'argmax').replace('\\prod', 'prod').replace('\\Sigma', 'Sigma').replace('\\math', 'math').replace('\\begin', 'begin').replace('\\the', 'the').replace('\\log', 'log').replace('\\mu', 'mu').replace('\\c', 'c').replace('\\end', 'end').replace('\\left', 'left').replace('\\right', 'right').replace('\\lam', 'lam').replace('\\[UNK]', '[UNK]').replace('\\pi', 'pi').replace('\\gamma', 'gamma').replace('\\{', '{').replace('\\}', '}').replace('\\frac', 'frac').replace('\\alpha', 'alpha').replace('\\beta', 'beta').replace(' _/\\_', '')
                cur_obj += good_line
        print(cnt, 'number of data have loaded to the dictionary!')
    return myDict

def save_obj_dict(path, myDict):
    jsObj = json.dumps(myDict)
    fileObj = open(path, 'w')
    fileObj.write(jsObj)
    fileObj.close()
    print('data have been saved to', path)
    
def load_dict(path):
    myDict = dict()
    with open(path) as f:
        myDict = json.load(f)
    return myDict
    
if __name__ == '__main__':
    save_obj_dict('id2content.json', load_json_list('webcms3json/messages.json'))