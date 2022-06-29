import json
from utils.config import *
import ast

from utils.utils_general import *


def read_langs(file_name, max_line = None):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr, tmp_kb_arr = [], [], [], [], []
    last_user_utter, dialog_history = [], []
    max_resp_len = 0
    
    with open(file_name) as fin:
        cnt_lin, sample_counter = 1, 1
        for line in fin:
            line = line.strip()
            if line:

                nid, line = line.split(' ', 1)
                if '\t' in line:
                    u, r, gold_ent = line.split('\t')
                    gen_u = generate_memory(u, "$u", str(nid)) 
                    context_arr += gen_u
                    conv_arr += gen_u

                    last_user_utter = gen_u
                    dialog_history += gen_u
                
                    gold_ent = ast.literal_eval(gold_ent)
                    ent_index = list(set(gold_ent))

                    tmp_entity_dic = generate_tmp_entity_dic(tmp_kb_arr)

                    ptr_index = []
                    for key in r.split():
                        index = [loc for loc, val in enumerate(context_arr) if (val[0] == key and key in ent_index)]
                        if (index):
                            index = max(index) 
                        else: 
                            index = len(context_arr)
                        ptr_index.append(index) 

                    selector_index = [1 if (word_arr[0] in ent_index or word_arr[0] in r.split()) else 0 for word_arr in context_arr] + [1]

                    sketch_response = generate_template(r, gold_ent, kb_arr, tmp_entity_dic)
                    
                    data_detail = {
                        'context_arr':list(context_arr+[['$$$$']*MEM_TOKEN_SIZE]), 
                        'response':r,
                        'sketch_response':sketch_response,
                        'ptr_index':ptr_index+[len(context_arr)],
                        'selector_index':selector_index,
                        'ent_index':ent_index,
                        'conv_arr':list(conv_arr),
                        'kb_arr':list(kb_arr),
                        'last_user_utter':list(last_user_utter),
                        'dialog_history':list(dialog_history),
                        'id':int(sample_counter),
                        'ID':int(cnt_lin)}
                    data.append(data_detail)
                    
                    gen_r = generate_memory(r, "$s", str(nid))
                    context_arr += gen_r
                    conv_arr += gen_r
                    dialog_history += gen_r

                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    sample_counter += 1
                else:
                    r = line
                    kb_info = generate_memory(r, "", str(nid))
                    context_arr = kb_info + context_arr
                    kb_arr += kb_info

                    tmp_kb_info = r.split(' ')
                    tmp_kb_arr.append(tmp_kb_info)

            else:
                cnt_lin += 1
                context_arr, conv_arr, kb_arr, tmp_kb_arr = [], [], [], []
                
                last_user_utter, dialog_history = [], []

                if(max_line and cnt_lin >= max_line):
                    break

    return data, max_resp_len


def generate_memory(sent, speaker, time, task_type=None):
    sent_new = []
    sent_token = sent.split(' ')
    if speaker=="$u" or speaker=="$s":
        for idx, word in enumerate(sent_token):
            temp = [word, speaker, 'turn'+str(time), 'word'+str(idx)] + ["PAD"]*(MEM_TOKEN_SIZE-4)
            sent_new.append(temp)
    else:
        if task_type:
            sent_token.append(task_type)
        sent_token = sent_token[::-1] + ["PAD"]*(MEM_TOKEN_SIZE-len(sent_token)) 
        sent_new.append(sent_token) 
    return sent_new


def generate_template(response, gold_ent, kb_arr, tmp_entity_dic):
    sketch_response = []
    if gold_ent == []:
        sketch_response = response.split(' ')
    else:
        for word in response.split(' '):
            if word not in gold_ent:
                sketch_response.append(word)
            else:
                ent_type = None
                for key in tmp_entity_dic.keys():
                    if word in tmp_entity_dic[key]:
                        ent_type = key
                        break
                assert ent_type != None
                sketch_response.append('@' + ent_type)
    sketch_response = " ".join(sketch_response)
    return sketch_response


def generate_tmp_entity_dic(kb_arr_split):
    entity_dic = {}

    for kb_item in kb_arr_split:
        if len(kb_item) == 10:
            if entity_dic.get(kb_item[-2], None) == None:
                entity_dic[kb_item[-2]] = []
            if kb_item[-1] not in entity_dic[kb_item[-2]]:
                entity_dic[kb_item[-2]].append(kb_item[-1])
        elif len(kb_item) == 3:
            if entity_dic.get(kb_item[1], None) == None:
                entity_dic[kb_item[1]] = []
            if kb_item[2] not in entity_dic[kb_item[1]]:
                entity_dic[kb_item[1]].append(kb_item[2])
        else:
            print("[ERROR] generate_tmp_entity_dic.py error")

    return entity_dic


def prepare_data_seq(task, batch_size=100):
    file_train = 'data/CamRest/{}train.txt'.format(task)
    file_dev = 'data/CamRest/{}dev.txt'.format(task)
    file_test = 'data/CamRest/{}test.txt'.format(task)

    pair_train, train_max_len = read_langs(file_train, max_line=None)
    pair_dev, dev_max_len = read_langs(file_dev, max_line=None)
    pair_test, test_max_len = read_langs(file_test, max_line=None)
    max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1
    
    lang = Lang()

    train = get_seq(pair_train, lang, batch_size, True)
    dev   = get_seq(pair_dev, lang, batch_size, False)
    test  = get_seq(pair_test, lang, batch_size, False)
    
    print("Read %s sentence pairs train" % len(pair_train))
    print("Read %s sentence pairs dev" % len(pair_dev))
    print("Read %s sentence pairs test" % len(pair_test))
    print("Vocab_size: %s " % lang.n_words)
    print("Max. length of system response: %s " % max_resp_len)
    print("USE_CUDA={}".format(USE_CUDA))
    
    return train, dev, test, [], lang, max_resp_len

def get_data_seq(file_name, lang, max_len, batch_size=1):
    pair, _ = read_langs(file_name, max_line=None)
    d = get_seq(pair, lang, batch_size, False)
    return d
