import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F
import random
import numpy as np
import os
import json

from utils.measures import wer, moses_multi_bleu
from utils.masked_cross_entropy import *
from utils.config import *
from models.modules import *

class DAHIMN(nn.Module):
    def __init__(self, hidden_size, lang, max_resp_len, path, task, dataset, lr, hop, n_layer, dropout):
        super(DAHIMN, self).__init__()
        self.name = "DAHIMN"
        self.task = task
        self.dataset = dataset
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size    
        self.lang = lang
        self.lr = lr
        self.hop = hop
        self.n_layer = n_layer
        self.dropout = dropout
        self.max_resp_len = max_resp_len
        self.decoder_hop = hop
        self.softmax = nn.Softmax(dim=0)

        if path:
            if USE_CUDA:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th')
                self.extKnow = torch.load(str(path)+'/enc_kb.th')
                self.kbAware = torch.load(str(path)+'/aware_kb.th')
                self.decoder = torch.load(str(path)+'/dec.th')
            else:
                print("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th',lambda storage, loc: storage)
                self.extKnow = torch.load(str(path)+'/enc_kb.th',lambda storage, loc: storage)
                self.kbAware = torch.load(str(path)+'/aware_kb.th',lambda storage, loc: storage)
                self.decoder = torch.load(str(path)+'/dec.th',lambda storage, loc: storage)
        else:
            self.encoder = HistoryEncoder(lang.n_words, hidden_size, dropout)
            self.extKnow = KnowledgeEncoder(lang.n_words, hidden_size, hop, n_layer=n_layer, dropout=dropout)
            self.kbAware = AwareLayer(hidden_size)
            self.decoder = LocalMemoryDecoder(self.encoder.embedding, lang, hidden_size, self.decoder_hop, dropout) 

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=lr)
        self.extKnow_optimizer = optim.Adam(self.extKnow.parameters(), lr=lr)
        self.kbAware_optimizer = optim.Adam(self.kbAware.parameters(), lr=lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.decoder_optimizer, mode='max', factor=0.5, patience=1.0, min_lr=0.0001, verbose=True)
        self.criterion_bce = nn.BCELoss()
        self.reset()

        if USE_CUDA:
            self.encoder.cuda()
            self.extKnow.cuda()
            self.kbAware.cuda()
            self.decoder.cuda()

    def print_loss(self):    
        print_loss_avg = self.loss / self.print_every
        print_loss_g = self.loss_g / self.print_every
        print_loss_v = self.loss_v / self.print_every
        print_loss_l = self.loss_l / self.print_every
        self.print_every += 1     
        return 'L:{:.2f},LE:{:.2f},LG:{:.2f},LP:{:.2f}'.format(print_loss_avg, print_loss_g, print_loss_v, print_loss_l)
    
    def save_model(self, dec_type):
        if self.dataset == 'kvr':
            name_data = "KVR/"
        elif self.dataset == 'cam':
            name_data = "CAM/"
        else:
            print("[ERROR] dataset name is error, can't save model!")
            exit()

        hop_info = str(self.hop)
        layer_info = str(self.n_layer)
        directory = 'save/DAHIMN-'+args["addName"]+name_data+str(self.task)+'HDD'+str(self.hidden_size)+'BSZ'+str(args['batch'])+'DR'+str(self.dropout)+'HOP'+hop_info+'L'+layer_info+'lr'+str(self.lr)+'tfr'+str(args['teacher_forcing_ratio'])+str(dec_type)                 
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.encoder, directory + '/enc.th')
        torch.save(self.extKnow, directory + '/enc_kb.th')
        torch.save(self.kbAware, directory + '/aware_kb.th')
        torch.save(self.decoder, directory + '/dec.th')

    def reset(self):
        self.loss, self.print_every, self.loss_g, self.loss_v, self.loss_l = 0, 1, 0, 0, 0
    
    def _cuda(self, x):
        if USE_CUDA:
            return torch.Tensor(x).cuda()
        else:
            return torch.Tensor(x)

    def train_batch(self, data, clip, reset=0):
        if reset: self.reset()

        self.encoder_optimizer.zero_grad()
        self.extKnow_optimizer.zero_grad()
        self.kbAware_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        

        use_teacher_forcing = random.random() < args['teacher_forcing_ratio'] 
        max_target_length = max(data['response_lengths'])
        all_decoder_outputs_vocab, all_decoder_outputs_ptr, _, _, global_pointer = self.encode_and_decode(data, max_target_length, use_teacher_forcing, False)
        
        loss_g = self.criterion_bce(global_pointer, data['selector_index'])
        loss_v = masked_cross_entropy(
            all_decoder_outputs_vocab.transpose(0, 1).contiguous(), 
            data['sketch_response'].contiguous(), 
            data['response_lengths'])
        loss_l = masked_cross_entropy(
            all_decoder_outputs_ptr.transpose(0, 1).contiguous(), 
            data['ptr_index'].contiguous(), 
            data['response_lengths'])
        loss = loss_g + loss_v + loss_l
        loss.backward()

        ec = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        ec = torch.nn.utils.clip_grad_norm_(self.extKnow.parameters(), clip)
        ec = torch.nn.utils.clip_grad_norm_(self.kbAware.parameters(), clip)
        dc = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)

        self.encoder_optimizer.step()
        self.extKnow_optimizer.step()
        self.kbAware_optimizer.step()
        self.decoder_optimizer.step()

        self.loss += loss.item()
        self.loss_g += loss_g.item()
        self.loss_v += loss_v.item()
        self.loss_l += loss_l.item()
    
    def encode_and_decode(self, data, max_target_length, use_teacher_forcing, get_decoded_words):

        if args['unk_mask'] and self.decoder.training:
            story_size = data['context_arr'].size()
            rand_mask = np.ones(story_size)
            bi_mask = np.random.binomial([np.ones((story_size[0],story_size[1]))], 1-self.dropout)[0]
            rand_mask[:,:,0] = rand_mask[:,:,0] * bi_mask
            rand_mask = self._cuda(rand_mask)
            conv_story = data['conv_arr']
            history = data['dialog_history']
            story = data['context_arr'] * rand_mask.long()
        else:
            story, conv_story, history = data['context_arr'], data['conv_arr'], data['dialog_history']
        
        dh_outputs, dh_hidden, local_query = self.encoder(history, 
                                                          data['last_user_utter'], 
                                                          data['dialog_history_lengths'], 
                                                          data['last_user_utter_lengths'], 
                                                          data['last_user_utter_mask'])
        global_pointer, kb_readout, memory_output, top_full_m, f_logit = self.extKnow.load_memory(story, 
                                                                                                  data['kb_arr_lengths'], 
                                                                                                  data['conv_arr_lengths'], 
                                                                                                  dh_hidden, 
                                                                                                  dh_outputs)
        
        aware_info, request_aware_info, last_step_kb_memory = self.kbAware(dh_outputs,
                                                                           dh_hidden, 
                                                                           kb_readout, 
                                                                           top_full_m,
                                                                           data['kb_arr_lengths'], 
                                                                           dh_hidden.squeeze(1), 
                                                                           local_query, 
                                                                           f_logit, 
                                                                           data['dialog_history_mask'])
        
        batch_size = len(data['context_arr_lengths'])
        self.copy_list = []
        for elm in data['context_arr_plain']:
            elm_temp = [ word_arr[0] for word_arr in elm ]
            self.copy_list.append(elm_temp) 
        
        outputs_vocab, outputs_ptr, decoded_fine, decoded_coarse = self.decoder.forward(
            self.extKnow,
            self.kbAware,
            story.size(), 
            data['context_arr_lengths'],
            data['kb_arr_lengths'],
            self.copy_list, 
            aware_info,
            request_aware_info,
            last_step_kb_memory,
            data['sketch_response'], 
            max_target_length, 
            batch_size, 
            use_teacher_forcing, 
            get_decoded_words, 
            global_pointer,
            dh_outputs,
            data['dialog_history_mask']) 

        return outputs_vocab, outputs_ptr, decoded_fine, decoded_coarse, global_pointer

    def evaluate(self, dev, matric_best, early_stop=None, epoch_num=None):
        print("STARTING EVALUATION")
        self.encoder.train(False)
        self.extKnow.train(False)
        self.kbAware.train(False)
        self.decoder.train(False)  
        
        ref, hyp = [], []
        acc, total = 0, 0
        dialog_acc_dict = {}

        if args['dataset'] == 'kvr':
            F1_pred, F1_cal_pred, F1_nav_pred, F1_wet_pred = 0, 0, 0, 0
            F1_count, F1_cal_count, F1_nav_count, F1_wet_count = 0, 0, 0, 0
            TP_all, FP_all, FN_all = 0, 0, 0

            TP_sche, FP_sche, FN_sche = 0, 0, 0
            TP_wea, FP_wea, FN_wea = 0, 0, 0
            TP_nav, FP_nav, FN_nav = 0, 0, 0

        elif args['dataset'] == 'cam':
            F1_pred, F1_count = 0, 0
            TP_all, FP_all, FN_all = 0, 0, 0

        pbar = tqdm(enumerate(dev),total=len(dev))
        new_precision, new_recall, new_f1_score = 0, 0, 0

        entity_path = ''
        if args['dataset'] == 'kvr':
            entity_path = 'data/KVR/kvret_entities.json'
        
        if entity_path:
            with open(entity_path) as f:
                global_entity = json.load(f)
                global_entity_list = []
                for key in global_entity.keys():
                    if key != 'poi':
                        global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
                    else:
                        for item in global_entity['poi']:
                            global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
                global_entity_list = list(set(global_entity_list))

        for j, data_dev in pbar: 
            _, _, decoded_fine, decoded_coarse, global_pointer = self.encode_and_decode(data_dev, self.max_resp_len, False, True)
            decoded_coarse = np.transpose(decoded_coarse)
            decoded_fine = np.transpose(decoded_fine)
            for bi, row in enumerate(decoded_fine):
                st = ''
                for e in row:
                    if e == 'EOS': break
                    else: st += e + ' '
                st_c = ''
                for e in decoded_coarse[bi]:
                    if e == 'EOS': break
                    else: st_c += e + ' '
                pred_sent = st.lstrip().rstrip()
                pred_sent_coarse = st_c.lstrip().rstrip()
                gold_sent = data_dev['response_plain'][bi].lstrip().rstrip()
                ref.append(gold_sent)
                hyp.append(pred_sent)
                
                if args['dataset'] == 'kvr': 
                    single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(data_dev['ent_index'][bi],
                                                                                         pred_sent.split(), 
                                                                                         global_entity_list, 
                                                                                         data_dev['kb_arr_plain'][bi])
                    F1_pred += single_f1
                    F1_count += count
                    TP_all += single_tp
                    FP_all += single_fp
                    FN_all += single_fn

                    single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(data_dev['ent_idx_cal'][bi], 
                                                                                         pred_sent.split(), 
                                                                                         global_entity_list, 
                                                                                         data_dev['kb_arr_plain'][bi])
                    F1_cal_pred += single_f1
                    F1_cal_count += count
                    F1_cal_count += count
                    TP_sche += single_tp
                    FP_sche += single_fp
                    FN_sche += single_fn

                    single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(data_dev['ent_idx_nav'][bi], 
                                                                                         pred_sent.split(), 
                                                                                         global_entity_list, 
                                                                                         data_dev['kb_arr_plain'][bi])
                    F1_nav_pred += single_f1
                    F1_nav_count += count
                    TP_nav += single_tp
                    FP_nav += single_fp
                    FN_nav += single_fn

                    single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(data_dev['ent_idx_wet'][bi],
                                                                                         pred_sent.split(), 
                                                                                         global_entity_list, 
                                                                                         data_dev['kb_arr_plain'][bi])
                    F1_wet_pred += single_f1
                    F1_wet_count += count
                    TP_wea += single_tp
                    FP_wea += single_fp
                    FN_wea += single_fn


                    # compute F1 SCORE
                    # single_f1, count = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(), global_entity_list, data_dev['kb_arr_plain'][bi])
                    # F1_pred += single_f1
                    # F1_count += count
                    # single_f1, count = self.compute_prf(data_dev['ent_idx_cal'][bi], pred_sent.split(), global_entity_list, data_dev['kb_arr_plain'][bi])
                    # F1_cal_pred += single_f1
                    # F1_cal_count += count
                    # single_f1, count = self.compute_prf(data_dev['ent_idx_nav'][bi], pred_sent.split(), global_entity_list, data_dev['kb_arr_plain'][bi])
                    # F1_nav_pred += single_f1
                    # F1_nav_count += count
                    # single_f1, count = self.compute_prf(data_dev['ent_idx_wet'][bi], pred_sent.split(), global_entity_list, data_dev['kb_arr_plain'][bi])
                    # F1_wet_pred += single_f1
                    # F1_wet_count += count

                elif args['dataset'] == 'cam':
                    single_tp, single_fp, single_fn, single_f1, count = self.compute_prf(data_dev['ent_index'][bi],
                                                                                         pred_sent.split(),
                                                                                         [],
                                                                                         data_dev['kb_arr_plain'][bi])
                    F1_pred += single_f1
                    F1_count += count
                    TP_all += single_tp
                    FP_all += single_fp
                    FN_all += single_fn

                    # single_f1, count = self.compute_prf(data_dev['ent_index'][bi], pred_sent.split(), [], data_dev['kb_arr_plain'][bi])
                    # F1_pred += single_f1
                    # F1_count += count

                total += 1
                if (gold_sent == pred_sent):
                    acc += 1

                if args['genSample']:
                    self.print_examples(bi, data_dev, pred_sent, pred_sent_coarse, gold_sent)

        self.encoder.train(True)
        self.extKnow.train(True)
        self.kbAware.train(True)
        self.decoder.train(True)

        bleu_score = moses_multi_bleu(np.array(hyp), np.array(ref), lowercase=True)
        acc_score = acc / float(total)
        print("ACC SCORE:\t"+str(acc_score))

        if args['dataset'] == 'kvr':

            print("BLEU SCORE:\t"+str(bleu_score))
            print("F1-macro SCORE:\t{}".format(F1_pred / float(F1_count)))
            print("F1-macro-sche SCORE:\t{}".format(F1_cal_pred / float(F1_cal_count)))
            print("F1-macro-wea SCORE:\t{}".format(F1_wet_pred / float(F1_wet_count)))
            print("F1-macro-nav SCORE:\t{}".format(F1_nav_pred / float(F1_nav_count)))

            P_score = TP_all / float(TP_all + FP_all) if (TP_all + FP_all) != 0 else 0
            R_score = TP_all / float(TP_all + FN_all) if (TP_all + FN_all) != 0 else 0
            P_nav_score = TP_nav / float(TP_nav + FP_nav) if (TP_nav + FP_nav) != 0 else 0
            P_sche_score = TP_sche / float(TP_sche + FP_sche) if (TP_sche + FP_sche) != 0 else 0
            P_wea_score = TP_wea / float(TP_wea + FP_wea) if (TP_wea + FP_wea) != 0 else 0
            R_nav_score = TP_nav / float(TP_nav + FN_nav) if (TP_nav + FN_nav) != 0 else 0
            R_sche_score = TP_sche / float(TP_sche + FN_sche) if (TP_sche + FN_sche) != 0 else 0
            R_wea_score = TP_wea / float(TP_wea + FN_wea) if (TP_wea + FN_wea) != 0 else 0

            F1_score = self.compute_F1(P_score, R_score)
            print("F1-micro SCORE:\t{}".format(F1_score))
            print("F1-micro-sche SCORE:\t{}".format(self.compute_F1(P_sche_score, R_sche_score)))
            print("F1-micro-wea SCORE:\t{}".format(self.compute_F1(P_wea_score, R_wea_score)))
            print("F1-micro-nav SCORE:\t{}".format(self.compute_F1(P_nav_score, R_nav_score)))

            # print("F1 SCORE:\t{}".format(F1_pred/float(F1_count)))
            # print("\tCAL F1:\t{}".format(F1_cal_pred/float(F1_cal_count))) 
            # print("\tWET F1:\t{}".format(F1_wet_pred/float(F1_wet_count))) 
            # print("\tNAV F1:\t{}".format(F1_nav_pred/float(F1_nav_count))) 
            # print("BLEU SCORE:\t"+str(bleu_score)) 
            
        elif args['dataset'] == 'cam':
            print("BLEU SCORE:\t"+str(bleu_score))
            print("F1-macro SCORE:\t{}".format(F1_pred / float(F1_count)))
            P_score = TP_all / float(TP_all + FP_all) if (TP_all + FP_all) != 0 else 0
            R_score = TP_all / float(TP_all + FN_all) if (TP_all + FN_all) != 0 else 0
            F1_score = self.compute_F1(P_score, R_score)
            print("F1-micro SCORE:\t{}".format(F1_score))

            # print("F1 SCORE:\t{}".format(F1_pred/float(F1_count)))
            # print("BLEU SCORE:\t"+str(bleu_score))
            
        
        if early_stop == 'BLEU':
            if (bleu_score >= matric_best):
                self.save_model('BLEU-'+str(bleu_score))
                print("BLEU MODEL SAVED")
            return bleu_score
        elif (early_stop == 'ENTF1'):
            if (F1_score >= matric_best):
                self.save_model('ENTF1-{:.4f}'.format(F1_score))
                print("ENTF1 MODEL SAVED")  
            return F1_score
        else:
            if (acc_score >= matric_best):
                self.save_model('ACC-{:.4f}'.format(acc_score))
                print("ACC MODEL SAVED")
            return acc_score

    def compute_prf(self, gold, pred, global_entity_list, kb_plain):
        local_kb_word = [k[0] for k in kb_plain]
        TP, FP, FN = 0, 0, 0
        if len(gold)!= 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in set(pred):
                if p in global_entity_list or p in local_kb_word:
                    if p not in gold:
                        FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) !=0 else 0
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return TP, FP, FN, F1, count

    # def compute_prf(self, gold, pred, global_entity_list, kb_plain):
    #     local_kb_word = kb_plain[1:-1]
    #     TP, FP, FN = 0, 0, 0
    #     if len(gold)!= 0:
    #         count = 1
    #         for g in gold:
    #             if g in pred:
    #                 TP += 1
    #             else:
    #                 FN += 1
    #         for p in set(pred):
    #             if p in global_entity_list or p in local_kb_word:
    #                 if p not in gold:
    #                     FP += 1
    #         precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
    #         recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
    #         F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
    #     else:
    #         precision, recall, F1, count = 0, 0, 0, 0
    #     return F1, count

    def compute_F1(self, precision, recall):
        F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        return F1


    def compute_prf_cam(self, gold, pred, global_entity_list, kb_plain):
        local_kb_word = [k[0] for k in kb_plain]
        TP, FP, FN = 0, 0, 0
        if len(gold)!= 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in set(pred):
                if p in global_entity_list or p in local_kb_word:
                    if p not in gold:
                        FP += 1
            precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
            recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return F1, count


    def print_examples(self, batch_idx, data, pred_sent, pred_sent_coarse, gold_sent):
        kb_len = len(data['context_arr_plain'][batch_idx])-data['conv_arr_lengths'][batch_idx]-1
        print("{}: ID{} id{} ".format(data['domain'][batch_idx], data['ID'][batch_idx], data['id'][batch_idx]))
        for i in range(kb_len): 
            kb_temp = [w for w in data['context_arr_plain'][batch_idx][i] if w!='PAD']
            kb_temp = kb_temp[::-1]
            if 'poi' not in kb_temp:
                print(kb_temp)
        flag_uttr, uttr = '$u', []
        for word_idx, word_arr in enumerate(data['context_arr_plain'][batch_idx][kb_len:]):
            if word_arr[1]==flag_uttr:
                uttr.append(word_arr[0])
            else:
                print(flag_uttr,': ', " ".join(uttr))
                flag_uttr = word_arr[1]
                uttr = [word_arr[0]]
        print('Sketch System Response : ', pred_sent_coarse)
        print('Final System Response : ', pred_sent)
        print('Gold System Response : ', gold_sent)
        print('\n')
