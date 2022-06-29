import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import *
from utils.utils_general import _cuda
from models.component import Attention, SelfAttention, RNNEncoder

class HistoryEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, n_layers=1):
        super(HistoryEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=PAD_token)
        self.embedding.weight.data.normal_(0, 0.1)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        self.self_attention = SelfAttention(hidden_size, dropout)

        self.history_W = nn.Linear(2 * hidden_size, hidden_size)
        self.query_W = nn.Linear(2 * hidden_size, hidden_size)

    def get_state(self, bsz):
        return _cuda(torch.zeros(2, bsz, self.hidden_size))

    def forward(self, history, user_query, history_lengths, user_query_lengths, user_query_mask=None):
        history = history.transpose(0, 1)  
        user_query = user_query.transpose(0, 1)  

        history_embed = self.embedding(history.contiguous().view(history.size(0), -1).long()) 
        history_embed = history_embed.view(history.size() + (history_embed.size(-1),))
        history_embed = torch.sum(history_embed, 2).squeeze(2)
        history_embed = self.dropout_layer(history_embed)
        history_hidden = self.get_state(history_embed.size(1))
        if history_lengths:
            history_embed = nn.utils.rnn.pack_padded_sequence(history_embed,
                                                                   history_lengths,
                                                                   batch_first=False,
                                                                   enforce_sorted=False)
        history_outputs, history_hidden = self.gru(history_embed, history_hidden)
        if history_lengths:
            history_outputs, _ = nn.utils.rnn.pad_packed_sequence(history_outputs, batch_first=False)
        history_hidden = self.history_W(torch.cat((history_hidden[0], history_hidden[1]), dim=1)).unsqueeze(1) 
        history_outputs = self.history_W(history_outputs).transpose(0, 1) 

        query_embed = self.embedding(user_query.contiguous().view(user_query.size(0), -1).long())
        query_embed = query_embed.view(user_query.size() + (query_embed.size(-1),))
        query_embed = torch.sum(query_embed, 2).squeeze(2)
        query_embed = self.dropout_layer(query_embed)
        query_hidden = self.get_state(query_embed.size(1))
        if user_query_lengths:
            query_embed = nn.utils.rnn.pack_padded_sequence(query_embed,
                                                            user_query_lengths,
                                                            batch_first=False,
                                                            enforce_sorted=False)
        query_outputs, query_hidden = self.gru(query_embed, query_hidden)
        if user_query_lengths:
            query_outputs, _ = nn.utils.rnn.pad_packed_sequence(query_outputs, batch_first=False)
        query_outputs = self.query_W(query_outputs).transpose(0, 1) 
        query_outputs = self.self_attention(query_outputs, user_query_mask)
        local_query = torch.sum(query_outputs, 1) 

        return history_outputs, history_hidden, local_query


class KnowledgeEncoder(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, n_layer, dropout):
        super(KnowledgeEncoder, self).__init__()
        self.max_hops = hop
        self.n_layer = n_layer
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.bidirectional = True
        self.dropout_layer = nn.Dropout(dropout)
        for hop in range(self.max_hops + 1):
            C = nn.Embedding(vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")

        self.T_gates = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(self.max_hops + 1)])

        self.FW = nn.Linear(embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def add_lm_embedding(self, full_memory, kb_len, conv_len, history_outputs):
        for bi in range(full_memory.size(0)):
            start, end = kb_len[bi], kb_len[bi] + conv_len[bi]
            full_memory[bi, start:end, :] = full_memory[bi, start:end, :] + history_outputs[bi, :conv_len[bi], :]
        return full_memory

    def load_single_layer_memory(self, story, kb_len, conv_len, init_query, history_outputs, last_layer_u, cur_layer):
        init_query = init_query.squeeze(1)
        if cur_layer > 0: 
            gate = self.sigmoid(self.T_gates[0](init_query))
            init_query = last_layer_u[0] * gate + init_query * (1 - gate)

        u_forward = [init_query]
        story_size = story.size()
        single_m_story = []

        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story_size[0], -1))  
            embed_A = embed_A.view(story_size + (embed_A.size(-1),))  
            embed_A = torch.sum(embed_A, 2).squeeze(2)  
            if not args["ablationH"]:
                embed_A = self.add_lm_embedding(embed_A, kb_len, conv_len, history_outputs)
            embed_A = self.dropout_layer(embed_A)

            if (len(list(u_forward[-1].size())) == 1):
                u_forward[-1] = u_forward[-1].unsqueeze(0) 
            u_temp = u_forward[-1].unsqueeze(1).expand_as(embed_A)
            prob_logit = torch.sum(embed_A * u_temp, 2)
            prob_ = self.softmax(prob_logit)

            embed_C = self.C[hop + 1](story.contiguous().view(story_size[0], -1).long())
            embed_C = embed_C.view(story_size + (embed_C.size(-1),))
            embed_C = torch.sum(embed_C, 2).squeeze(2)
            if not args["ablationH"]:
                embed_C = self.add_lm_embedding(embed_C, kb_len, conv_len, history_outputs)

            prob = prob_.unsqueeze(2).expand_as(embed_C)
            o_k = torch.sum(embed_C * prob, 1)
            
            u_k = u_forward[-1] + o_k
            if cur_layer > 0: 
                gate = self.sigmoid(self.T_gates[hop + 1](u_k))
                u_k = last_layer_u[hop + 1] * gate + u_k * (1 - gate)
            
            u_forward.append(u_k)
            single_m_story.append(embed_A)
        single_m_story.append(embed_C)

        return prob_logit, u_forward[-1], u_forward, o_k, single_m_story


    def load_memory(self, story, kb_len, conv_len, query, history_outputs):
        self.m_story = []
        last_layer_u = [0] * (self.max_hops + 1)
        for layer in range(self.n_layer):
            prob_logit, last_hop_query, last_layer_u, memory_output, single_m_story = self.load_single_layer_memory(story, 
                                                                                                                    kb_len, 
                                                                                                                    conv_len, 
                                                                                                                    query, 
                                                                                                                    history_outputs, 
                                                                                                                    last_layer_u,
                                                                                                                    layer)
            query = self.relu(self.FW(last_hop_query))
            self.m_story.append(single_m_story)

        return self.sigmoid(prob_logit), last_hop_query, memory_output, self.m_story[-1][-1], prob_logit

        
    def forward(self, query_vector, global_pointer):
        u_forward = [query_vector]
        for hop in range(self.max_hops):
            m_A = self.m_story[-1][hop]
            if not args["ablationG"]:  
                m_A = m_A * global_pointer.unsqueeze(2).expand_as(m_A)
            if (len(list(u_forward[-1].size())) == 1):
                u_forward[-1] = u_forward[-1].unsqueeze(0)  
            u_temp = u_forward[-1].unsqueeze(1).expand_as(m_A)
            prob_logits = torch.sum(m_A * u_temp, 2)
            prob_soft = self.softmax(prob_logits)
            m_C = self.m_story[-1][hop + 1]
            if not args["ablationG"]:
                m_C = m_C * global_pointer.unsqueeze(2).expand_as(m_C)
            prob = prob_soft.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C * prob, 1)
            u_k = u_forward[-1] + o_k
            u_forward.append(u_k)


        return prob_soft, prob_logits, u_forward[-1], self.m_story[-1][-1] 


class KBAwareLayer(nn.Module):
    def __init__(self, hidden_size):
        super(KBAwareLayer, self).__init__()
        self.hidden_size = hidden_size
        self.kb_W = nn.Linear(hidden_size, hidden_size, bias=True)
        self.g_W = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_W = nn.Linear(hidden_size, hidden_size, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def get_kb_memory_pointer(self,
                              top_full_memory,
                              kb_len,
                              forward_last_prob_logits):
        max_kb_len = max(kb_len)
        top_kb_memory = _cuda(torch.zeros(top_full_memory.size(0), max_kb_len, self.hidden_size))
        forward_kb_prob = _cuda(torch.zeros(forward_last_prob_logits.size(0), max_kb_len))
        for bi in range(top_full_memory.size(0)):
            end = kb_len[bi]
            top_kb_memory[bi, :end, :] = top_full_memory[bi, :end, :]
            forward_kb_prob[bi, :end] = forward_last_prob_logits[bi, :end]

        fusion_kb_memory = self.relu(self.kb_W(top_kb_memory))
        final_kb_prob = forward_kb_prob

        for bi, end in enumerate(kb_len):
            if end < max_kb_len:
                final_kb_prob[bi, end:] = -(1e+20)
        final_kb_prob = self.softmax(final_kb_prob)

        return fusion_kb_memory, final_kb_prob

    def kb_aware_encode(self,
                        top_full_memory,
                        kb_len,
                        query,
                        forward_last_prob_logits):
        fusion_kb_memory, kb_prob = self.get_kb_memory_pointer(top_full_memory,
                                                               kb_len,
                                                               forward_last_prob_logits)
        
        fusion_kb_memory = fusion_kb_memory * kb_prob.unsqueeze(2).expand_as(fusion_kb_memory)
        kb_aware_info = torch.sum(fusion_kb_memory, 1).unsqueeze(1)

        return kb_aware_info, fusion_kb_memory

    def forward(self,
                top_full_memory,
                kb_len,
                query,
                forward_last_prob_logits, 
                last_step_kb_memory):
        fusion_kb_memory, kb_prob = self.get_kb_memory_pointer(top_full_memory,
                                                               kb_len,
                                                               forward_last_prob_logits)
        
        kb_memory = fusion_kb_memory * kb_prob.unsqueeze(2).expand_as(fusion_kb_memory)
        highway_gate = self.sigmoid(self.g_W(kb_memory))
        last_step_kb_memory = self.tanh(self.v_W(last_step_kb_memory))

        fusion_kb_memory = (1 - highway_gate) * kb_memory + highway_gate * last_step_kb_memory
        kb_aware_info = torch.sum(fusion_kb_memory, 1).unsqueeze(1)

        return kb_aware_info, fusion_kb_memory


class RequestAwareLayer(nn.Module):
    def __init__(self, hidden_size, attn_mode="dot"):
        super(RequestAwareLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = Attention(query_size=hidden_size,
                                   key_size=hidden_size,
                                   mode=attn_mode)
        self.weaken_w = nn.Linear(hidden_size, hidden_size, bias=True)
        self.strengthen_w = nn.Linear(hidden_size, hidden_size, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, request_query, history_outputs, history_mask=None):

        weighted_history, _ = self.attention(query=request_query.unsqueeze(1),
                                             key=history_outputs,
                                             value=history_outputs,
                                             mask=history_mask)
        request_aware_info = torch.sum(weighted_history, 1).unsqueeze(1)
        return request_aware_info


class GateLayer(nn.Module):
    def __init__(self, hidden_size, req_attn_mode="dot"):
        super(GateLayer, self).__init__()
        self.hidden_size = hidden_size
        self.kb_aware_layer = KBAwareLayer(hidden_size)
        self.request_aware_layer = RequestAwareLayer(hidden_size, attn_mode=req_attn_mode)
        self.WaTen = nn.Parameter(torch.full([1],1.0))
        self.WbTen = nn.Parameter(torch.full([1],1.0))
        self.sigmoid = nn.Sigmoid()


    def gate_for_encode(self,
                        top_full_memory,
                        kb_len,
                        kb_query,
                        forward_last_prob_logits,
                        request_query=None,
                        history_outputs=None,
                        history_mask=None,
                        request_aware_info=None):

        kb_aware_info, last_step_kb_memory = self.kb_aware_layer.kb_aware_encode(top_full_memory,
                                                                                 kb_len,
                                                                                 kb_query,
                                                                                 forward_last_prob_logits)
        if request_aware_info == None:
            assert history_outputs != None and request_query != None
            request_aware_info = self.request_aware_layer(request_query, history_outputs, history_mask)
        else:
            request_aware_info = request_aware_info


        gated_aware_info = self.sigmoid(self.WaTen) * kb_aware_info + self.sigmoid(self.WbTen) * request_aware_info
        return gated_aware_info, request_aware_info, last_step_kb_memory

    def forward(self,
                top_full_memory,
                kb_len,
                kb_query,
                forward_last_prob_logits,
                last_step_kb_memory,
                request_query=None,
                history_outputs=None,
                history_mask=None,
                request_aware_info=None):

        kb_aware_info, last_step_kb_memory = self.kb_aware_layer(top_full_memory,
                                                                 kb_len,
                                                                 kb_query,
                                                                 forward_last_prob_logits,
                                                                 last_step_kb_memory)
        if request_aware_info == None:
            assert history_outputs != None and request_query != None
            request_aware_info = self.request_aware_layer(request_query, history_outputs, history_mask)
        else:
            request_aware_info = request_aware_info

        gated_aware_info = self.sigmoid(self.WaTen) * kb_aware_info + self.sigmoid(self.WbTen) * request_aware_info
        
        return gated_aware_info, request_aware_info, last_step_kb_memory


class AwareLayer(nn.Module):
    def __init__(self, hidden_size, req_attn_mode="dot"):
        super(AwareLayer, self).__init__()
        self.hidden_size = hidden_size
        self.gate_layer = GateLayer(hidden_size, req_attn_mode=req_attn_mode)
        self.project = nn.Linear(hidden_size, hidden_size, bias=True)
        self.relu = nn.ReLU()

    def forward(self,
                history_outputs,
                history_hidden,
                memory_output,
                top_full_memory,
                kb_len,
                kb_query,
                request_query,
                forward_last_prob_logits,
                history_mask=None):

        memory_output = memory_output.unsqueeze(1) 
        kb_aware_info, request_aware_info, last_step_kb_memory = self.gate_layer.gate_for_encode(top_full_memory,
                                                                                                 kb_len,
                                                                                                 kb_query,
                                                                                                 forward_last_prob_logits,
                                                                                                 request_query=request_query,
                                                                                                 history_outputs=history_outputs,
                                                                                                 history_mask=history_mask,
                                                                                                 request_aware_info=None)

        fusion_context_info = self.relu(self.project(kb_aware_info + memory_output + history_hidden)).squeeze(1) 
        
        return fusion_context_info, request_aware_info, last_step_kb_memory


class LocalMemoryDecoder(nn.Module):
    def __init__(self, shared_emb, lang, embedding_dim, hop, dropout, his_attn_mode="additive"):
        super(LocalMemoryDecoder, self).__init__()
        self.num_vocab = lang.n_words
        self.lang = lang
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.embedding = shared_emb
        self.softmax = nn.Softmax(dim=-1)
        self.sketch_rnn = nn.GRU(embedding_dim, embedding_dim, dropout=dropout)
        self.history_attention = Attention(query_size=embedding_dim,
                                           key_size=embedding_dim,
                                           hidden_size=embedding_dim,
                                           mode=his_attn_mode)
        self.relu = nn.ReLU()
        self.W2 = nn.Linear(5 * embedding_dim, embedding_dim, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, 
                extKnow,
                awareLayer,
                story_size,
                story_lengths,
                kb_lengths,
                copy_list,
                encode_hidden,
                request_aware_info,
                last_step_kb_memory,
                target_batches,
                max_target_length,
                batch_size,
                use_teacher_forcing,
                get_decoded_words,
                global_pointer,
                history_outputs,
                history_mask=None):
        all_decoder_outputs_vocab = _cuda(torch.zeros(max_target_length, batch_size, self.num_vocab))
        all_decoder_outputs_ptr = _cuda(torch.zeros(max_target_length, batch_size, story_size[1]))
        decoder_input = _cuda(torch.LongTensor([SOS_token] * batch_size))
        memory_mask_for_step = _cuda(torch.ones(story_size[0], story_size[1]))
        decoded_fine, decoded_coarse = [], []

        hidden = encode_hidden.unsqueeze(0)

        for t in range(max_target_length):
            concat_input_list = []

            embed_q = self.dropout_layer(self.embedding(decoder_input))  
            if len(embed_q.size()) == 1: embed_q = embed_q.unsqueeze(0)
            
            _, hidden = self.sketch_rnn(embed_q.unsqueeze(0), hidden)
            query_vector = hidden[0]
            concat_input_list.append(embed_q)
            concat_input_list.append(hidden.squeeze(0))

            prob_soft, prob_logits, memory_output, top_m = extKnow(query_vector, global_pointer)
            all_decoder_outputs_ptr[t] = prob_logits
            concat_input_list.append(memory_output) 

            history_attn_outputs, _ = self.history_attention(hidden.transpose(0, 1),
                                                             history_outputs,
                                                             history_outputs,
                                                             history_mask)

            history_attn_outputs = torch.sum(history_attn_outputs, 1) 
            concat_input_list.append(history_attn_outputs)
            kb_aware_info, _, last_step_kb_memory = awareLayer.gate_layer(top_m,
                                                                          kb_lengths,
                                                                          query_vector,
                                                                          prob_logits,
                                                                          last_step_kb_memory,
                                                                          request_query=None,
                                                                          history_outputs=None,
                                                                          history_mask=None,
                                                                          request_aware_info=request_aware_info)
            concat_input_list.append(kb_aware_info.squeeze(1))

            concat_input = torch.cat(concat_input_list, dim=1)
            output = self.relu(self.W2(concat_input))
            p_vocab = self.attend_vocab(self.embedding.weight, output)
            all_decoder_outputs_vocab[t] = p_vocab
            _, topvi = p_vocab.data.topk(1)

            if use_teacher_forcing:
                decoder_input = target_batches[:, t]
            else:
                decoder_input = topvi.squeeze()

            if get_decoded_words:

                search_len = min(5, min(story_lengths))
                prob_soft = prob_soft * memory_mask_for_step
                _, toppi = prob_soft.data.topk(search_len)
                temp_f, temp_c = [], []

                for bi in range(batch_size):
                    token = topvi[bi].item()  
                    temp_c.append(self.lang.index2word[token])

                    if '@' in self.lang.index2word[token]:
                        cw = 'UNK'
                        for i in range(search_len):
                            if toppi[:, i][bi] < story_lengths[bi] - 1:
                                cw = copy_list[bi][toppi[:, i][bi].item()]
                                break
                        temp_f.append(cw)

                        if args['record']:
                            memory_mask_for_step[bi, toppi[:, i][bi].item()] = 0
                    else:
                        temp_f.append(self.lang.index2word[token])

                decoded_fine.append(temp_f)
                decoded_coarse.append(temp_c)

        return all_decoder_outputs_vocab, all_decoder_outputs_ptr, decoded_fine, decoded_coarse

    def attend_vocab(self, seq, cond):
        scores_ = cond.matmul(seq.transpose(1,0))
        return scores_



class AttrProxy(object):

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))
