import torch
import torch.utils.data as data
from utils.config import *

def _cuda(x):
    if USE_CUDA:
        return x.cuda()
    else:
        return x

class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word)
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])
      
    def index_words(self, story, trg=False):
        if trg:
            for word in story.split(' '):
                self.index_word(word)
        else:
            for word_triple in story:
                for word in word_triple:
                    self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, src_word2id, trg_word2id):
        """Reads source and target sequences from txt files."""
        self.data_info = {}
        for k in data_info.keys():
            self.data_info[k] = data_info[k]

        self.num_total_seqs = len(data_info['context_arr'])
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
    
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        context_arr = self.data_info['context_arr'][index]
        context_arr = self.preprocess(context_arr, self.src_word2id, trg="3")
        
        response = self.data_info['response'][index]
        response = self.preprocess(response, self.trg_word2id, trg="1")
        
        ptr_index = torch.Tensor(self.data_info['ptr_index'][index])
        selector_index = torch.Tensor(self.data_info['selector_index'][index])
        
        conv_arr = self.data_info['conv_arr'][index]
        conv_arr = self.preprocess(conv_arr, self.src_word2id, trg="3")
        
        kb_arr = self.data_info['kb_arr'][index]
        kb_arr = self.preprocess(kb_arr, self.src_word2id, trg="3")
        
        sketch_response = self.data_info['sketch_response'][index]
        sketch_response = self.preprocess(sketch_response, self.trg_word2id, trg="1")

        last_user_utter = self.data_info['last_user_utter'][index]
        last_user_utter = self.preprocess(last_user_utter, self.src_word2id, trg="3")
        
        dialog_history = self.data_info['dialog_history'][index]
        dialog_history = self.preprocess(dialog_history, self.src_word2id, trg="3")
        
        data_info = {}
        for k in self.data_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = self.data_info[k][index]

        data_info['context_arr_plain'] = self.data_info['context_arr'][index]
        data_info['response_plain'] = self.data_info['response'][index]
        data_info['kb_arr_plain'] = self.data_info['kb_arr'][index]

        return data_info

    def __len__(self):
        return self.num_total_seqs
    
    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        if trg == "1":
            story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')] + [EOS_token]
        elif trg == "2":
            story = []
            for utterance in sequence:
                for word in utterance:
                    if word in word2id:
                        story.append(word2id[word])
                    else:
                        story.append(UNK_token)
        elif trg == "3": 
            story = []
            for i, word_triple in enumerate(sequence):
                story.append([])
                for ii, word in enumerate(word_triple):
                    temp = word2id[word] if word in word2id else UNK_token
                    story[i].append(temp)
        else:
            print("[ERROR] You need to provide the [trg]")
            exit(1)
        story = torch.Tensor(story)
        return story

    def collate_fn(self, data):
        def merge(sequences,story_dim):
            lengths = [len(seq) for seq in sequences]
            max_len = 1 if max(lengths)==0 else max(lengths)
            if story_dim:
                padded_seqs = torch.ones(len(sequences), max_len, MEM_TOKEN_SIZE).long()
                mask = torch.ones(len(sequences), max_len).byte()
                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    if len(seq) != 0:
                        padded_seqs[i,:end,:] = seq[:end]
                        mask[i,:end] = torch.zeros(end)
            else:
                padded_seqs = torch.ones(len(sequences), max_len).long()
                mask = torch.ones(len(sequences), max_len).byte()
                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    padded_seqs[i, :end] = seq[:end]
                    mask[i,:end] = torch.zeros(end)

            return padded_seqs, lengths, mask

        def merge_index(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).float()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths
        

        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]


        context_arr, context_arr_lengths, context_arr_mask = merge(item_info['context_arr'], True)
        response, response_lengths, response_mask = merge(item_info['response'], False)
        selector_index, _ = merge_index(item_info['selector_index'])
        ptr_index, _, _ = merge(item_info['ptr_index'], False)
        conv_arr, conv_arr_lengths, conv_arr_mask = merge(item_info['conv_arr'], True)
        sketch_response, _, _ = merge(item_info['sketch_response'], False)
        kb_arr, kb_arr_lengths, kb_arr_mask = merge(item_info['kb_arr'], True)

        last_user_utter, last_user_utter_lengths, last_user_utter_mask = merge(item_info['last_user_utter'], True)
        dialog_history, dialog_history_lengths, dialog_history_mask = merge(item_info['dialog_history'], True)
        
        context_arr = _cuda(context_arr.contiguous())
        response = _cuda(response.contiguous())
        selector_index = _cuda(selector_index.contiguous())
        ptr_index = _cuda(ptr_index.contiguous())
        conv_arr = _cuda(conv_arr.contiguous())
        sketch_response = _cuda(sketch_response.contiguous())
        if(len(list(kb_arr.size()))>1): kb_arr = _cuda(kb_arr.contiguous())

        last_user_utter = _cuda(last_user_utter.contiguous())
        dialog_history = _cuda(dialog_history.contiguous())

        context_arr_mask = _cuda(context_arr_mask.contiguous())
        dialog_history_mask = _cuda(dialog_history_mask.contiguous())
        last_user_utter_mask = _cuda(last_user_utter_mask.contiguous())
        kb_arr_mask = _cuda(kb_arr_mask.contiguous())
        
        data_info = {}
        for k in item_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = item_info[k]

        data_info['context_arr_lengths'] = context_arr_lengths
        data_info['response_lengths'] = response_lengths
        data_info['conv_arr_lengths'] = conv_arr_lengths
        data_info['kb_arr_lengths'] = kb_arr_lengths

        data_info['last_user_utter_lengths'] = last_user_utter_lengths
        data_info['dialog_history_lengths'] = dialog_history_lengths

        data_info['last_user_utter_mask'] = last_user_utter_mask
        data_info['kb_arr_mask'] = kb_arr_mask
        data_info['dialog_history_mask'] = dialog_history_mask

        return data_info


def get_seq(pairs, lang, batch_size, type):   
    data_info = {}
    for k in pairs[0].keys():
        data_info[k] = []
    
    for pair in pairs:
        for k in pair.keys():
            data_info[k].append(pair[k])
        if(type):
            lang.index_words(pair['context_arr'])
            lang.index_words(pair['response'], trg=True)
            lang.index_words(pair['sketch_response'], trg=True)
    
    dataset = Dataset(data_info, lang.word2index, lang.word2index)
    data_loader = torch.utils.data.DataLoader(dataset = dataset,
                                              batch_size = batch_size,
                                              shuffle = type,
                                              collate_fn = dataset.collate_fn)
    return data_loader
