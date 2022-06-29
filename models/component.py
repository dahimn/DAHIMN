import torch
import random
import math
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, dropout=0.0):
        super(SelfAttention, self).__init__()
        self.projector = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequence, sequence_mask=None):
        batch_size, seq_len, hidden_size = sequence.size()
        sequence = self.dropout(sequence)
        scores = self.projector(sequence.contiguous().view(-1, hidden_size)).view(batch_size, seq_len)
        if sequence_mask is not None:
            scores.data.masked_fill_(sequence_mask.data.bool(), -float('inf'))

        scores = self.softmax(scores)
        weighted_sequence = scores.unsqueeze(2).expand_as(sequence).mul(sequence)
        return weighted_sequence


class Attention(nn.Module):
    def __init__(self,
                 query_size,
                 key_size=None,
                 hidden_size=None,
                 mode="dot",
                 return_attn_only=False,
                 do_project=False):
        super(Attention, self).__init__()
        assert (mode in ["dot", "general", "additive", "scaled-dot"]), (
            "Unsupported attention mode: {mode}"
        )
        self.query_size = query_size
        self.key_size = key_size
        self.hidden_size = hidden_size
        self.mode = mode
        self.return_attn_only = return_attn_only
        self.do_project = do_project
        self.softmax = nn.Softmax(dim=-1)

        if mode == "general": 
            self.linear1 = nn.Linear(
                self.query_size, self.key_size, bias=False
            )
        elif mode == "additive": 
            self.linear1 = nn.Linear(
                self.query_size, self.hidden_size, bias=True
            )
            self.linear2 = nn.Linear(
                self.key_size, self.hidden_size, bias=True
            )
            self.tanh = nn.Tanh()
            self.v = nn.Linear(self.hidden_size, 1, bias=False)

        if self.do_project:
            self.linear_project = nn.Sequential(
                nn.Linear(in_features=self.hidden_size + self.key_size,
                          out_features=self.hidden_size),
                nn.Tanh()
            )

    def forward(self, query, key, value=None, mask=None):
        if self.mode == "dot":
            assert query.size(-1) == key.size(-1)
            attn_score = torch.bmm(query, key.transpose(1, 2))
        elif self.mode == "scaled-dot":
            assert query.size(-1) == key.size(-1)
            attn_score = torch.bmm(query, key.transpose(1, 2))
            attn_score = attn_score / math.sqrt(query.size(-1))
        elif self.mode == "general":
            assert self.query_size == query.size(-1)
            assert self.key_size == key.size(-1)
            query_vec = self.linear1(query)
            attn_score = torch.bmm(query_vec, key.transpose(1, 2))
        else:
            assert self.query_size == query.size(-1)
            assert self.key_size == key.size(-1)
            hidden_vec = self.linear1(query).unsqueeze(2) + self.linear2(key).unsqueeze(1)
            hidden_vec = self.tanh(hidden_vec)
            attn_score = self.v(hidden_vec).squeeze(-1)

        if value is None:
            value = key

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, query.size(1), 1)
            attn_score.masked_fill_(mask.bool(), -float("inf"))

        attn_weights = self.softmax(attn_score)
        if self.return_attn_only:
            return attn_weights

        weighted_sum = torch.bmm(attn_weights, value)

        return weighted_sum, attn_weights


class RNNEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 embedder=None,
                 num_layers=1,
                 bidirectional=True,
                 dropout=0.0):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        rnn_hidden_size = hidden_size // num_directions

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_hidden_size = rnn_hidden_size
        self.embedder = embedder
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.rnn_hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          bidirectional=self.bidirectional)

    def forward(self, inputs, hidden=None):

        if isinstance(inputs, tuple):
            inputs, lengths = inputs
        else:
            inputs, lengths = inputs, None

        if self.embedder is not None:
            rnn_inputs = self.embedder(inputs)
        else:
            rnn_inputs = inputs

        batch_size = rnn_inputs.size(0)

        if lengths is not None:
            num_valid = lengths.gt(0).int().sum().item()
            sorted_lengths, indices = lengths.sort(descending=True)
            rnn_inputs = rnn_inputs.index_select(0, indices)

            rnn_inputs = pack_padded_sequence(
                rnn_inputs[:num_valid],
                sorted_lengths[:num_valid].tolist(),
                batch_first=True)

            if hidden is not None:
                hidden = hidden.index_select(1, indices)[:, :num_valid]

        outputs, last_hidden = self.rnn(rnn_inputs, hidden)

        if self.bidirectional:
            last_hidden = self._bridge_bidirectional_hidden(last_hidden)

        if lengths is not None:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)

            if num_valid < batch_size:
                zeros = outputs.new_zeros(
                    batch_size - num_valid, outputs.size(1), self.hidden_size)
                outputs = torch.cat([outputs, zeros], dim=0)

                zeros = last_hidden.new_zeros(
                    self.num_layers, batch_size - num_valid, self.hidden_size)
                last_hidden = torch.cat([last_hidden, zeros], dim=1)

            _, inv_indices = indices.sort()
            outputs = outputs.index_select(0, inv_indices)
            last_hidden = last_hidden.index_select(1, inv_indices)

        return outputs, last_hidden

    def _bridge_bidirectional_hidden(self, hidden):

        num_layers = hidden.size(0) // 2
        _, batch_size, hidden_size = hidden.size()
        return hidden.view(num_layers, 2, batch_size, hidden_size)\
            .transpose(1, 2).contiguous().view(num_layers, batch_size, hidden_size * 2)