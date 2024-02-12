# -*- coding: utf-8 -*-

import copy
from turtle import forward
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from models.attn_model import Seq2SeqAttention
import torch.nn.functional as F
from utils.train_util import mean_with_lens, max_with_lens, generate_length_mask

import sys
import os

class PosteriorBaseEncoder(nn.Module):
    
    def __init__(self, word_dim, embed_size,  vocab_size):
        super(PosteriorBaseEncoder, self).__init__()
        self.word_dim = word_dim
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.word_embedding = nn.Embedding(vocab_size, word_dim)

    def init(self):
        for m in self.modules():
            m.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def load_word_embeddings(self, weight, freeze=True):
        assert weight.shape[0] == self.vocab_size, "vocabulary size mismatch"
        self.word_embedding = nn.Embedding.from_pretrained(torch.as_tensor(weight), freeze=freeze)
        if weight.shape[1] != 512:
            self.word_embedding = nn.Sequential(
                    self.word_embedding,
                    nn.Linear(weight.shape[1], 512)
                )

    def forward(self, x, lengths,enc_mem=None,audio_lens=None):
        raise NotImplementedError

class PriorBaseEncoder(nn.Module):
    def __init__(self, word_dim, audiofeats_size,embed_size,  vocab_size):
        super(PriorBaseEncoder, self).__init__()
        self.word_dim = word_dim
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.audiofeats_size = audiofeats_size
        self.word_embedding = nn.Embedding(vocab_size, word_dim)

    def init(self):
        for m in self.modules():
            m.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def load_word_embeddings(self, weight, freeze=True):
        assert weight.shape[0] == self.vocab_size, "vocabulary size mismatch"
        self.word_embedding = nn.Embedding.from_pretrained(torch.as_tensor(weight), freeze=freeze)

        if weight.shape[1] != 512:
            self.word_embedding = nn.Sequential(
                    self.word_embedding,
                    nn.Linear(weight.shape[1], 512)
                )
    def forward(self, word,enc_mem,hiddens_state,last_z,lens):
        raise NotImplementedError


class PosteriorRNN(PosteriorBaseEncoder):
    def __init__(self, word_dim, embed_size, vocab_size,**kwargs):
        super(PosteriorRNN,self).__init__(word_dim, embed_size, vocab_size)
        self.hidden_size = kwargs.get('hidden_size', 256)
        self.bidirectional = kwargs.get('bidirectional', True)
        self.num_layers = kwargs.get('num_layers', 1)
        self.dropout = kwargs.get('dropout', 0.3)
        self.rnn_type = kwargs.get('rnn_type', "GRU")

        #Initializing the posterior network
        self.network = getattr(nn, self.rnn_type)(
            word_dim,
            self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            batch_first=True)
        self.mean_log_out = nn.Linear(embed_size + 2*self.hidden_size, 2*embed_size)

        self.init()

    def init_hidden(self, bsz, hidsize, device):
        h = Variable(torch.zeros(self.num_layers * (2 if self.bidirectional else 1), bsz, hidsize), requires_grad=True).to(device)
        return h  

    def forward(self, x,lengths):
        device =  self.mean_log_out.weight.device
        x = self.word_embedding(x[:,:-1].long().to(device))

        lengths = lengths - 1
        hiddens = self.init_hidden(x.data.size(0),self.hidden_size,x.device)
        emb = pack_padded_sequence(x, lengths, batch_first=True)
        outputs, hidden_t = self.network(emb, hiddens)
        output_unpack = pad_packed_sequence(outputs, batch_first=True)
        hidden_o = output_unpack[0]#.sum(1)

        means = torch.zeros(x.size(0), x.size(1), self.embed_size).to(x.device)
        logs = torch.zeros(x.size(0), x.size(1), self.embed_size).to(x.device)
        z_t_1 = torch.zeros(x.size(0), self.embed_size).to(x.device)
        z = torch.zeros(x.size(0), x.size(1), self.embed_size).to(x.device)

        for t in range(max(lengths)):
            
            mean_log_out = self.mean_log_out(torch.cat([hidden_o[:,t,:],z_t_1],dim=1))
            mean =  mean_log_out[:,:self.embed_size]
            log =  mean_log_out[:,self.embed_size:]
            
            epsilon = torch.randn(mean.shape).to(mean.device)
            z_t = epsilon * torch.exp(.5 * log) + mean
            

            means[:,t,:] = mean
            logs[:,t,:] = log
            z[:,t,:] = z_t
            z_t_1 = z_t#.clone()
        
        return {"q_means":means,
        "q_logs":logs,
        'q_z':z}

class PosteriorRNN_hybrid(PosteriorBaseEncoder):
    def __init__(self, word_dim, embed_size, vocab_size,**kwargs):
        super(PosteriorRNN_hybrid,self).__init__(word_dim, embed_size, vocab_size)
        self.hidden_size = kwargs.get('hidden_size', 256)
        self.bidirectional = kwargs.get('bidirectional', True)
        self.num_layers = kwargs.get('num_layers', 1)
        self.dropout = kwargs.get('dropout', 0.3)
        self.rnn_type = kwargs.get('rnn_type', "GRU")

        #Initializing the posterior network
        self.network = getattr(nn, self.rnn_type)(
            word_dim,
            self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            batch_first=True)
        
        # self.utturance_mean_log = nn.Linear(2*self.hidden_size, embed_size)
        self.token_mean_log = nn.Linear(2*self.hidden_size, 2*embed_size)
        self.init()

    def init_hidden(self, bsz, hidsize, device):
        h = Variable(torch.zeros(self.num_layers * (2 if self.bidirectional else 1), bsz, hidsize), requires_grad=True).to(device)
        return h  

    def forward(self, x,lengths):
        device =  self.token_mean_log.weight.device
        x = self.word_embedding(x[:,:-1].long().to(device))

        lengths = lengths - 1
        hiddens = self.init_hidden(x.data.size(0),self.hidden_size,x.device)
        emb = pack_padded_sequence(x, lengths, batch_first=True)
        outputs, hidden_t = self.network(emb, hiddens)
        output_unpack = pad_packed_sequence(outputs, batch_first=True)
        hidden_o = output_unpack[0]#.sum(1)

        token_mean_log_out = self.token_mean_log(hidden_o)
        token_means =  token_mean_log_out[:,:,:self.embed_size]
        token_logs =  token_mean_log_out[:,:,self.embed_size:]
        epsilon = torch.randn(token_means.shape).to(token_logs.device)
        token_zs = epsilon * torch.exp(.5 * token_logs) + token_means
        
        hidden_mean = mean_with_lens(hidden_o, lengths)
        hidden_max = max_with_lens(hidden_o, lengths)
        hidden = hidden_mean + hidden_max

        # utt_mean_log_out = self.utturance_mean_log(hidden)

        # utt_mean =  utt_mean_log_out[:,:self.embed_size]
        # utt_log =  utt_mean_log_out[:,self.embed_size:]

        # epsilon = torch.randn(utt_mean.shape).to(utt_mean.device)
        # utt_z = epsilon * torch.exp(.5 * utt_log) + utt_mean

        return {"q_means":token_means,
        "q_logs":token_logs,
        'q_z':token_zs,
        "q_means_utt":hidden,
        "q_logs_utt":None,
        "q_z_utt":None}

class PriorRNN(PriorBaseEncoder):
    #step by step
    def __init__(self, word_dim, audiofeats_size, embed_size, vocab_size,**kwargs):
        super(PriorRNN,self).__init__(word_dim, audiofeats_size, embed_size, vocab_size)
        self.hidden_size = kwargs.get('hidden_size', 256)
        self.bidirectional = kwargs.get('bidirectional', False)
        self.num_layers = kwargs.get('num_layers', 1)
        self.dropout = kwargs.get('dropout', 0.3)
        self.rnn_type = kwargs.get('rnn_type', "LSTM")
        self.word_attn = Seq2SeqAttention(audiofeats_size,word_dim,audiofeats_size)
        #Initializing the prior network getattr(nn, self.rnn_type)
        self.network = getattr(nn, self.rnn_type)(
            word_dim+audiofeats_size+embed_size,
            self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            batch_first=True)
        self.mean_log_out = nn.Linear(self.hidden_size, 2*embed_size)

        self.init()

    def init_hidden(self, bs,device):
        if self.rnn_type == "LSTM":
            return (torch.zeros(self.num_layers * (2 if self.bidirectional else 1), bs, self.embed_size).to(device),
                    torch.zeros(self.num_layers * (2 if self.bidirectional else 1), bs, self.embed_size).to(device))
        else:
            return torch.zeros(self.num_layers * (2 if self.bidirectional else 1),bs, self.embed_size).to(device)  
    
    def forward(self,word,enc_mem,hiddens_state,last_z,lens):
        word = word.to(enc_mem.device)
        x = self.word_embedding(word).to(enc_mem.device)

        embs, attn_weight = self.word_attn(x.squeeze(1),enc_mem,lens)

        output,hiddens_state = self.network(torch.cat([x.squeeze(1),embs.squeeze(1),last_z], dim=-1).unsqueeze(1), hiddens_state) 
        # print(output.shape)
        mean_log_out = self.mean_log_out(output.squeeze(1))
        # print(mean_log_out.shape)
        mean =  mean_log_out[:,:mean_log_out.size(-1)//2]
        log =  mean_log_out[:,mean_log_out.size(-1)//2:]
        epsilon = torch.randn(mean.shape).to(mean.device)
        # epsilon = torch.rand(mean.shape).to(mean.device)

        z_t = epsilon * torch.exp(.5 * log) + mean

        return {"mean":mean,
                "log":log,
                'hiddens_state':hiddens_state,
                "z":z_t
                }
