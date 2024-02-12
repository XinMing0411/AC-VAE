# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn

from utils.train_util import generate_length_mask


class BaseDecoder(nn.Module):
    """
    Take word/audio embeddings and output the next word probs
    Base decoder, cannot be called directly
    All decoders should inherit from this class
    """

    def __init__(self, embed_size, vocab_size, enc_mem_size):
        super(BaseDecoder, self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.enc_mem_size = enc_mem_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        raise NotImplementedError


class RNNDecoder(BaseDecoder):

    def __init__(self, vocab_size, enc_mem_size, **kwargs):
        embed_size = kwargs.get("embed_size", 256)
        super(RNNDecoder, self).__init__(embed_size, vocab_size, enc_mem_size)
        dropout_p = kwargs.get("dropout", 0.0)
        hidden_size = kwargs.get('hidden_size', 256)
        num_layers = kwargs.get('num_layers', 1)
        bidirectional = kwargs.get('bidirectional', False)
        rnn_type = kwargs.get('rnn_type', "GRU")
        self.dropoutlayer = nn.Dropout(dropout_p)
        self.model = getattr(nn, rnn_type)(
            input_size=embed_size + enc_mem_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional)
        self.classifier = nn.Linear(
            hidden_size * (bidirectional + 1), vocab_size)
        nn.init.kaiming_uniform_(self.word_embeddings.weight)
        nn.init.kaiming_uniform_(self.classifier.weight)

    def load_word_embeddings(self, embeddings, tune=True, **kwargs):
        assert embeddings.shape[0] == self.vocab_size, "vocabulary size mismatch!"
        
        embeddings = torch.as_tensor(embeddings).float()
        self.word_embeddings.weight = nn.Parameter(embeddings)
        for para in self.word_embeddings.parameters():
            para.requires_grad = tune

        if embeddings.shape[1] != self.embed_size:
            assert "projection" in kwargs, "embedding size mismatch!"
            if kwargs["projection"]:
                self.word_embeddings = nn.Sequential(
                    self.word_embeddings,
                    nn.Linear(embeddings.shape[1], self.embed_size)
                )
    
    def forward(self, **kwargs):
        """
        RNN-style decoder must implement `forward` like this:
            accept a word input and last time hidden state, return the word
            logits output and hidden state of this timestep
        the return dict must contain at least `logits` and `states`
        """
        w = kwargs["word"]
        states = kwargs["state"]
        enc_mem = kwargs["enc_mem"]

        w = w.to(enc_mem.device)
        embed = self.dropoutlayer(self.word_embeddings(w))
        
        # embed: [N, T, embed_size]
        embed = torch.cat((embed, enc_mem), dim=-1)

        out, states = self.model(embed, states)
        # out: [N, T, hs], states: [num_layers * num_dire, N, hs]

        output = {
            "states": states,
            "output": out,
            "logits": self.classifier(out)
        }

        return output

    def init_hidden(self, bs):
        bidirectional = self.model.bidirectional
        num_layers = self.model.num_layers
        hidden_size = self.model.hidden_size
        return torch.zeros((bidirectional + 1) * num_layers, bs, hidden_size)


class RNNLuongAttnDecoder(RNNDecoder):

    def __init__(self, 
                 input_size,
                 attn_hidden_size, 
                 vocab_size,
                 **kwargs):
        super(RNNLuongAttnDecoder, self).__init__(input_size, vocab_size, **kwargs)
        self.hc2attn_h = nn.Linear(input_size + self.model.hidden_size, attn_hidden_size)
        self.classifier = nn.Linear(attn_hidden_size, vocab_size)
        nn.init.kaiming_uniform_(self.hc2attn_h.weight)
        nn.init.kaiming_uniform_(self.classifier.weight)

    def forward(self, *rnn_input, **attn_args): 
        x, h = rnn_input
        attn = attn_args["attn"]
        h_enc = attn_args["h_enc"]
        src_lens = attn_args["src_lens"]

        x = x.unsqueeze(1)
        out, h = self.model(x, h)
        c, attn_weight = attn(h.squeeze(0), h_enc, src_lens)
        attn_h = torch.tanh(self.hc2attn_h(torch.cat((h.squeeze(0), c), dim=-1)))
        logits = self.classifier(attn_h)

        return {"states": h, "logits": logits, "weights": attn_weight}


class RNNBahdanauAttnDecoder(RNNDecoder):

    def __init__(self, 
                 vocab_size,
                 enc_mem_size,
                 **kwargs):
        from models.attn_model import Seq2SeqAttention
        super(RNNBahdanauAttnDecoder, self).__init__(vocab_size, enc_mem_size, **kwargs)
        attn_size = kwargs.get("attn_size", self.model.hidden_size)
        self.attn = Seq2SeqAttention(enc_mem_size, self.model.hidden_size, attn_size)

    def forward(self, **kwargs):
        w = kwargs["word"]
        states = kwargs["state"]
        enc_mem = kwargs["enc_mem"]
        enc_mem_lens = kwargs["enc_mem_lens"]

        w = w.long().to(enc_mem.device)
        embed = self.dropoutlayer(self.word_embeddings(w))

        # embed: [N, 1, embed_size]
        c, attn_weight = self.attn(states.squeeze(0), enc_mem, enc_mem_lens)

        rnn_input = torch.cat((embed, c.unsqueeze(1)), dim=-1)

        out, states = self.model(rnn_input, states)

        output = {
            "states": states,
            "output": out,
            "logits": self.classifier(out),
            "weights": attn_weight
        }
        return output

class VAERNNBahdanauAttnDecoder(RNNDecoder):

    def __init__(self, 
                 vocab_size,
                 enc_mem_size,
                 **kwargs):
        from models.attn_model import Seq2SeqAttention
        super(VAERNNBahdanauAttnDecoder, self).__init__(vocab_size, enc_mem_size*2, **kwargs)
        attn_size = kwargs.get("attn_size", self.model.hidden_size)
        self.attn = Seq2SeqAttention(enc_mem_size, self.model.hidden_size, attn_size)

    def forward(self, **kwargs):
        w = kwargs["word"]
        states = kwargs["state"]
        enc_mem = kwargs["enc_mem"]
        enc_mem_lens = kwargs["enc_mem_lens"]
        z = kwargs["z"].to(enc_mem.device)

        w = w.long().to(enc_mem.device)
        embed = self.dropoutlayer(self.word_embeddings(w))

        # embed: [N, 1, embed_size]
        c, attn_weight = self.attn(states.squeeze(0), enc_mem, enc_mem_lens)

        rnn_input = torch.cat((embed, c.unsqueeze(1),z.unsqueeze(1)), dim=-1)
        # print(z.shape)
        # if len(z.shape) ==3:
        #     rnn_input = torch.cat((embed, c.unsqueeze(1),z.unsqueeze(1)), dim=-1)
        # else:
        #     rnn_input = torch.cat((embed, c.unsqueeze(1),z), dim=-1)
        out, states = self.model(rnn_input, states)

        output = {
            "state": states,
            "output": out,
            "logits": self.classifier(out),
            "weights": attn_weight,
            "rnn_input":rnn_input
        }
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [T, N, E]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerDecoder(BaseDecoder):

    def __init__(self, vocab_size, enc_mem_size, **kwargs):
        embed_size = kwargs.get("embed_size", 256)
        super(TransformerDecoder, self).__init__(embed_size, vocab_size, enc_mem_size)
        self.nhead = kwargs.get("nhead", 4)
        self.dropout_p = kwargs.get("dropout", 0.5)
        self.nlayers = kwargs.get("nlayers", 2)
        self.hidden_size = kwargs.get("hidden_size", 2048)
        
        self.dropoutlayer = nn.Dropout(self.dropout_p)
        self.pos_encoder = PositionalEncoding(self.embed_size, self.dropout_p)
        layers = nn.TransformerDecoderLayer(d_model=self.embed_size, nhead=self.nhead, dim_feedforward=self.hidden_size,dropout=self.dropout_p)
        self.model = nn.TransformerDecoder(layers, self.nlayers)
        self.outputlayer = nn.Linear(self.embed_size, vocab_size)
        nn.init.kaiming_uniform_(self.word_embeddings.weight)
        nn.init.kaiming_uniform_(self.outputlayer.weight)

    def generate_square_subsequent_mask(self, max_length):
        mask = (torch.triu(torch.ones(max_length, max_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, **kwargs):
        words = kwargs["words"]
        enc_mem = kwargs["enc_mem"]
        enc_mem_lens = kwargs["enc_mem_lens"]
        tgt_key_padding_mask = kwargs["caps_padding_mask"]


        enc_mem = enc_mem.transpose(0, 1) # [T_src, N, emb_size]
        # words = torch.LongTensor(wordscpu.numpy()).to(enc_mem.device)
        words = words.long().to(enc_mem.device)
        # print(enc_mem.device)
        # print(self.word_embeddings(words).device)
        embed = self.dropoutlayer(self.word_embeddings(words)) * math.sqrt(int(self.embed_size)) # [N, T, emb_size]
        embed = embed.transpose(0, 1) # [T, N, emb_size]
        embed = self.pos_encoder(embed)

        tgt_mask = self.generate_square_subsequent_mask(embed.size(0)).to(enc_mem.device)
        memory_key_padding_mask = ~generate_length_mask(enc_mem_lens).to(enc_mem.device)
        # print(memory_key_padding_mask.device)
        output = self.model(embed, enc_mem, tgt_mask=tgt_mask, 
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
        output = output.transpose(0, 1)
        # print(output.shape)
        # print(output[:,0,:])
        output = {
            "outputs": output,
            "logits": self.outputlayer(output),
        }
        return output

class VAETransformerDecoder(BaseDecoder):

    def __init__(self, vocab_size, enc_mem_size, **kwargs):
        embed_size = kwargs.get("embed_size", 256)
        super(VAETransformerDecoder, self).__init__(embed_size, vocab_size, enc_mem_size)
        self.nhead = kwargs.get("nhead", 4)
        self.dropout_p = kwargs.get("dropout", 0.5)
        self.nlayers = kwargs.get("nlayers", 2)
        self.hidden_size = kwargs.get("hidden_size", 2048)
        self.activation = kwargs.get("activation", "gelu")

        self.dropoutlayer = nn.Dropout(self.dropout_p)
        self.pos_encoder = PositionalEncoding(self.embed_size, self.dropout_p)
        layers = nn.TransformerDecoderLayer(d_model=self.embed_size, nhead=self.nhead, dim_feedforward=self.hidden_size, dropout=self.dropout_p, activation=self.activation)
        self.model = nn.TransformerDecoder(layers, self.nlayers)
        self.outputlayer = nn.Linear(self.embed_size, vocab_size)
        nn.init.kaiming_uniform_(self.word_embeddings.weight) 
        nn.init.kaiming_uniform_(self.outputlayer.weight)

    def generate_square_subsequent_mask(self, max_length):
        mask = (torch.triu(torch.ones(max_length, max_length)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, **kwargs):
        words = kwargs["words"]
        enc_mem = kwargs["enc_mem"]
        enc_mem_lens = kwargs["enc_mem_lens"]
        tgt_key_padding_mask = kwargs["caps_padding_mask"]
        z = kwargs["z"].to(enc_mem.device)

        enc_mem = enc_mem.transpose(0, 1) # [T_src, N, emb_size]
        z = z.transpose(0, 1) # [T_src, N, emb_size]
        enc_mem_z = torch.cat([enc_mem,z],0)

        words = words.long().to(enc_mem.device)
        embed = self.dropoutlayer(self.word_embeddings(words)) * math.sqrt(int(self.embed_size)) # [N, T, emb_size]
        embed = embed.transpose(0, 1) # [T, N, emb_size]
        embed = self.pos_encoder(embed)

        tgt_mask = self.generate_square_subsequent_mask(embed.size(0)).to(enc_mem.device)
        memory_key_padding_mask = ~generate_length_mask(enc_mem_lens).to(enc_mem.device)
        enc_mem_z_padding_mask = torch.cat([memory_key_padding_mask,tgt_key_padding_mask],1)

        output = self.model(tgt=embed, memory= enc_mem_z, tgt_mask=tgt_mask, 
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=enc_mem_z_padding_mask)
        output = output.transpose(0, 1)

        output = {
            "output": output[:,-1,:].squeeze(1),
            "logits": self.outputlayer(output)[:,-1,:].squeeze(1),
        }
        return output
