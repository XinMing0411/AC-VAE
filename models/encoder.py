# -*- coding: utf-8 -*-

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# sys.path.append(os.getcwd())
from utils.train_util import mean_with_lens, max_with_lens, generate_length_mask

class E2EASREncoder(nn.Module):
    
    def __init__(self, model): 
        super(E2EASREncoder, self).__init__()
        self.model = model
        self.embed_size = 320

    def forward(self, *input):
        x, lens = input
        output, lens, _ = self.model(x, lens)
        # output: [N, T, E]
        N = x.size(0)
        idxs = torch.arange(output.size(1), device="cpu").repeat(N).view(N, output.size(1))
        mask = (idxs < lens.view(-1, 1)).to(output.device)
        # mask: [N, T]

        out_mean_time = output * mask.unsqueeze(-1)
        out_mean = out_mean_time.sum(1) / lens.unsqueeze(1).to(x.device)

        return {
            "audio_embeds": out_mean,
            "audio_embeds_time": out_mean_time,
            "state": None,
            "audio_embeds_lens": lens
        }


def load_espnet_encoder(model_path, pretrained=True):
    import json
    import argparse
    from pathlib import Path
    from espnet.nets.pytorch_backend.e2e_asr import E2E
    
    model_dir = (Path(model_path).parent)
    with open(str(Path(model_dir)/"model.json"), "r") as f:
        idim, odim, conf = json.load(f)
    model = E2E(idim, odim, argparse.Namespace(**conf))
    if pretrained:
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    encoder = E2EASREncoder(model.enc)
    return encoder

class BaseEncoder(nn.Module):
    
    """
    Encodes the given input into a fixed sized dimension
    Base encoder class, cannot be called directly
    All encoders should inherit from this class
    """

    def __init__(self, inputdim, embed_size):
        super(BaseEncoder, self).__init__()
        self.inputdim = inputdim
        self.embed_size = embed_size

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
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        raise NotImplementedError


class CNNEncoder(BaseEncoder):

    def __init__(self, inputdim, embed_size, **kwargs):
        """

        :inputdim: TODO
        :embed_size: TODO
        :**kwargs: TODO

        """
        super(CNNEncoder, self).__init__(inputdim, embed_size)
        self._filtersizes = kwargs.get('filtersizes', [5, 3, 3])
        self._filter = kwargs.get('filter', [32, 32, 32])
        self._filter = [1] + self._filter
        net = nn.ModuleList()
        for nl, (h0, h1, filtersize) in enumerate(
                zip(self._filter, self._filter[1:], self._filtersizes)):
            if nl > 0:
                # GLU Output halves
                h0 = h0//2
            net.append(
                nn.Conv2d(
                    h0,
                    h1,
                    filtersize,
                    padding=int(
                        filtersize /
                        2),
                    bias=False))
            net.append(nn.BatchNorm2d(h1))
            net.append(nn.GLU(dim=1))
            net.append(nn.MaxPool2d((1, 2)))
        self.network = nn.Sequential(*net)

        def calculate_size(input_size):
            x = torch.randn(input_size).unsqueeze(0)
            output = self.network(x)
            return output.size()[1:]
        outputdim = calculate_size((1, 500, inputdim))[-1]
        self.outputlayer = nn.Linear(
            self._filter[-1]//2 * outputdim, self._embed_size)
        self.init()

    def forward(self, x):
        # Add dimension for filters
        x = x.unsqueeze(1)
        x = self.network(x)
        # Pool the time dimension
        x = x.mean(2)
        x = x.view(x.shape[0], x.shape[1]*x.shape[2])
        return self.outputlayer(x), None


class PreTrainedCNN(BaseEncoder):

    """Model that does not update its layers expect last layer"""

    def __init__(self, inputdim, embed_size, pretrained_model, **kwargs):
        """TODO: to be defined1.

        :inputdim: Input feature dimension
        :embed_size: Output of this module
        :**kwargs: Extra arguments ( config file )

        """
        super(PreTrainedCNN, self).__init__(inputdim, embed_size)

        # Remove last output layer
        modules = list(pretrained_model.children())[:-1]
        self.network = nn.Sequential(*modules)

        def calculate_size(input_size):
            x = torch.randn(input_size).unsqueeze(0)
            output = pretrained_model.network(x)
            return output.size()[1:]
        outputdim = calculate_size((1, 500, inputdim))[-1]//2
        self.outputlayer = nn.Linear(
            outputdim * pretrained_model._filter[-1],
            embed_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        with torch.no_grad():
            x = self.network(x)
            x = x.mean(2)
            x = x.view(x.shape[0], x.shape[1]*x.shape[2])
        return self.outputlayer(x), None


class Block2D(nn.Module):

    def __init__(self, cin, cout, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(cin),
            nn.Conv2d(cin,
                      cout,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),
            nn.LeakyReLU(inplace=True, negative_slope=0.1))

    def forward(self, x):
        return self.block(x)


class LinearSoftPool(nn.Module):
    """LinearSoftPool
    Linear softmax, takes logits and returns a probability, near to the actual maximum value.
    Taken from the paper:
        A Comparison of Five Multiple Instance Learning Pooling Functions for Sound Event Detection with Weak Labeling
    https://arxiv.org/abs/1810.09050
    """
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, time_decision):
        return (time_decision**2).sum(self.pooldim) / time_decision.sum(
            self.pooldim)


class MeanPool(nn.Module):

    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, logits, decision):
        return torch.mean(decision, dim=self.pooldim)

class AttentionPool(nn.Module):  
    """docstring for AttentionPool"""  
    def __init__(self, inputdim, outputdim=10, pooldim=1, **kwargs):  
        super().__init__()  
        self.inputdim = inputdim  
        self.outputdim = outputdim  
        self.pooldim = pooldim  
        self.transform = nn.Linear(inputdim, outputdim)  
        self.activ = nn.Softmax(dim=self.pooldim)  
        self.eps = 1e-7  


    def forward(self, logits, decision):  
        # Input is (B, T, D)  
        # B, T , D  
        w = self.activ(torch.clamp(self.transform(logits), -15, 15))  
        detect = (decision * w).sum(  
            self.pooldim) / (w.sum(self.pooldim) + self.eps)  
        # B, T, D  
        return detect


class MMPool(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.avgpool = nn.AvgPool2d(dims)
        self.maxpool = nn.MaxPool2d(dims)

    def forward(self, x):
        return self.avgpool(x) + self.maxpool(x)

def parse_poolingfunction(poolingfunction_name='mean', **kwargs):
    """parse_poolingfunction
    A heler function to parse any temporal pooling
    Pooling is done on dimension 1
    :param poolingfunction_name:
    :param **kwargs:
    """
    poolingfunction_name = poolingfunction_name.lower()
    if poolingfunction_name == 'mean':
        return MeanPool(pooldim=1)
    elif poolingfunction_name == 'linear':
        return LinearSoftPool(pooldim=1)
    elif poolingfunction_name == 'attention':  
        return AttentionPool(inputdim=kwargs['inputdim'],  
                             outputdim=kwargs['outputdim'])


class CRNNEncoder(BaseEncoder):

    def __init__(self, inputdim, embed_size, **kwargs):
        super(CRNNEncoder, self).__init__(inputdim, embed_size)
        features = nn.ModuleList()
        self.use_hidden = False
        assert embed_size == 256, "CRNN10 only supports output feature dimension 512"
        self.features = nn.Sequential(
            Block2D(1, 32),
            nn.LPPool2d(4, (2, 4)),
            Block2D(32, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 4)),
            Block2D(128, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (1, 4)),
            nn.Dropout(0.3),
        )
        with torch.no_grad():
            rnn_input_dim = self.features(torch.randn(1, 1, 500,
                                                      inputdim)).shape
            rnn_input_dim = rnn_input_dim[1] * rnn_input_dim[-1]

        self.gru = nn.GRU(rnn_input_dim,
                          128,
                          bidirectional=True,
                          batch_first=True)
        self.features.apply(self.init_weights)

    def forward(self, *input):
        x, lens = input
        lens = copy.deepcopy(lens)
        lens = torch.as_tensor(lens)
        N, T, _ = x.shape
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x, _ = self.gru(x)
        # x = nn.functional.interpolate(
            # x.transpose(1, 2),
            # T,
            # mode='linear',
            # align_corners=False).transpose(1, 2)
        lens /= 4

        # idxs = torch.arange(x.size(1), device="cpu").repeat(N).view(N, x.size(1))
        # mask = (idxs < lens.view(-1, 1)).to(x.device)
        # # mask: [N, T]

        # x_mean_time = x * mask.unsqueeze(-1)
        # x_mean = x_mean_time.sum(1) / lens.unsqueeze(1).to(x.device)

        # # x_max = x
        # # x_max[~mask] = float("-inf")
        # # x_max, _ = x_max.max(1)
        # # out = x_mean + x_max

        # out = x_mean
        out = mean_with_lens(x, lens)

        # return {
            # "audio_embeds": out,
            # "audio_embeds_time": x_mean_time,
            # "state": None,
            # "audio_embeds_lens": lens
        # }
        return {
            "audio_embeds": x, # [N, T, E]
            "audio_embeds_pooled": out,
            "audio_embeds_lens": lens,
            "state": None
        }


class CRNN8_Sub4(BaseEncoder):
    
    def __init__(self, inputdim, embed_size, **kwargs):
        super(CRNN8_Sub4, self).__init__(inputdim, embed_size)

        def _block(in_channel, out_channel):
            return nn.Sequential(
                nn.Conv2d(in_channel,
                          out_channel,
                          kernel_size=3,
                          bias=False,
                          padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(out_channel,
                          out_channel,
                          kernel_size=3,
                          bias=False,
                          padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
            )

        self.features = nn.Sequential(
            _block(1, 64),
            MMPool((2, 2)),
            nn.Dropout(0.2, True),
            _block(64, 128),
            MMPool((2, 2)),
            nn.Dropout(0.2, True),
            _block(128, 256),
            MMPool((1, 2)),
            nn.Dropout(0.2, True),
            _block(256, 512),
            MMPool((1, 2)),
            nn.Dropout(0.2, True),
            nn.AdaptiveAvgPool2d((None, 1)),
        )
        self.init_bn = nn.BatchNorm2d(inputdim)
        self.embedding = nn.Linear(512, 512)
        self.features.apply(self.init_weights)
        self.gru = nn.GRU(512, 256, bidirectional=True, batch_first=True)

    def forward(self, *input):
        x, lens = input
        lens = copy.deepcopy(lens)
        lens = torch.as_tensor(lens)
        N, T, _ = x.shape
        x = x.unsqueeze(1)  # B x 1 x T x D
        x = x.transpose(1, 3)
        x = self.init_bn(x)
        x = x.transpose(1, 3)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.embedding(x))
        x, _ = self.gru(x)

        lens //= 4
        # x: [N, T, E]
        idxs = torch.arange(x.size(1), device="cpu").repeat(N).view(N, x.size(1))
        mask = (idxs < lens.view(-1, 1)).to(x.device)
        # mask: [N, T]

        x_mean_time = x * mask.unsqueeze(-1)
        x_mean = x_mean_time.sum(1) / lens.unsqueeze(1).to(x.device)

        # x_max = x
        # x_max[~mask] = float("-inf")
        # x_max, _ = x_max.max(1)
        # out = x_mean + x_max

        out = x_mean

        return {
            "audio_embeds": out,
            "audio_embeds_time": x_mean_time,
            "state": None,
            "audio_embeds_lens": lens
        }


class CNN10QEncoder(BaseEncoder):

    def __init__(self, inputdim, embed_size, **kwargs):
        super(CNN10QEncoder, self).__init__(inputdim, embed_size)
        self.use_hidden = False

        def _block(in_channel, out_channel):
            return nn.Sequential(
                nn.Conv2d(in_channel,
                          out_channel,
                          kernel_size=3,
                          bias=False,
                          padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(out_channel,
                          out_channel,
                          kernel_size=3,
                          bias=False,
                          padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
            )

        self.features = nn.Sequential(
            _block(1, 64),
            nn.AvgPool2d((2, 2)),
            nn.Dropout(0.2, True),
            _block(64, 128),
            nn.AvgPool2d((2, 2)),
            nn.Dropout(0.2, True),
            _block(128, 256),
            nn.AvgPool2d((2, 2)),
            nn.Dropout(0.2, True),
            _block(256, 512),
            nn.AvgPool2d((2, 2)),
            nn.Dropout(0.2, True),
            nn.AdaptiveAvgPool2d((None, 1)),
        )
        self.init_bn = nn.BatchNorm2d(inputdim)
        # self.outputlayer = nn.Linear(512, embed_size)
        self.embedding = nn.Linear(512, 512)

    def forward(self, *input):
        return self._forward(*input)

    def _forward(self, *input):
        x, lens = input
        lens = copy.deepcopy(lens)
        lens = torch.as_tensor(lens)
        N = x.size(0)
        x = x.unsqueeze(1)  # [N, 1, T, D]
        x = x.transpose(1, 3)
        x = self.init_bn(x)
        x = x.transpose(1, 3)
        x = self.features(x) # [N, 512, T/16, 1]
        x = x.transpose(1, 2).contiguous().flatten(-2) # [N, T/16, 512]

        lens = torch.div(lens, 16, rounding_mode='floor')

        x_mean = mean_with_lens(x, lens)
        x_max = max_with_lens(x, lens)
        out = x_mean + x_max
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.embedding(out)

        return {
            "audio_embeds": x,
            "audio_embeds_pooled": out,
            "state": None,
            "audio_embeds_lens": lens
        }

class CNN10DEncoder(CNN10QEncoder):

    def __init__(self, inputdim, embed_size, **kwargs):
        super(CNN10DEncoder, self).__init__(inputdim, embed_size)
        self.outputlayer = nn.Linear(512, embed_size)

    def forward(self, *input):
        output = super(CNN10DEncoder, self)._forward(*input)
        x = output["audio_embeds"]
        x = self.embedding(x) # [N, T/16, 512]
        x = F.relu_(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.outputlayer(x)
        output["audio_embeds"] = x
        return output

class CNN10Encoder(BaseEncoder):

    def __init__(self, inputdim, embed_size, **kwargs):
        super(CNN10Encoder, self).__init__(inputdim, embed_size)
        assert embed_size == 512, "pretrained CNN10 only supports output feature dimension 512"
        self.use_hidden = False
        self.features = nn.Sequential(
            Block2D(1, 64),
            Block2D(64, 64),
            nn.LPPool2d(4, (2, 4)),
            Block2D(64, 128),
            Block2D(128, 128),
            nn.LPPool2d(4, (2, 2)),
            Block2D(128, 256),
            Block2D(256, 256),
            nn.LPPool2d(4, (1, 2)),
            Block2D(256, 512),
            Block2D(512, 512),
            nn.LPPool2d(4, (1, 2)),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool2d((None, 1)),
        )

        # self.temp_pool = parse_poolingfunction(kwargs.get('temppool', 'attention'),
                                               # inputdim=512,
                                               # outputdim=embed_size)
        # self.outputlayer = nn.Linear(512, embed_size)
        self.features.apply(self.init_weights)
        # self.outputlayer.apply(self.init_weights)

    def forward(self, *input):
        x, lens = input
        lens = torch.as_tensor(lens)
        batch, time, dim = x.shape
        x = x.unsqueeze(1)
        x = self.features(x)
        x = x.transpose(1, 2).contiguous().flatten(-2)
        # decison_time = self.outputlayer(x)
        # decison_time = nn.functional.interpolate(
            # decison_time.transpose(1, 2),
            # time,
            # mode='linear',
            # align_corners=False).transpose(1, 2)
        # x = self.temp_pool(x, decison_time).squeeze(1)

        N = x.size(0)
        lens /= 4
        idxs = torch.arange(x.size(1), device="cpu").repeat(N).view(N, x.size(1))
        mask = (idxs < lens.view(-1, 1)).to(x.device)

        x_mean = x * mask.unsqueeze(-1)
        x_mean = x_mean.sum(1) / lens.unsqueeze(1).to(x.device)

        out = x_mean
        return {
            "audio_embeds": out,
            "audio_embeds_time": x,
            "state": None,
            "audio_embeds_lens": lens
        }


class CNN10CRNNEncoder(BaseEncoder):

    def __init__(self, inputdim, embed_size, crnn, cnn, **kwargs):
        super(CNN10CRNNEncoder, self).__init__(inputdim, embed_size)
        self.use_hidden = False
        self.crnn = crnn
        self.cnn = cnn

    def forward(self, *input):
        crnn_feat, _ = self.crnn(*input)
        cnn_feat, _ = self.cnn(*input)
        # out = (crnn_feat + cnn_feat) / 2
        out = torch.cat((crnn_feat, cnn_feat), dim=-1)
        return out, None

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x

class Cnn10(BaseEncoder):
    def __init__(self, inputdim, embed_size, **kwargs):
        super(Cnn10, self).__init__(inputdim, embed_size)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.embed_pooled = nn.Linear(512, 512, bias=True)
        # self.embed = nn.Linear(512, 256, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        # init_layer(self.embed)
        init_layer(self.embed_pooled)
 
    def forward(self, input, lens):
        """
        Input: (batch_size, data_length)"""
           
        x = input.unsqueeze(1) # (batch_size, 1, time_steps, dim)
        lens = torch.as_tensor(lens)
        lens //= 16
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        out = x1 + x2
        out = F.dropout(out, p=0.5, training=self.training)
        out = F.relu_(self.embed_pooled(out))
        embedding = F.dropout(out, p=0.5, training=self.training)
        
        x = x.transpose(1, 2).contiguous()
        
        output_dict = {'audio_embeds': x,
           'audio_embeds_pooled': embedding,
           'state': None,
           'audio_embeds_lens': lens}

        return output_dict

class RNNEncoder(BaseEncoder):

    def __init__(self, inputdim, embed_size, **kwargs):
        super(RNNEncoder, self).__init__(inputdim, embed_size)
        hidden_size = kwargs.get('hidden_size', 256)
        bidirectional = kwargs.get('bidirectional', False)
        num_layers = kwargs.get('num_layers', 1)
        dropout = kwargs.get('dropout', 0.3)
        rnn_type = kwargs.get('rnn_type', "GRU")
        self.representation = kwargs.get('representation', 'time')
        assert self.representation in ('time', 'mean')
        self.use_hidden = kwargs.get('use_hidden', False)
        self.network = getattr(nn, rnn_type)(
            inputdim,
            hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True)
        self.outputlayer = nn.Linear(
            hidden_size * (bidirectional + 1), embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init()

    def forward(self, *input):
        x, lens = input
        lens = torch.as_tensor(lens)
        # x: [N, T, D]
        packed = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        packed_out, hid = self.network(packed)
        # hid: [num_layers, N, hidden]
        out_time, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # out: [N, T, hidden]
        if not self.use_hidden:
            hid = None
        if self.representation == 'mean':
            N = x.size(0)
            idxs = torch.arange(x.size(1), device="cpu").repeat(N).view(N, x.size(1))
            mask = (idxs < lens.view(-1, 1)).to(x.device)
            # mask: [N, T]
            out = out_time * mask.unsqueeze(-1)
            out = out.sum(1) / lens.unsqueeze(1).to(x.device)
        elif self.representation == 'time':
            indices = (lens - 1).reshape(-1, 1, 1).expand(-1, 1, out_time.size(-1))
            # indices: [N, 1, hidden]
            out = torch.gather(out_time, 1, indices).squeeze(1)

        out = self.bn(self.outputlayer(out))
        return {
            "audio_embeds": out,
            "audio_embeds_time": out_time,
            "state": hid,
            "audio_embeds_lens": lens
        }

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes ,stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetEncoder(BaseEncoder):
    
    def __init__(self, inputdim, embed_size, **kwargs):
        super().__init__(inputdim, embed_size)
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False), # (0)
            nn.BatchNorm2d(64), # (1)
            nn.ReLU(inplace=True), # (2)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), # (3)
            nn.Sequential(
                BasicBlock(64, 64),
                BasicBlock(64, 64)
            ), # (4)
            nn.Sequential(
                BasicBlock(64, 128, stride=2, downsample=nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(128)
                )),
                BasicBlock(128, 128)
            ), # (5)
            nn.Sequential(
                BasicBlock(128, 256, stride=2, downsample=nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(256)
                )),
                BasicBlock(256, 256)
            ), # (6)
            nn.Sequential(
                BasicBlock(256, 512, stride=2, downsample=nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
                    nn.BatchNorm2d(512)
                )),
                BasicBlock(512, 512)
            ) # (7)
        )
        self.layer = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, feats, feat_lens):
        x = feats
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.network(x)
        x = self.layer(x)
        x = x.squeeze(2)
        x = x.transpose(-1, -2) # [N, T*, 512]
        embed_lens = copy.deepcopy(feat_lens)
        embed_lens = torch.as_tensor(embed_lens)
        for i in range(5):
            embed_lens = (embed_lens - 1) // 2 + 1
        output_dict = {
            "audio_embeds": x,
            "audio_embeds_pooled": x.mean(1),
            "state": None,
            "audio_embeds_lens": embed_lens
        }
        return output_dict

##################################
#
# PANN Cnn14_encoder_16k预训练模型
#
##################################

class Cnn14_16k(BaseEncoder):
    def __init__(self,inputdim, embed_size, **kwargs):
        
        super(Cnn14_16k, self).__init__(inputdim, embed_size)


        # # Spectrogram extractor
        # self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size, 
        #     win_length=window_size, window=window, center=center, pad_mode=pad_mode, 
        #     freeze_parameters=True)

        # # Logmel feature extractor
        # self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, 
        #     n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
        #     freeze_parameters=True)

        # # Spec augmenter
        # self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
        #     freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        # self.embeddingfc = nn.Linear(512, 512)
        # self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        # init_layer(self.embeddingfc)
        # init_layer(self.fc_audioset)
 
    def forward(self, input, lens):
        """
        Input: (batch_size, data_length)"""

        # x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        # x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = input.unsqueeze(1)
        lens = torch.as_tensor(lens)
        lens //= 32
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        
        # if self.training:
        #     x = self.spec_augmenter(x)

        # # Mixup on spectrogram
        # if self.training and mixup_lambda is not None:
        #     x = do_mixup(x, mixup_lambda)
        # print(x.shape)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        out = x1 + x2
        
        out = F.dropout(out, p=0.5, training=self.training)
        out = F.relu_(self.fc1(out))
        out = F.dropout(out, p=0.5, training=self.training)
        # embedding = F.dropout(out, p=0.5, training=self.training)
        
        x = x.transpose(1, 2).contiguous()
        
        output_dict = {'audio_embeds': x,
           'audio_embeds_pooled': out,
           'state': None,
           'audio_embeds_lens': lens}

        return output_dict

##################################
#
# PANN ResNet38_encoder_32k预训练模型
#
##################################

def _resnet_conv3x3(in_planes, out_planes):
    #3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)

def _resnet_conv1x1(in_planes, out_planes):
    #1x1 convolution
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)

class _ResnetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('_ResnetBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in _ResnetBasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.stride = stride

        self.conv1 = _resnet_conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _resnet_conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            out = F.avg_pool2d(x, kernel_size=(2, 2))
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class _ResnetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(_ResnetBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.stride = stride
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = _resnet_conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = _resnet_conv3x3(width, width)
        self.bn2 = norm_layer(width)
        self.conv3 = _resnet_conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        init_layer(self.conv3)
        init_bn(self.bn3)
        nn.init.constant_(self.bn3.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            x = F.avg_pool2d(x, kernel_size=(2, 2))

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out
    
class _ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(_ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[0])
                init_bn(downsample[1])
            elif stride == 2:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2), 
                    _resnet_conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
                init_layer(downsample[1])
                init_bn(downsample[2])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    
class ResNet38(BaseEncoder):
    def __init__(self,inputdim, embed_size, **kwargs):
        
        super(ResNet38, self).__init__(inputdim, embed_size)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        # self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)

        self.resnet = _ResNet(block=_ResnetBasicBlock, layers=[3, 4, 6, 3], zero_init_residual=True)

        self.conv_block_after1 = ConvBlock(in_channels=512, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048)
        # self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        # init_layer(self.fc_audioset)


    def forward(self, input,lens):
        """
        Input: (batch_size, data_length)"""

        # x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        # x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)
        
        x = input.unsqueeze(1)
        lens = torch.as_tensor(lens)
        lens //= 32
        x = input.unsqueeze(1)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = torch.mean(x, dim=3)

        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        out = x1 + x2
        out = F.dropout(out, p=0.5, training=self.training)
        out = F.relu_(self.fc1(out))
        out = F.dropout(out, p=0.5, training=self.training)
        
        x = x.transpose(1, 2).contiguous()
        output_dict = {'audio_embeds': x,
           'audio_embeds_pooled': out,
           'state': None,
           'audio_embeds_lens': lens}

        return output_dict



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


class TransformerEncoder(BaseEncoder):

    def __init__(self, raw_feat_dim, fc_feat_dim, attn_feat_dim, d_model, **kwargs):
        super().__init__(raw_feat_dim, fc_feat_dim, attn_feat_dim)
        self.d_model = d_model
        dropout = kwargs.get("dropout", 0.2)
        self.nhead = kwargs.get("nhead", self.d_model // 64)
        self.nlayers = kwargs.get("nlayers", 2)
        self.dim_feedforward = kwargs.get("dim_feedforward", self.d_model * 4)

        self.attn_proj = nn.Sequential(
            nn.Linear(attn_feat_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.d_model)
        )
        layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                           nhead=self.nhead,
                                           dim_feedforward=self.dim_feedforward,
                                           dropout=dropout)
        self.model = nn.TransformerEncoder(layer, self.nlayers)
        self.fc_out_transform = nn.Linear(self.d_model, self.d_model)
        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_dict):
        attn_feats = input_dict["attn_feats"]
        attn_feat_lens = input_dict["attn_feat_lens"]
        attn_feat_lens = torch.as_tensor(attn_feat_lens)

        attn_feats = self.attn_proj(attn_feats) # [bs, T, d_model]
        attn_feats = attn_feats.transpose(0, 1)

        src_key_padding_mask = ~generate_length_mask(attn_feat_lens, attn_feats.size(0)).to(attn_feats.device)
        output = self.model(attn_feats, src_key_padding_mask=src_key_padding_mask)

        attn_embs = output.transpose(0, 1)
        fc_embs = embedding_pooling(attn_embs, attn_feat_lens, "mean+max")
        fc_embs = F.dropout(fc_embs, p=0.5, training=self.training)
        fc_embs = self.fc_out_transform(fc_embs)
        fc_embs = F.relu_(fc_embs)
        fc_embs = F.dropout(fc_embs, p=0.5, training=self.training)
        return {
            "attn_embs": attn_embs,
            "fc_embs": fc_embs,
            "attn_emb_lens": attn_feat_lens
        }


class M2TransformerEncoder(BaseEncoder):

    def __init__(self, raw_feat_dim, fc_feat_dim, attn_feat_dim, d_model, **kwargs):
        try:
            from m2transformer.models.transformer import MemoryAugmentedEncoder, ScaledDotProductAttentionMemory
        except:
            raise ImportError("meshed-memory-transformer not installed; please run `pip install git+https://github.com/ruotianluo/meshed-memory-transformer.git`")
        super().__init__(raw_feat_dim, fc_feat_dim, attn_feat_dim)
        self.d_model = d_model
        dropout = kwargs.get("dropout", 0.1)
        self.nhead = kwargs.get("nhead", self.d_model // 64)
        self.nlayers = kwargs.get("nlayers", 2)
        self.dim_feedforward = kwargs.get("dim_feedforward", self.d_model * 4)

        self.attn_proj = nn.Linear(attn_feat_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, dropout, 200)
        self.model = MemoryAugmentedEncoder(self.nlayers, 0, self.attn_feat_dim,
                                            d_model=self.d_model,
                                            h=self.nhead,
                                            d_ff=self.dim_feedforward,
                                            dropout=dropout,
                                            attention_module=ScaledDotProductAttentionMemory,
                                            attention_module_kwargs={"m": 40})
        self.init_params()

    def init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, input_dict):
        attn_feats = input_dict["attn_feats"]
        attn_embs, attn_emb_mask = self.model(attn_feats)
        fc_embs = attn_embs.mean(-2)
        return {
            "fc_embs": fc_embs,
            "attn_embs": attn_embs,
            "attn_emb_mask": attn_emb_mask
        }


if __name__ == "__main__":
    import os

    state_dict = torch.load(os.path.join(os.getcwd(), "experiments/pretrained_encoder/ResNet38_mAP=0.434.pth"), map_location="cpu",)
    encoder = ResNet38(64, 527)

    encoder.load_state_dict(state_dict, strict=False)

    x = torch.randn(4,1571, 64)

    out = encoder(x, torch.tensor([1571, 1071, 985, 666]))
    print(out["audio_embeds"].shape)
