import math
import gc
import random
import sys
from typing import List, Optional, Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import json
import torch
import h5py
import os


sys.path.append(str(Path.cwd()))
from utils.build_vocab import Vocabulary
from datasets import augment

class GlobalDataset(torch.utils.data.Dataset):
    def __init__(self,vocabulary,istrain=True):
        if istrain == True:
            self.caption_info = json.load(open("/home/zhangyiming/Data-processed/clotho/development/text.json", "r"))["audios"]
        else:
            self.caption_info = json.load(open("/home/zhangyiming/Data-processed/clotho/validation/text.json", "r"))["audios"]
        
        self.captions = [caption["tokens"] for info in self.caption_info for caption in info["captions"]]
        # self.val_captions = [caption["tokens"] for info in self.val_caption_info for caption in info["captions"]]
        # self.captions.extend(self.val_captions)

        self.vocabulary = vocabulary
        self.max_lengths = 22
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, index):
        caption = [self.vocabulary('<start>')] + \
            [self.vocabulary(token) for token in self.captions[index].split()] + \
            [self.vocabulary('<end>')]
        caption = torch.as_tensor(caption)

        # caption_len = len(caption)
        # padding = self.max_lengths - len(caption)
        # if padding > 0:
        #     caption = torch.cat((caption, torch.zeros(padding, dtype=torch.int64)))
        # elif padding < 0:
        #     caption = caption[:self.max_lengths]
        # caption_len = caption_len if padding > 0 else self.max_lengths
        return caption

def collate_fn():

    def collate_wrapper(data_batches):
        # x: [feature, caption]
        # data_batches: [[feat1, cap1], [feat2, cap2], ..., [feat_n, cap_n]]

        def merge_seq(dataseq, dim=0):
            lengths = [seq.shape for seq in dataseq]
            # Assuming duration is given in the first dimension of each sequence
            maxlengths = tuple(np.max(lengths, axis=dim))
            # print(maxlengths)
            # For the case that the lengths are 2 dimensional
            lengths = np.array(lengths)[:, dim]
            padded = torch.zeros((len(dataseq),) + maxlengths,dtype=int)
            for i, seq in enumerate(dataseq):
                end = lengths[i]
                padded[i, :end] = seq[:end]
            return padded, lengths
        
        data_out = []
        data_len = []
        data_seq, tmp_len = merge_seq(data_batches)
        data_len.append(tmp_len)
        data_out.append(data_seq)
        data_out.extend(data_len)
 
        return data_out

    return collate_wrapper