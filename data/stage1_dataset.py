# -*- coding: utf-8 -*-

import torch
import json
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
from utils.build_vocab import Vocabulary

class Stage1DataSet(Dataset):
    """Some Information about Stage1DataSet"""
    def __init__(self,vocabulary,jsonfile,Extembedding=False,**kwargs):
        super(Stage1DataSet, self).__init__()
        self.vocabulary = vocabulary
        self.Extembedding = Extembedding
        self.uttnumbers = kwargs.get('uttnumbers', 4)
        self.caption_df = json.load(open(jsonfile, "r"))["audios"]
        

    def __getitem__(self, audio_idx:int):
        captions = []
        for temp in self.caption_df[audio_idx]["captions"]:
            tokens = temp["tokens"].split()
            print(temp["cap_id"])
            caption = [self.vocabulary('<start>')] + \
                [self.vocabulary(token) for token in tokens] + \
                [self.vocabulary('<end>')]
            captions.append(caption)
            
        if self.Extembedding == True:
            return np.array(captions)[:self.uttnumbers],self.caption_df[audio_idx]["audio_id"]
        
        return np.array(captions)[np.random.permutation(5)[:self.uttnumbers]],self.caption_df[audio_idx]["audio_id"]
        
    def __len__(self):
        return len(self.caption_df)

def collate_fn(batch):
    #batch: [[captions_emb,id] for seq in batch]
    id = []
    captions = []
    for caption in batch:
        
        captions = captions + list(caption[0])
        id = id + [caption[1]]*len(list(caption[0]))
        
    lens = [len(caption) for caption in captions]
    seq_len = np.max(lens)
    padded = np.zeros((len(captions),seq_len),dtype=int)

    for i, seq in enumerate(captions):
            end = lens[i]
            padded[i, :end] = seq[:end]
    return torch.as_tensor(padded), torch.as_tensor(lens),id


if __name__ == "__main__":
    import argparse
    import pickle
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'vocabulary',
        default='data/clotho_v2/dev/vocab.pkl',
        type=str,
        nargs="?")
    parser.add_argument(
        'jsonfile',
        default='data/clotho_v2/dev_val/text.json',
        type=str,
        nargs="?")
    args = parser.parse_args()
    vocabulary = pickle.load(open(args.vocabulary, "rb"))
    dataset = Stage1DataSet(vocabulary,args.jsonfile)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True,collate_fn=collate_fn)
    agg = 0
    for captions, lens, id in train_loader:
        # agg += len(feat)
        print(id[0])
        print(captions.shape)
        print(lens.shape)
        break
    print(len(dataset))