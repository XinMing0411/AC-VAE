# coding=utf-8
#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch
import gensim
from gensim.models import Word2Vec
from tqdm import tqdm
import fire
import json

import sys
import os
sys.path.append(os.getcwd())
from utils.build_vocab import Vocabulary



def build_sentence_vector(sentence,size,w2v_model):
    sen_vec=np.zeros(size)
    count=0
    for word in sentence:
        try:
            sen_vec+=w2v_model[word]
            count+=1
        except KeyError:
            continue
    if count!=0:
        sen_vec/=count
    return sen_vec


def create_embedding(caption_file,pretrained_weights_path,outputpath):
    caption_df = json.load(open(caption_file))["audios"]
    model = gensim.models.KeyedVectors.load_word2vec_format(
                fname=pretrained_weights_path,
                binary=True,
            )
    with tqdm(total=len(caption_df), ascii=True) as pbar:
        for audio in caption_df:
            audio_id = audio["audio_id"]
            emb_path = os.path.join(outputpath,audio_id)
            os.makedirs(emb_path,exist_ok=True)
            for caption in audio["captions"]:
                caption_embeds = []
                cap_id = caption["cap_id"]
                sent = caption["tokens"]
                sen_vec = build_sentence_vector(sent,model.vector_size,model)
                # print(sen_vec.shape)
                np.save(os.path.join(emb_path,str(int(cap_id)-1)+".npy"),sen_vec)
                caption_embeds.append(sen_vec)
                # print(np.average(np.array(caption_embeds),0).shape)
            np.save(os.path.join(emb_path,"caption.npy"),np.average(np.array(caption_embeds),0))
            pbar.update()



# print(model.vector_size)
# caption_df = pd.read_json(caption_file)
# caption_df["tokens"] = caption_df["tokens"].apply(lambda x: ["<start>"] + [token for token in x] + ["<end>"])
pretrained_weights_path = "gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz"
caption_file = "data/clotho_v1/dev/text.json"
outputpath = "data/clotho_v1/word2vec_sentembeddings"
create_embedding(caption_file,pretrained_weights_path,outputpath)