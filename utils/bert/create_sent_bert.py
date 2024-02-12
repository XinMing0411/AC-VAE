# -*- coding: utf-8 -*-

from ast import Index
import sys
import os
import fire
import pickle
import numpy as np
from tqdm import tqdm
import json
from bert_serving.client import BertClient

sys.path.append(os.getcwd())
from utils.build_vocab import Vocabulary

caption_file =  "data/clotho_v1/dev/text.json"
ip = "127.0.0.1"
embedding_path = "data/clotho_v1/bert_sentembeddings"
caption_df = json.load(open(caption_file, "r"))["audios"]
client = BertClient(ip)
embeddings = {}
with tqdm(total=len(caption_df), ascii=True) as pbar:
    for row in caption_df:
        captions = row["captions"]
        key = row["audio_id"]
        emb_path = os.path.join(embedding_path,key)
        os.makedirs(emb_path,exist_ok=True)
        for caption in captions:
            caption_embeds = []
            sent = caption["caption"]
            index = caption["cap_id"]
            sent_emb = np.array(client.encode([sent])).reshape(-1)
            np.save(os.path.join(emb_path,str(int(index)-1)+".npy"),sent_emb)
            caption_embeds.append(sent_emb)
        np.save(os.path.join(emb_path,"caption.npy"),np.average(np.array(caption_embeds),0))
        
        pbar.update()

