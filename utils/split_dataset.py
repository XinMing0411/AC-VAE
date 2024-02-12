import itertools
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
import os

bert_path = "data/clotho_v2/bert_sentembeddings"
caption_file =  "data/clotho_v2/dev_val/text.json"
caption_df = json.load(open(caption_file, "r"))["audios"]

def list_multiply(embeddings):
    total = 0
    for (a,b) in itertools.combinations(embeddings, 2):
        total += cosine_similarity(a.reshape(1,-1),b.reshape(1,-1))[0][0]
    return total

with tqdm(total=len(caption_df), ascii=True) as pbar:
    bert_embeddings = dict()
    bert_sim = dict()
    for row in caption_df:
        captions = row["captions"]
        key = row["audio_id"]
        bert_embeddings[key] = []
        emb_path = os.path.join(bert_path,key)

        for index in range(5):
            bert_embeddings[key].append(np.load(os.path.join(emb_path,str(int(index))+".npy")))

        bert_sim[key] = list_multiply(bert_embeddings[key])
        pbar.update()

bert_sim_sorted = sorted(bert_sim.items(), key=lambda x:x[1], reverse=False)
hard_dict = dict(bert_sim_sorted[:len(bert_sim_sorted)//2])
easy_dict = dict(bert_sim_sorted[len(bert_sim_sorted)//2:])
hard_data = []
easy_data = []

with tqdm(total=len(caption_df), ascii=True) as pbar:
    for row in caption_df:
        key = row["audio_id"]
        if key in hard_dict:
            hard_data.append(row)
        else:
            easy_data.append(row)
        pbar.update()
hard_data = { "audios": hard_data }
easy_data = { "audios": easy_data }
json.dump(hard_data, open("data/clotho_v2/dev_val/hard_text.json", "w"), indent=4)
json.dump(easy_data, open("data/clotho_v2/dev_val/easy_data.json", "w"), indent=4)