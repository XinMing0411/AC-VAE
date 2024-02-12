# coding='utf-8'
from time import time
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.manifold import TSNE

def get_data(jsonfile,bertpath,proxypath):
    caption_df = json.load(open(jsonfile, "r"))["audios"]
    index = np.random.randint(0,len(caption_df),20)
    bertembeddings = []
    proxyembeddings = []
    labels = []
    for i in index:

        audio_id = caption_df[i]["audio_id"]
        bert_audio = os.path.join(bertpath,audio_id)
        proxy_audio = os.path.join(proxypath,audio_id)
        bertembedding = []
        proxyembedding = []
        for filename in range(1,6):
            bert = np.load(bert_audio+"/"+str(filename)+".npy")
            proxy = np.load(proxy_audio+"/"+str(filename-1)+".npy")
            bertembedding.append(bert)
            proxyembedding.append(proxy)
        bertembeddings.append(bertembedding)
        proxyembeddings.append(proxyembedding)
        labels.append([i]*5)
    
    return np.array(labels),np.array(bertembeddings),np.array(proxyembeddings)

bert_tsne = TSNE(n_components=2, init='pca', random_state=0)
proxy_tsne = TSNE(n_components=2, init='pca', random_state=0)

labels,bertembeddings,proxyembeddings = get_data("data/clotho_v1/dev/text.json","data/bert_sent/clotho_v1","data/clotho_v1/embeddings/3")
labels = np.reshape(labels,(100,))
print(bertembeddings.shape)
print(proxyembeddings.shape)
bertembeddings = np.reshape(bertembeddings,(100,-1))
proxyembeddings =np.reshape(proxyembeddings,(100,-1))

bertresult = bert_tsne.fit_transform(bertembeddings)
proxyresult = proxy_tsne.fit_transform(proxyembeddings)

class_num = len(np.unique(labels))
df = pd.DataFrame()
df["y"] = labels
df["bertcomp-1"] = bertresult[:,0]
df["bertcomp-2"] = bertresult[:,1]


bertfig = sns.scatterplot(x="bertcomp-1", y="bertcomp-2", hue=df.y,
                palette=sns.color_palette("hls", class_num),
                data=df).set(title="Bearing data T-SNE projection unsupervised")[0]

bert_fig = bertfig.get_figure()
bert_fig.savefig("bert_tsne.png", dpi = 400)
plt.close(bert_fig)

dft = pd.DataFrame()
dft["y"] = labels
dft["proxycomp-1"] = proxyresult[:,0]
dft["proxycomp-2"] = proxyresult[:,1]

proxyfig = sns.scatterplot(x="proxycomp-1", y="proxycomp-2", hue=dft.y,
                palette=sns.color_palette("hls", class_num),
                data=dft).set(title="Bearing data T-SNE projection unsupervised")[-1]

proxy_fig = proxyfig.get_figure()
proxy_fig.savefig("proxy_tsne.png", dpi = 400)
