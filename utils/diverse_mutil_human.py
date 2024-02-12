import json
import random
random.seed(1)
import argparse
import numpy as np
from functools import partial
from multiprocessing import Pool
from div_utils import compute_div_n ,compute_global_div_n
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

def eval_div_stats(preds_n):
    capsById = preds_n
    tokenizer = PTBTokenizer()
    n_caps_perimg = len(capsById[list(capsById.keys())[0]])
    # print(capsById)
    _capsById = capsById # save the untokenized version
    capsById = tokenizer.tokenize(capsById)
    # print(capsById)
    div_1, adiv_1 = compute_div_n(capsById,1)
    div_2, adiv_2 = compute_div_n(capsById,2)


    globdiv_1, _= compute_global_div_n(capsById,1)

    # compute mbleu
    scorer = Bleu(4)
    all_scrs = []
    scrperimg = np.zeros((n_caps_perimg, len(capsById)))

    for i in range(n_caps_perimg):
        tempRefsById = {}
        candsById = {}
        for k in capsById:
            tempRefsById[k] = capsById[k][:i] + capsById[k][i+1:]
            candsById[k] = [capsById[k][i]]

        score, scores = scorer.compute_score(tempRefsById, candsById)
        all_scrs.append(score)
        scrperimg[i,:] = scores[1]

    all_scrs = np.array(all_scrs)
    
    out = {}
    out['overall'] = {'Div1': div_1, 'Div2': div_2, 'gDiv1': globdiv_1}
    for k, score in zip(range(4), all_scrs.mean(axis=0).tolist()):
        out['overall'].update({'mBLeu_%d'%(k+1): score})

    return globdiv_1,div_1,div_2,out['overall']['mBLeu_4']


def calc_ngram(words, n=2):
    return zip(*[words[i:] for i in range(n)])

def calc_self_bleu(sentences, num_workers):
    pool = Pool(num_workers)
    result = []
    for idx in range(len(sentences)):
        hypothesis = sentences[idx]
        references = [sentences[_] for _ in range(len(sentences)) if _ != idx]
        result.append(pool.apply_async(
            partial(sentence_bleu, smoothing_function=SmoothingFunction().method1),
            args=(references, hypothesis))
        )
    score = 0.0
    cnt = 0
    for i in result:
        score += i.get()
        cnt += 1
    pool.close()
    pool.join()
    return score / cnt

parser = argparse.ArgumentParser()
parser.add_argument("system_output", type=str)
parser.add_argument("train_corpus", type=str)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--system_output_index", type=int, default=None)
parser.add_argument("--diversity_output", type=str, default=None)


args = parser.parse_args()
output = json.load(open(args.system_output))
train_data = json.load(open(args.train_corpus))["audios"]
# train_captions = [cap_item["tokens"] for item in train_data
#                     for cap_item in item["captions"]]
# train_captions = set(train_captions)
system_output_index = args.system_output_index

vocabulary = set()
num_novel_captions = 0
ngrams = [set() for _ in range(2)]
num_total_words = 0
pred_captions = []
gt_n = dict()
for item in train_data:
    audio_id = item["audio_id"]
    gt_n[audio_id] = []
    for cap_item in item["captions"]:
        tokens = cap_item["tokens"]
        gt_n[audio_id] = gt_n[audio_id] +[{
                    "audio_id": audio_id,
                    "caption": tokens
                }]
Vocab,div_1,div_2,mBLeu_4 = eval_div_stats(gt_n)
if args.diversity_output:
    with open(args.diversity_output, "w") as writer:
        print(f"Vocabulary size: {Vocab}", file=writer)
        print(f"Distinct-1: {div_1:.2g}", file=writer)
        print(f"Distinct-2: {div_2:.2g}", file=writer)
        print(f"Self-BLEU: {mBLeu_4:.2g}", file=writer)
# if "predictions" in output:
#     preds_n = dict()
#     for cap_item in output["predictions"]:

#         captions = cap_item["captions"]
#         filename = cap_item["filename"]

#         preds_n[filename] = []
#         for caption in captions:
#             tokens = caption["tokens"]
#             # preds_n[filename].append({
#             #             "audio_id": filename,
#             #             "caption": tokens
#             #         })
#             preds_n[filename] = preds_n[filename] +[{
#                         "audio_id": filename,
#                         "caption": tokens
#                     }]
#     Vocab,div_1,div_2,mBLeu_4 = eval_div_stats(preds_n)

#     if args.diversity_output:
#         with open(args.diversity_output, "w") as writer:
#             print(f"Vocabulary size: {Vocab}", file=writer)
#             print(f"Distinct-1: {div_1:.2g}", file=writer)
#             print(f"Distinct-2: {div_2:.2g}", file=writer)
#             print(f"Self-BLEU: {mBLeu_4:.2g}", file=writer)