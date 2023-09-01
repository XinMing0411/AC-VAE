import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
import logging
import sys
from pprint import pformat

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths.cpu(), batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, attn_feats, attn_feat_lens):
    packed, inv_ix = sort_pack_padded_sequence(attn_feats, attn_feat_lens)
    if isinstance(module, torch.nn.RNNBase):
        return pad_unsort_packed_sequence(module(packed)[0], inv_ix)
    else:
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)

def generate_length_mask(lens, max_length=None):
    lens = torch.as_tensor(lens)
    N = lens.size(0)
    if max_length is None:
        max_length = max(lens)
    idxs = torch.arange(max_length).repeat(N).view(N, max_length)
    mask = (idxs < lens.view(-1, 1))
    return mask

def mean_with_lens(features, lens):
    """
    features: [N, T, ...] (assume the second dimension represents length)
    lens: [N,]
    """
    lens = torch.as_tensor(lens)
    if max(lens) != features.size(1):
        max_length = features.size(1)
        mask = generate_length_mask(lens, max_length)
    else:
        mask = generate_length_mask(lens)
    mask = mask.to(features.device) # [N, T]

    while mask.ndim < features.ndim:
        mask = mask.unsqueeze(-1)
    feature_mean = features * mask
    feature_mean = feature_mean.sum(1)
    while lens.ndim < feature_mean.ndim:
        lens = lens.unsqueeze(1)
    feature_mean = feature_mean / lens.to(features.device)
    # feature_mean = features * mask.unsqueeze(-1)
    # feature_mean = feature_mean.sum(1) / lens.unsqueeze(1).to(features.device)
    return feature_mean

def max_with_lens(features, lens):
    """
    features: [N, T, ...] (assume the second dimension represents length)
    lens: [N,]
    """
    lens = torch.as_tensor(lens)
    mask = generate_length_mask(lens).to(features.device) # [N, T]

    feature_max = features.clone()
    feature_max[~mask] = float("-inf")
    feature_max, _ = feature_max.max(1)
    return feature_max

def repeat_tensor(x, n):
    return x.unsqueeze(0).repeat(n, *([1] * len(x.shape)))

def init(m, method="kaiming"):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        if method == "kaiming":
            nn.init.kaiming_uniform_(m.weight)
        elif method == "xavier":
            nn.init.xavier_uniform_(m.weight)
        else:
            raise Exception(f"initialization method {method} not supported")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        if method == "kaiming":
            nn.init.kaiming_uniform_(m.weight)
        elif method == "xavier":
            nn.init.xavier_uniform_(m.weight)
        else:
            raise Exception(f"initialization method {method} not supported")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        if method == "kaiming":
            nn.init.kaiming_uniform_(m.weight)
        elif method == "xavier":
            nn.init.xavier_uniform_(m.weight)
        else:
            raise Exception(f"initialization method {method} not supported")

def compute_batch_score(decode_res,
                        key2refs,
                        keys,
                        start_idx,
                        end_idx,
                        vocabulary,
                        scorer):
    """
    Args:
        decode_res: decoding results of model, [N, max_length]
        key2refs: references of all samples, dict(<key> -> [ref_1, ref_2, ..., ref_n]
        keys: keys of this batch, used to match decode results and refs
    Return:
        scores of this batch, [N,]
    """

    if scorer is None:
        from pycocoevalcap.cider.cider import Cider
        scorer = Cider()

    hypothesis = {}
    references = {}

    for i in range(len(keys)):

        if keys[i] in hypothesis.keys():
            continue

        # prepare candidate sentence
        candidate = []
        for w_t in decode_res[i]:
            if w_t == start_idx:
                continue
            elif w_t == end_idx:
                break
            candidate.append(vocabulary.idx2word[w_t])

        hypothesis[keys[i]] = [" ".join(candidate), ]

        # prepare reference sentences
        references[keys[i]] = key2refs[keys[i]]

    score, scores = scorer.compute_score(references, hypothesis)
    key2score = {key: scores[i] for i, key in enumerate(references.keys())}
    results = np.zeros(decode_res.shape[0])
    for i in range(decode_res.shape[0]):
        results[i] = key2score[keys[i]]
    return results 

def get_centroids_prior(embeddings):
    centroids = []
    for speaker in embeddings:
        centroid = 0
        for utterance in speaker:
            centroid = centroid + utterance
        centroid = centroid/len(speaker)
        centroids.append(centroid)
    centroids = torch.stack(centroids)
    return centroids

def get_centroids(embeddings):
    centroids = embeddings.mean(dim=1)
    return centroids

def get_centroid(embeddings, speaker_num, utterance_num):
    centroid = 0
    for utterance_id, utterance in enumerate(embeddings[speaker_num]):
        if utterance_id == utterance_num:
            continue
        centroid = centroid + utterance
    centroid = centroid/(len(embeddings[speaker_num])-1)
    return centroid

def get_utterance_centroids(embeddings):
    """
    Returns the centroids for each utterance of a speaker, where
    the utterance centroid is the speaker centroid without considering
    this utterance
    Shape of embeddings should be:
        (speaker_ct, utterance_per_speaker_ct, embedding_size)
    """
    sum_centroids = embeddings.sum(dim=1)
    # we want to subtract out each utterance, prior to calculating the
    # the utterance centroid
    sum_centroids = sum_centroids.reshape(
        sum_centroids.shape[0], 1, sum_centroids.shape[-1]
    )
    # we want the mean but not including the utterance itself, so -1
    num_utterances = embeddings.shape[1] - 1
    centroids = (sum_centroids - embeddings) / num_utterances
    return centroids

def get_cossim_prior(embeddings, centroids):
    # Calculates cosine similarity matrix. Requires (N, M, feature) input
    cossim = torch.zeros(embeddings.size(0),embeddings.size(1),centroids.size(0))
    for speaker_num, speaker in enumerate(embeddings):
        for utterance_num, utterance in enumerate(speaker):
            for centroid_num, centroid in enumerate(centroids):
                if speaker_num == centroid_num:
                    centroid = get_centroid(embeddings, speaker_num, utterance_num)
                output = F.cosine_similarity(utterance,centroid,dim=0)+1e-6
                cossim[speaker_num][utterance_num][centroid_num] = output
    return cossim

def get_cossim(embeddings, centroids):
    # number of utterances per speaker
    num_utterances = embeddings.shape[1]
    utterance_centroids = get_utterance_centroids(embeddings)

    # flatten the embeddings and utterance centroids to just utterance,
    # so we can do cosine similarity
    utterance_centroids_flat = utterance_centroids.view(
        utterance_centroids.shape[0] * utterance_centroids.shape[1],
        -1
    )
    embeddings_flat = embeddings.view(
        embeddings.shape[0] * num_utterances,
        -1
    )
    # the cosine distance between utterance and the associated centroids
    # for that utterance
    # this is each speaker's utterances against his own centroid, but each
    # comparison centroid has the current utterance removed
    cos_same = F.cosine_similarity(embeddings_flat, utterance_centroids_flat)

    # now we get the cosine distance between each utterance and the other speakers'
    # centroids
    # to do so requires comparing each utterance to each centroid. To keep the
    # operation fast, we vectorize by using matrices L (embeddings) and
    # R (centroids) where L has each utterance repeated sequentially for all
    # comparisons and R has the entire centroids frame repeated for each utterance
    centroids_expand = centroids.repeat((num_utterances * embeddings.shape[0], 1))
    embeddings_expand = embeddings_flat.unsqueeze(1).repeat(1, embeddings.shape[0], 1)
    embeddings_expand = embeddings_expand.view(
        embeddings_expand.shape[0] * embeddings_expand.shape[1],
        embeddings_expand.shape[-1]
    )
    cos_diff = F.cosine_similarity(embeddings_expand, centroids_expand)
    cos_diff = cos_diff.view(
        embeddings.size(0),
        num_utterances,
        centroids.size(0)
    )
    # assign the cosine distance for same speakers to the proper idx
    same_idx = list(range(embeddings.size(0)))
    cos_diff[same_idx, :, same_idx] = cos_same.view(embeddings.shape[0], num_utterances)
    cos_diff = cos_diff + 1e-6
    return cos_diff

def calc_loss_prior(sim_matrix):
    # Calculates loss from (N, M, K) similarity matrix
    per_embedding_loss = torch.zeros(sim_matrix.size(0), sim_matrix.size(1))
    for j in range(len(sim_matrix)):
        for i in range(sim_matrix.size(1)):
            per_embedding_loss[j][i] = -(sim_matrix[j][i][j] - ((torch.exp(sim_matrix[j][i]).sum()+1e-6).log_()))
    loss = per_embedding_loss.sum()    
    return loss, per_embedding_loss

def calc_loss(sim_matrix):
    same_idx = list(range(sim_matrix.size(0)))
    pos = sim_matrix[same_idx, :, same_idx]
    neg = (torch.exp(sim_matrix).sum(dim=2) + 1e-6).log_()
    per_embedding_loss = -1 * (pos - neg)
    loss = per_embedding_loss.sum()
    return loss, per_embedding_loss

def parse_config_or_kwargs(config_file, **kwargs):
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    # passed kwargs will override yaml config
    return dict(yaml_config, **kwargs)

def genlogger(outputfile, level="INFO"):
    formatter = logging.Formatter(
        "[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger(__name__ + "." + outputfile)
    logger.setLevel(getattr(logging, level))
    # Log results to std
    # stdhandler = logging.StreamHandler(sys.stdout)
    # stdhandler.setFormatter(formatter)
    # Dump log to file
    filehandler = logging.FileHandler(outputfile)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    # logger.addHandler(stdhandler)
    return logger

def pprint_dict(in_dict, outputfun=sys.stdout.write, formatter='yaml'):
    """pprint_dict
    :param outputfun: function to use, defaults to sys.stdout
    :param in_dict: dict to print
    """
    if formatter == 'yaml':
        format_fun = yaml.dump
    elif formatter == 'pretty':
        format_fun = pformat
    for line in format_fun(in_dict).split('\n'):
        outputfun(line)
    
def convert_idx2sentence(word_ids, vocabulary, zh=False):
    candidate = []
    for word_id in word_ids:
        word = vocabulary.idx2word[word_id]
        if word == "<end>":
            break
        elif word == "<start>":
            continue
        candidate.append(word)
    if not zh:
        candidate = " ".join(candidate)
    return candidate

class ExponentialDecayScheduler(torch.optim.lr_scheduler._LRScheduler):
    
    def __init__(self, optimizer, total_iters,final_lrs, linear_warmup=False, warmup_iters=3000, last_epoch=-1, verbose=False):
        self.total_iters = total_iters
        self.final_lrs = final_lrs
        self.linear_warmup = linear_warmup
        if not isinstance(self.final_lrs, list) and not isinstance(self.final_lrs, tuple):
            self.final_lrs = [self.final_lrs] * len(optimizer.param_groups)
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch, verbose)

    def _get_closed_form_lr(self):
        warmup_coeff = 1.0
        current_iter = self._step_count
        if current_iter < self.warmup_iters:
            warmup_coeff = current_iter / self.warmup_iters
        current_lrs = []
        if not self.linear_warmup:
            for base_lr, final_lr in zip(self.base_lrs, self.final_lrs):
                current_lr = warmup_coeff * base_lr * math.exp(((current_iter - self.warmup_iters) / self.total_iters) * math.log(final_lr / base_lr))
                current_lrs.append(current_lr)
        else:
            for base_lr, final_lr in zip(self.base_lrs, self.final_lrs):
                if current_iter <= self.warmup_iters:
                    current_lr = warmup_coeff * base_lr
                else:
                    current_lr = warmup_coeff * base_lr * math.exp(((current_iter - self.warmup_iters) / self.total_iters) * math.log(final_lr / base_lr))
                current_lrs.append(current_lr)
        return current_lrs

    def get_lr(self):
        return self._get_closed_form_lr()


class NoamScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, model_size=512, warmup_iters=3000, last_epoch=-1, verbose=False):
        self.model_size = model_size
        self.warmup_iters = warmup_iters
        self.factors = [group["lr"] / (self.model_size ** (-0.5) * self.warmup_iters ** (-0.5)) for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch, verbose)

    def _get_closed_form_lr(self):
        current_iter = self._step_count
        current_lrs = []
        for factor in self.factors:
            current_lr = factor * self.model_size ** (-0.5) * min(current_iter ** (-0.5), current_iter * self.warmup_iters ** (-1.5))
            current_lrs.append(current_lr)
        return current_lrs

    def get_lr(self):
        return self._get_closed_form_lr()

def generate_length_mask(lens, max_length=None):
    lens = torch.as_tensor(lens)
    N = lens.size(0)
    if max_length is None:
        max_length = max(lens)
    idxs = torch.arange(max_length).repeat(N).view(N, max_length)
    mask = (idxs < lens.view(-1, 1))
    return mask

def pack_wrapper(module, attn_feats, attn_feat_lens):
    packed, inv_ix = sort_pack_padded_sequence(attn_feats, attn_feat_lens)
    if isinstance(module, torch.nn.RNNBase):
        return pad_unsort_packed_sequence(module(packed)[0], inv_ix)
    else:
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)

def init(m, method="kaiming"):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        if method == "kaiming":
            nn.init.kaiming_uniform_(m.weight)
        elif method == "xavier":
            nn.init.xavier_uniform_(m.weight)
        else:
            raise Exception(f"initialization method {method} not supported")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        if method == "kaiming":
            nn.init.kaiming_uniform_(m.weight)
        elif method == "xavier":
            nn.init.xavier_uniform_(m.weight)
        else:
            raise Exception(f"initialization method {method} not supported")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        if method == "kaiming":
            nn.init.kaiming_uniform_(m.weight)
        elif method == "xavier":
            nn.init.xavier_uniform_(m.weight)
        else:
            raise Exception(f"initialization method {method} not supported")

def update_ss_ratio(config, num_iter):
    num_epoch = config["epochs"]
    mode = config["train_args"]["ss_args"]["ss_mode"]
    if mode == "exponential":
        config["train_args"]["ss_args"]["ss_ratio"] = 0.01 ** (1.0 / num_epoch / num_iter)
    elif mode == "linear":
        config["train_args"]["ss_args"]["ss_ratio"] -= (1.0 - config["train_args"]["ss_args"]["final_ss_ratio"]) / num_epoch / num_iter