# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import sys
import logging
from torch._C import device
import yaml
import torch
import numpy as np
import pandas as pd
from utils import score_util
import sklearn.preprocessing as pre
from pprint import pformat

sys.path.append(os.getcwd())

def load_pretrained_model(model: torch.nn.Module, pretrained, outputfun):
    if not os.path.exists(pretrained):
        outputfun(f"Loading pretrained model from {pretrained} failed!")
        return
    state_dict = torch.load(pretrained, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in state_dict.items() if (k in model_dict) and (
            model_dict[k].shape == v.shape)
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=True)


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


def encode_labels(labels: pd.Series, encoder=None):
    """encode_labels

    Encodes labels

    :param labels: pd.Series representing the raw labels e.g., Speech, Water
    :param encoder (optional): Encoder already fitted 
    returns encoded labels (one hot) and the encoder
    """
    assert isinstance(labels, pd.Series), "Labels need to series"
    if not encoder:
        encoder = pre.LabelEncoder()
        encoder.fit(labels)
    labels_encoded = encoder.transform(labels)
    return labels_encoded.tolist(), encoder


def parse_config_or_kwargs(config_file, **kwargs):
    with open(config_file) as con_reader:
        yaml_config = yaml.load(con_reader, Loader=yaml.FullLoader)
    # passed kwargs will override yaml config
    return dict(yaml_config, **kwargs)


def store_yaml(config, config_file):
    with open(config_file, "w") as con_writer:
        yaml.dump(config, con_writer, default_flow_style=False)


def parse_augments(augment_list):
    """parse_augments
    parses the augmentation string in configuration file to corresponding methods

    :param augment_list: list
    """
    from datasets import augment

    specaug_kwargs = {"timemask": False, "freqmask": False, "timewarp": False}
    augments = []
    for transform in augment_list:
        if transform == "timemask":
            specaug_kwargs["timemask"] = True
        elif transform == "freqmask":
            specaug_kwargs["freqmask"] = True
        elif transform == "timewarp":
            specaug_kwargs["timewarp"] = True
        elif transform == "randomcrop":
            augments.append(augment.random_crop)
        elif transform == "timeroll":
            augments.append(augment.time_roll)
    augments.append(augment.spec_augment(**specaug_kwargs))
    return augments


def criterion_improver(mode):
    assert mode in ("loss", "acc", "score")
    best_value = np.inf if mode == "loss" else 0

    def comparator(x, best_x):
        return x < best_x if mode == "loss" else x > best_x

    def inner(x):
        nonlocal best_value

        if comparator(x, best_value):
            best_value = x
            return True
        return False
    return inner


def log_results(engine,
                optimizer,
                val_evaluator,
                val_dataloader,
                outputfun=sys.stdout.write,
                train_metrics=["loss", "accuracy"],
                val_metrics=["loss", "accuracy"],
                ):
    train_results = engine.state.metrics
    val_evaluator.run(val_dataloader)
    val_results = val_evaluator.state.metrics
    output_str_list = [
        "Validation Results - Epoch : {:<4}".format(engine.state.epoch)
    ]
    for metric in train_metrics:
        output = train_results[metric]
        if isinstance(output, torch.Tensor):
            output = output.item()
        output_str_list.append("{} {:<5.2g} ".format(
            metric, output))
    for metric in val_metrics:
        output = val_results[metric]
        if isinstance(output, torch.Tensor):
            output = output.item()
        output_str_list.append("{} {:5<.2g} ".format(
            metric, output))
    lr = optimizer.param_groups[0]["lr"]
    output_str_list.append(f"lr {lr:5<.2g} ")

    outputfun(" ".join(output_str_list))


def run_val(engine, evaluator, dataloader):
    evaluator.run(dataloader)


def save_model_on_improved(engine,
                           criterion_improved, 
                           metric_key,
                           dump,
                           save_path):
    if criterion_improved(engine.state.metrics[metric_key]):
        torch.save(dump, save_path)
    # torch.save(dump, str(Path(save_path).parent / "model.last.pth"))


def update_lr(engine, scheduler, metric=None):
    if scheduler.__class__.__name__ == "ReduceLROnPlateau":
        assert metric is not None, "need validation metric for ReduceLROnPlateau"
        val_result = engine.state.metrics[metric]
        scheduler.step(val_result)
    else:
        scheduler.step()


def update_ss_ratio(engine, config, num_iter):
    num_epoch = config["epochs"]
    mode = config["ss_args"]["ss_mode"]
    if mode == "exponential":
        config["ss_args"]["ss_ratio"] = 0.01 ** (1.0 / num_epoch / num_iter)
    elif mode == "linear":
        config["ss_args"]["ss_ratio"] -= (1.0 - config["ss_args"]["final_ss_ratio"]) / num_epoch / num_iter


def generate_length_mask(lens):
    lens = torch.as_tensor(lens)
    N = lens.size(0)
    T = max(lens)
    idxs = torch.arange(T).repeat(N).view(N, T)
    mask = (idxs < lens.view(-1, 1))
    return mask


def mean_with_lens(features, lens):
    """
    features: [N, T, ...] (assume the second dimension represents length)
    lens: [N,]
    """
    lens = torch.as_tensor(lens)
    mask = generate_length_mask(lens).to(features.device) # [N, T]

    feature_mean = features * mask.unsqueeze(-1)
    feature_mean = feature_mean.sum(1) / lens.unsqueeze(1).to(features.device)
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


class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.0,device = 0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.device = device

    def forward(self, logit, target):
        pred = logit.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            # print(target.data.unsqueeze(1).long())
            true_dist.scatter_(1, target.unsqueeze(1).long().to(self.device), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class Normal_kl_loss(torch.nn.Module):
    def __init__(self,device = 0, dim=-1):
        super(Normal_kl_loss, self).__init__()
        self.dim = dim
        self.device = device

    def forward(self, mu1, lv1, mu2, lv2):
        v1 = torch.exp(lv1)
        v2 = torch.exp(lv2)
        lstd1 = lv1 / 2.
        lstd2 = lv2 / 2.

        kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
        return kl.sum(-1).mean()

def fix_batchnorm(model):
    classname = model.__class__.__name__
    if classname.find("BatchNorm") != -1:
        model.eval()

class GMM_kl_loss(torch.nn.Module):
    def __init__(self,device = 0, dim=-1):
        super(GMM_kl_loss, self).__init__()
        self.dim = dim
        self.device = device

    def forward(self, mu1, lv1, c1, mu2, lv2, c2):
        c1 = torch.softmax(c1, dim=0)
        c2 = torch.softmax(c2, dim=0)
        v1 = torch.exp(lv1)
        v2 = torch.exp(lv2)
        lstd1 = lv1 / 2.
        lstd2 = lv2 / 2.

        kl_i = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
        kl_c = (c1 * (c1.log() - c2.log())).sum()
        kl = (c1 * kl_i).sum(0)
        
        return kl_c + kl.sum(-1).mean()
class Nscst_Loss(torch.nn.Module):
    def __init__(self,scorer,reduction="mean",sample_n=5, device = 0):
        super().__init__()
        self.reduction = reduction
        self.sample_n = sample_n
        self.scorer = scorer
        self.device = device
        self.pad_idx = 0
        self.start_idx = 1
        self.end_idx = 2
    
    def get_critical_reward(self, sampled_seqs,
                            keys, key2refs, vocabulary):
        sampled_seqs = sampled_seqs.cpu().numpy()

        sampled_score = score_util.compur_batch_score_samplen(sampled_seqs,
                                                       key2refs,
                                                       keys,
                                                       self.start_idx,
                                                       self.end_idx,
                                                       vocabulary,
                                                       self.scorer)
        
        sampled_score = torch.as_tensor(sampled_score).reshape(-1,self.sample_n)
        
        baseline_score = (sampled_score.sum(1, keepdim=True) - sampled_score) / (sampled_score.shape[1] - 1)

        reward = sampled_score - baseline_score
        
        reward = reward.reshape(-1)
        sampled_score = sampled_score.reshape(-1)
        return {"reward": reward, "score": sampled_score}

    def forward(self,output,keys, key2refs, vocabulary):

        loss_output = {}

        keys = [key for key in keys for _ in range(self.sample_n)]

        reward_score = self.get_critical_reward(output["sampled_seqs"],
                                                    keys,
                                                    key2refs,
                                                    vocabulary)
        
        loss_output["reward"] = torch.as_tensor(reward_score["reward"])
        loss_output["score"] = torch.as_tensor(reward_score["score"])

        reward_score["reward"] = np.reshape(reward_score["reward"],(-1,))
        reward = np.repeat(reward_score["reward"][:, np.newaxis], output["sampled_seqs"].size(-1), 1)
        mask = (output["sampled_seqs"] != self.end_idx).float()
        mask = torch.cat([torch.ones(mask.size(0), 1), mask[:, :-1]], 1)
        mask = torch.as_tensor(mask).float()
        loss = - output["sampled_logprobs"] * reward * mask
        loss = loss.to(self.device)

        # loss: [N, max_length]
        loss = torch.sum(loss, dim=1).mean()

        loss_output["loss"] = loss
        loss_output["reward"] = loss_output["reward"].mean()

        return loss_output

class scst_Loss(torch.nn.Module):
    def __init__(self,scorer,reduction="mean", device = 0):
        super().__init__()
        self.reduction = reduction
        self.scorer = scorer
        self.device = device
        self.pad_idx = 0
        self.start_idx = 1
        self.end_idx = 2

    def get_critical_reward(self, greedy_seqs, sampled_seqs,
                                 keys, key2refs, vocabulary, scorer):
        # greedy_seqs, sampled_seqs: [N, max_length]
        greedy_seqs = greedy_seqs.cpu().numpy()
        sampled_seqs = sampled_seqs.cpu().numpy()

        sampled_score = score_util.compute_batch_score(sampled_seqs,
                                                       key2refs,
                                                       keys,
                                                       self.start_idx,
                                                       self.end_idx,
                                                       vocabulary,
                                                       scorer)
        greedy_score = score_util.compute_batch_score(greedy_seqs, 
                                                      key2refs,
                                                      keys,
                                                      self.start_idx,
                                                      self.end_idx,
                                                      vocabulary,
                                                      scorer)
        reward = sampled_score - greedy_score
        return {"reward": reward, "score": sampled_score}

    def forward(self,output,keys, key2refs, vocabulary):


        reward_score = self.get_critical_reward(output["greedy_seqs"],
                                                output["sampled_seqs"],
                                                keys,
                                                key2refs,
                                                vocabulary,
                                                self.scorer)
        
        output["reward"] = torch.as_tensor(reward_score["reward"])
        output["score"] = torch.as_tensor(reward_score["score"])

        reward = np.repeat(reward_score["reward"][:, np.newaxis], output["sampled_seqs"].size(-1), 1)
        reward = torch.as_tensor(reward).float()
        mask = (output["sampled_seqs"] != self.end_idx).float()
        mask = torch.cat([torch.ones(mask.size(0), 1), mask[:, :-1]], 1)
        mask = torch.as_tensor(mask).float()
        loss = - output["sampled_logprobs"] * reward * mask
        loss = loss.to(self.device)
        # loss: [N, max_length]
        loss = torch.sum(loss, dim=1).mean()
        # loss = torch.sum(loss) / torch.sum(mask)
        output["loss"] = loss

        return output
def repeat_tensor(x, n):
    return x.unsqueeze(0).repeat(n, *([1] * len(x.shape)))