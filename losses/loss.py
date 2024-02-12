from typing import Dict

import numpy as np
import torch
import ignite.metrics as metrics
from ignite.engine.engine import Engine
from utils import score_util

from models.utils import generate_length_mask, mean_with_lens


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        self.reduction = reduction

    def forward(self, output: Dict):
        # logits: [bs, max_len, c]
        # targets: [bs, max_len]
        # lens: [bs]
        logits = output["logits"]
        targets = output["targets"]
        lens = output["lens"]
        c = logits.size(-1)
        loss = self.loss_fn(logits.reshape(-1, c), targets.reshape(-1))
        loss = loss.reshape(*targets.shape)
        mask = generate_length_mask(lens).to(logits.device)
        loss *= mask
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            loss = loss.sum() / mask.sum()
            return loss
        elif self.reduction == "sum":
            loss = loss.sum()
            return loss

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing=0.0, dim=-1, reduction="mean"):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim
        self.reduction = reduction

    def forward(self, output: Dict):
        # logits: [bs, max_len, c]
        # targets: [bs, max_len]
        # lens: [bs]
        logits = output["logits"]
        targets = output["targets"]
        lens = output["lens"]
        preds = logits.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(preds)
            true_dist.fill_(self.smoothing / (logits.size(-1) - 1))
            true_dist.scatter_(self.dim, targets.data.unsqueeze(self.dim), self.confidence)
        loss = torch.sum(-true_dist * preds, dim=self.dim)
        mask = generate_length_mask(lens).to(logits.device)
        loss *= mask
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            loss = loss.sum() / mask.sum()
            return loss
        elif self.reduction == "sum":
            loss = loss.sum()
            return loss

class AugmentLossWrapper(torch.nn.Module):
    def __init__(self, loss_fn, eps=1e-12):
        super().__init__()
        self.loss_fn = loss_fn
        self.loss_fn.reduction = "none"
        self.eps = eps

    def forward(self, output: Dict):
        loss = self.loss_fn(output)
        cap_ids = output["cap_ids"]
        use_aug_prob = output["use_aug_prob"]
        aug_mask = np.array(["aug" not in cap_id for cap_id in cap_ids])
        use_aug = np.random.choice(
            [0, 1],
            size=(~aug_mask).sum(),
            p=[1 - use_aug_prob, use_aug_prob]
        )
        aug_mask[~aug_mask] = use_aug
        aug_mask = torch.as_tensor(aug_mask).to(loss.device)
        loss *= aug_mask.reshape(-1, 1)
        mask = generate_length_mask(output["lens"]).to(loss.device)
        mask *= aug_mask.reshape(-1, 1)
        return loss.sum() / (mask.sum() + self.eps)

def reparameterize_argmax(logits, dim=-1):
    # y = torch.softmax(logits, dim)
    y = logits
    shape = y.size()
    _, ind = y.max(dim=dim)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size()).to(logits.device)
    return torch.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1.0):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

class ConditionLossWrapper(torch.nn.Module):
    def __init__(self, loss_fn, dscrm, alpha=1, sample_method="argmax"):
        super().__init__()
        self.loss_fn = loss_fn
        self.dscrm = dscrm
        self.alpha = alpha
        self.sample_method = sample_method
        self.condition_fn = torch.nn.BCELoss()

    def forward(self, output: Dict):
        word_loss = self.loss_fn(output)
        logits = output["logits"]
        conditions = output["conditions"].to(logits.device)
        # preds = reparameterize_argmax(logits).to(logits.device)
        # preds = gumbel_softmax(logits).to(logits.device)
        if self.sample_method == "argmax":
            preds = reparameterize_argmax(logits)
        elif self.sample_method == "gumbel":
            preds = gumbel_softmax(logits)
        elif self.sample_method == "weighted":
            preds = torch.softmax(logits, -1)
        else:
            raise Exception(f"sample method {self.sample_method} not supported")
        preds = preds.to(logits.device)
        lens = output["lens"] - 1 # remove <eos>
        probs = self.dscrm({"caps": preds, "lens": lens})
        condition_loss = self.condition_fn(probs, conditions)
        loss = word_loss + self.alpha * condition_loss
        return loss, word_loss, condition_loss
        
class SpecificityLossWrapper(torch.nn.Module):
    def __init__(self, loss_fn, word_specificity, sentence_reduce="sum", alpha=1):
        super().__init__()
        self.loss_fn = loss_fn
        self.word_specificity = word_specificity # [vocab_size]
        self.alpha = alpha
        self.sentence_reduce = sentence_reduce
        self.condtion_fn = torch.nn.MSELoss()

    def forward(self, output: Dict):
        word_loss = self.loss_fn(output)
        logits = output["logits"]
        conditions = output["conditions"].to(logits.device)
        probs = torch.softmax(logits, dim=-1)
        cond_pred = torch.matmul(probs, self.word_specificity) # [N, T]
        lens = output["lens"] - 1 # remove <eos>
        if self.sentence_reduce == "sum":
            mask = generate_length_mask(lens, max_length=cond_pred.size(1)).to(logits.device)
            cond_pred *= mask
            cond_pred = cond_pred.sum(1)
        else:
            cond_pred = mean_with_lens(cond_pred, lens)
        condition_loss = self.condtion_fn(cond_pred, conditions)
        loss = word_loss + self.alpha * condition_loss
        return loss, word_loss, condition_loss

class Loss(metrics.Loss):

    def update(self, output: Dict) -> None:
        # logit: [bs, max_len, c]
        # target: [bs, max_len]
        # lens: [bs]
        lens = output["lens"]
        average_loss = self._loss_fn(output).detach()

        if len(average_loss.shape) != 0:
            raise ValueError("loss_fn did not return the average loss.")

        n = torch.sum(lens)
        self._sum += average_loss.to(self._device) * n
        self._num_examples += n

    @torch.no_grad()
    def iteration_completed(self, engine: Engine) -> None:
        output = self._output_transform(engine.state.output)
        self.update(output)

class Nscst_Loss(torch.nn.Module):
    def __init__(self,scorer,reduction="mean",sample_n=5):
        super().__init__()
        self.reduction = reduction
        self.sample_n = sample_n
        self.scorer = scorer
    
    def get_critical_reward(self, sampled_seqs,
                            keys, key2refs, vocabulary, scorer):
        sampled_seqs = sampled_seqs.cpu().numpy()

        sampled_score = score_util.compute_batch_score(sampled_seqs,
                                                       key2refs,
                                                       keys,
                                                       self.model.start_idx,
                                                       self.model.end_idx,
                                                       vocabulary,
                                                       scorer)
        
        baseline_score = (sampled_score.sum(1, keepdim=True) - sampled_score) / (sampled_score.shape[1] - 1)

        reward = sampled_score - baseline_score
        
        return {"reward": reward, "score": sampled_score}

    def forward(self,output,keys, key2refs, vocabulary):

        loss_output = {}

        keys = [key for key in keys for _ in range(self.sample_n)]

        reward_score = self.get_critical_reward(output["sampled_seqs"],
                                                    keys,
                                                    key2refs,
                                                    vocabulary,
                                                    self.scorer)
        

        reward_score["reward"] = np.reshape(reward_score["reward"],(-1,))
        reward = np.repeat(reward_score["reward"][:, np.newaxis], output["seqs"].size(-1), 1)
        reward = torch.as_tensor(reward).float()
        mask = (output["sampled_seqs"] != self.model.end_idx).float()
        mask = torch.cat([torch.ones(mask.size(0), 1), mask[:, :-1]], 1)
        mask = torch.as_tensor(mask).float()
        loss = - output["sampled_logprobs"] * reward * mask
        loss = loss.to(output["sampled_logprobs"].device)

        # loss: [N, max_length]
        loss = torch.sum(loss, dim=1).mean()

        loss_output["loss"] = loss

        return loss_output

class Bce_logits_Loss(torch.nn.Module):

    def __init__(self, pos_weight=1):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, input, target):
        epsilon = 1e-7
        #   self.pos_weight = self.pos_weight.to(input.device)
        # input = input.sigmoid()
        input_clamp = torch.clamp(input, min=1e-7, max=1-1e-7)
        my_bce_loss = -1 * (self.pos_weight * target * torch.log(input_clamp)
                            + (1 - target) * torch.log(1 - input_clamp))
        #   add_loss = (target - 0.5) ** 2 * 4
        mean_loss = my_bce_loss.mean()
        if torch.any(torch.isnan(mean_loss)):
            print(input_clamp)
            print(input)
            exit() 
        return mean_loss

class AdverseLossWrapper(torch.nn.Module):
    def __init__(self, criterion_loss,kl_loss, dscrm_model, alpha=1,beta=1,sample_method="argmax"):
        super().__init__()
        self.criterion_loss = criterion_loss
        self.kl_loss = kl_loss
        self.dscrm_model = dscrm_model
        self.alpha = alpha
        self.beta = beta
        self.sample_method = sample_method
        self.dscrm_loss = Bce_logits_Loss()
    
    def forward(self, output: Dict):
        loss_criterion = self.criterion_loss(output["packed_logits"], output["targets"])
        loss_kl = self.kl_loss(output["q_means"].to(output["packed_logits"].device),output["q_logs"].to(output["packed_logits"].device),output["p_means"].to(output["packed_logits"].device),output["p_logs"].to(output["packed_logits"].device))
        
        logits = output["logits"]
        label = output["label"].to(logits.device)
        if self.sample_method == "argmax":
            preds = reparameterize_argmax(logits)
        elif self.sample_method == "gumbel":
            preds = gumbel_softmax(logits)
        elif self.sample_method == "weighted":
            preds = torch.softmax(logits, -1)
        else:
            raise Exception(f"sample method {self.sample_method} not supported")
        
        preds = preds.to(logits.device)
        lens = output["lens"] - 1 # remove <eos>

        probs = self.dscrm_model({"audio_feats":output["audio_feats"],"feats_lens":output["feats_lens"] , "caps": preds, "lens": lens})
        loss_dscrm = self.dscrm_loss(probs, label)

        loss = loss_criterion + self.beta * loss_kl + self.alpha * loss_dscrm
        if self.alpha > 0:
            loss = loss_dscrm
        else:
            loss = loss_criterion + self.beta * loss_kl
        # print(loss_criterion)
        # print(loss_kl)
        # print(loss_dscrm)
        return loss,loss_criterion,loss_kl,loss_dscrm