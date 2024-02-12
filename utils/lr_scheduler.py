import math
import torch


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

class WarmupLinearSchedule(torch.optim.lr_scheduler._LRScheduler):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, gramma,update_epoch,last_epoch=-1,verbose=False):
        self.warmup_steps = warmup_steps
        self.gramma = gramma
        self.update_epoch = update_epoch
        super().__init__(optimizer,last_epoch,verbose)
    
    def _get_closed_form_lr(self):
        current_iter = self._step_count
        warmup_coeff = 1.0
        current_lrs = []
        if (current_iter) < self.warmup_steps:
            warmup_coeff = (current_iter) / self.warmup_steps
        if (current_iter) < self.warmup_steps:
            current_lr = warmup_coeff * self.base_lrs[0]    
        else:
            current_lr = warmup_coeff * self.base_lrs[0] * math.pow(self.gramma,(current_iter - self.warmup_steps)//self.update_epoch)

        current_lrs.append(current_lr)
        
        return current_lrs
    
    def get_lr(self):
        return self._get_closed_form_lr()


if __name__ == "__main__":
    model = torch.nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), 0.0005)
    epochs = 25
    iters = 1
    scheduler = WarmupLinearSchedule(optimizer, 5,0.1,10)#warmup_steps, gramma,update_epoch,last_epoch=-1,verbose=False):     warmup_steps: 5
    criterion = torch.nn.MSELoss()
    lrs = []
    for epoch in range(1, epochs + 1):
        for iteration in range(1, iters + 1):
            optimizer.zero_grad()
            x = torch.randn(4, 10)
            y = torch.randn(4, 5)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            lr = scheduler.get_last_lr()[0]
            print(f"lr: {lr}")
            scheduler.step()
            # print(f"lr: {scheduler.get_last_lr()}")
            lrs.append(scheduler.get_last_lr())
    # import matplotlib.pyplot as plt
    # plt.plot(list(range(1, len(lrs) + 1)), lrs, '-o', markersize=1)
    # plt.legend(loc="best")
    # plt.xlabel("Iteration")
    # plt.ylabel("LR")

    # plt.savefig("lr_curve.png", dpi=100)
