import math
import torch
from torch.optim import SGD, Adam, AdamW, lr_scheduler
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.optim.lr_scheduler import StepLR, LambdaLR




def get_cosine_schedule_with_warmup(optimizer: torch.optim.Optimizer,
                                    num_warmup_steps: int,
                                    num_training_steps: int,
                                    num_cycles: float = 0.5,
                                    last_epoch: int = -1):
    """Create cosine learn rate scheduler with linear warm up built in."""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps))
        return max(
            0.0, 0.5 *
            (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def make_optimizer_and_schedule(args, model, params, num_it_per_ep):

    # Make optimizer
    param_list = model.parameters() if params is None else params
    lr = args['training']['learn_rate']
    optimizer = args['training']['optimizer']

    if optimizer == 'sgd':
        optimizer = SGD(param_list, lr, momentum=0.9)

    elif optimizer == 'adam':
        optimizer = Adam(param_list, lr)

    elif optimizer == 'adamw':
        optimizer = AdamW(param_list, lr)

    else:
        raise ValueError('Unknown optimizer {}'.format(optimizer))

    scheduler = args['training']['scheduler']["which"]
    num_epochs = args['training']['num_epochs']


    if scheduler == 'cos_warmup':

        num_warmup_steps = args['training']['scheduler']['params']['num_warmup_steps']
        if isinstance(num_warmup_steps, float):  # fraction of total train
            args['training']['scheduler']['params']['num_warmup_steps'] = int(
                num_warmup_steps * num_epochs * num_it_per_ep)

        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_training_steps=num_it_per_ep * num_epochs,
                                                    **args['training']['scheduler']['params'])

    elif scheduler == 'step_lr':

        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.scheduler.step_lr.step_size, gamma=args.scheduler.step_lr.gamma)

    elif args.scheduler.name == 'CosineLR':
        scheduler = CosineLRScheduler(optimizer, t_initial=args.train.epochs, lr_min=args.scheduler.cosine.lr_min,
                                     k_decay=args.scheduler.cosine.k_decay, warmup_t=args.scheduler.cosine.warmup_t,
                                     warmup_lr_init=args.scheduler.cosine.warmup_lr_init)

    else:
        raise ValueError('Unknown scheduler {}'.format(args.scheduler.name))



    return optimizer, scheduler



if __name__ =="__main__":

    from timm import create_model
    from timm.optim import create_optimizer
    from types import SimpleNamespace

    model = create_model('resnet34')

    args = SimpleNamespace()
    args.weight_decay = 0
    args.lr = 5e-4
    args.opt = 'adam'
    args.momentum = 0.9

    optimizer = create_optimizer(args, model)

    from matplotlib import pyplot as plt


    def get_lr_per_epoch(scheduler, num_epoch):
        lr_per_epoch = []
        for epoch in range(num_epoch):
            lr_per_epoch.append(scheduler.get_epoch_values(epoch))
        return lr_per_epoch

    num_epoch = 30
    scheduler = CosineLRScheduler(optimizer, t_initial=num_epoch, k_decay=1, lr_min=1e-5, warmup_t=0,
                                  warmup_lr_init=1e-6,
                                  )

    lr_per_epoch = []
    for i in range(num_epoch):
        scheduler.step(i)
        lr_per_epoch.append(optimizer.param_groups[0]['lr'])

    plt.plot([i for i in range(num_epoch)], lr_per_epoch)
    plt.show()




