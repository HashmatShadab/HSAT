import json
import logging
import os
from functools import partial
from typing import Dict, Any
import torch
import torchvision

import dill
import hydra
import numpy as np
import torch
import wandb
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from datasets.loaders import get_dataloaders_ft
from helpers import init_distributed_mode, get_rank, is_main_process, get_world_size
from models import MLP, resnet_backbone, ContrastiveLearningNetwork
from models.resnet_multi_bn_stl import resnet50 as resnet50_multi_bn
from models import resnetv2_50, resnetv2_50_gn
from models import timm_wideresnet50_2, timm_resnet50, timm_resnetv2_50
from scheduler import make_optimizer_and_schedule
import torch
from helpers import  accuracy, MetricLogger, setup_seed
from timm.layers import convert_sync_batchnorm
from attacks import PGD, FGSM, FFGSM

def plot_grid(w, save=False, name="grid.png"):
    import matplotlib.pyplot as plt
    grid_img = torchvision.utils.make_grid(w)
    plt.imshow(grid_img.permute(1,2,0).cpu())
    if save:
        plt.savefig(name)
    plt.show()




log = logging.getLogger(__name__)


class FineTuneModel(torch.nn.Module):
    """
    HiDiscModel

    This class represents a High-dimensional Contrastive Learning Model. It is a subclass of `torch.nn.Module`.

    Attributes:
        cf_ (Dict[str, Any]): A dictionary containing the configuration parameters for the model.
        model (ContrastiveLearningNetwork): The main model used for high-dimensional contrastive learning.

    Methods:
        __init__(self, cf: Dict[str, Any]): Initializes the HiDiscModel object.
        forward(self, img): Performs forward pass on the input image.

    """

    def __init__(self, cf: Dict[str, Any], num_classes: int):
        super().__init__()
        self.cf_ = cf

        if cf["model"]["backbone"] == "resnet50":
            backbone = resnet_backbone(arch=cf["model"]["backbone"])
        elif cf["model"]["backbone"] == "resnet50_multi_bn":
            backbone = resnet50_multi_bn()
        elif cf["model"]["backbone"] == "resnetv2_50":
            backbone = resnetv2_50()
        elif cf["model"]["backbone"] == "resnetv2_50_gn":
            backbone = resnetv2_50_gn()
        elif cf["model"]["backbone"] == "wide_resnet50_2":
            backbone = timm_wideresnet50_2(pretrained=False)
        elif cf["model"]["backbone"] == "resnet50_timm":
            backbone = timm_resnet50(pretrained=False)
        elif cf["model"]["backbone"] == "resnetv2_50_timm":
            backbone = timm_resnetv2_50(pretrained=False)
        elif cf["model"]["backbone"] == "resnet50_timm_pretrained":
            backbone = timm_resnet50(pretrained=True)
        elif cf["model"]["backbone"] == "resnetv2_50_timm_pretrained":
            backbone = timm_resnetv2_50(pretrained=True)
        elif cf["model"]["backbone"] == "wide_resnet50_2_pretrained":
            backbone = timm_wideresnet50_2(pretrained=True)
        else:
            raise NotImplementedError()

        self.bb = backbone
        #self.fc = torch.nn.Linear(in_features=2048, out_features=num_classes)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.BatchNorm1d(512),  # Helps with stable learning
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_classes)
        )

    def forward(self, img, bn_name=None):

        if bn_name is not None:
            features = self.bb(img, bn_name)
        else:
            features = self.bb(img)

        logits = self.fc(features)

        return logits


@hydra.main(version_base=None, config_path="conf", config_name="ft")
def main(args):
    """
    Entry point of the program.
    """

    log.info("Info level message")
    # log.debug("Debug level message")

    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Orig working directory    : {get_original_cwd()}")

    setup_seed(args['infra']['seed'])

    # Initialize DDP if needed
    init_distributed_mode(args.distributed)

    if is_main_process():
        log.info(OmegaConf.to_yaml(args))  # log.info all the command line arguments

        # Create the output director if not exits
    if get_rank() == 0 and args.out_dir is not None:

        if args.wandb.use:
            wandb.init(project=args.wandb.project, entity=args.wandb.entity, mode=args.wandb.mode,
                       name=args.wandb.exp_name, group=args.wandb.group_name)

        log.info(f"Saving output to {os.path.join(os.getcwd())}")

    train_loader, validation_loader = get_dataloaders_ft(args)

    model = FineTuneModel(args, num_classes=args.model.num_classes)
    dual_bn = True if args.model.backbone == "resnet50_multi_bn" else False
    model.to(device="cuda")



    if args.model.finetuning == 'linear':
        # Freeze the backbone
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    elif args.model.finetuning == 'full':
        parameters =list(model.parameters())
    else:
        raise NotImplementedError()


    log.info(f'==> [Number of parameters of the model is {len(parameters)}]')

    if args.model.start_from_ssl_ckpt:

        if not os.path.isfile(args.model.start_from_ssl_ckpt):
            print(f": No checkpoints found in {os.getcwd()}")
            raise FileNotFoundError
        """
        Loads only model weights for evaluation.
        if restart_from_ckpt, then model weights will be loaded from the output dir,
        no need to pass the checkpoints path.

        """
        log.info(f"Loading SSL weights from checkpoint {args.model.start_from_ssl_ckpt}")
        checkpoint = torch.load(args.model.start_from_ssl_ckpt, pickle_module=dill)
        model_weights = checkpoint["model"]
        # remove the module from the keys
        model_weights = {k.replace("module.", ""): v for k, v in model_weights.items()}
        # remove model.model from the keys
        model_weights = {k.replace("model.", ""): v for k, v in model_weights.items()}

        msg = model.load_state_dict(model_weights, strict=False)
        log.info(f"Loaded SSL weights in model with msg: {msg}")




    parma_list = list(filter(lambda p: p.requires_grad, model.parameters()))

    # Define loss, create optimizer and scheduler
    num_it_per_ep = len(train_loader) // get_world_size()
    optimizer, scheduler = make_optimizer_and_schedule(args, model, parma_list, num_it_per_ep)

    criterion = torch.nn.CrossEntropyLoss()

    # print total number of learnable parameters
    if is_main_process():
        log.info(f"Total number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


    if args.distributed.distributed:
        if args.model.backbone == "resnet50_multi_bn" or args.model.backbone == "resnet50":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            model = convert_sync_batchnorm(model)
        unused_params = True if args.model.backbone == "resnet50_multi_bn" else False
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.distributed.gpu],
                                                          find_unused_parameters=unused_params,
                                                          broadcast_buffers=False)

    start_epoch, loss = restart_from_checkpoint("checkpoint.pth", model, optimizer, scheduler)

    # Perform evaluation and exit if `eval_only` is set
    if args.eval_only:
        # Validate the model
        validate_clean(validation_loader, model, criterion)
        return  # Safe exit

    if args.training.train_attack == 'pgd':
        train_attack = PGD(model=model, steps=args.training.attack_steps, eps=args.training.attack_eps/255.0)
    else:
        train_attack = False

    eval_attack = FFGSM(model=model, eps=8/255.0)

    # Training loop
    best_clean_accuracy = 0
    best_adv_accuracy = 0
    for epoch in range(start_epoch, args.training.num_epochs):
        # Train for one epoch
        train_stats = train_one_epoch(epoch=epoch, train_loader=train_loader, model=model,
                                      optimizer=optimizer, criterion=criterion, scheduler=scheduler,
                                      dual_bn=dual_bn, train_attack=train_attack)

        val_stats = validate_clean(validation_loader, model, criterion, dual_bn=dual_bn)
        adv_stats = validate_adv(validation_loader, model, criterion, dual_bn=dual_bn, eval_attack=eval_attack)


        #  Save the checkpoints
        if (epoch + 1) % args.training.save_checkpoint_interval == 0 and is_main_process():
            save_checkpoints(epoch + 1, model, optimizer, scheduler, train_stats,
                             name=f'checkpoint_{epoch + 1}.pth')

        if is_main_process():
            save_checkpoints(epoch + 1, model, optimizer, scheduler, train_stats,
                             name=f'checkpoint.pth')

            clean_acc_for_epoch = val_stats['acc1']
            adv_acc_for_epoch = adv_stats['acc1']
            if clean_acc_for_epoch < best_clean_accuracy:
                best_clean_accuracy = clean_acc_for_epoch
                save_checkpoints(epoch + 1, model, optimizer, scheduler, train_stats,
                                 name=f'best_clean_acc_checkpoint.pth')
                log.info(f"==> [Best Clean Acc: {best_clean_accuracy}]")
            if adv_acc_for_epoch < best_adv_accuracy :
                best_adv_accuracy = adv_acc_for_epoch
                save_checkpoints(epoch + 1, model, optimizer, scheduler, train_stats,
                                 name=f'best_adv_acc_checkpoint.pth')
                log.info(f"==> [Best Adv Acc: {best_adv_accuracy}]")

        # Log the epoch stats
        log_stats_train = {
            'Epoch': epoch,
            **{f'train_{key}': value for key, value in train_stats.items()},
            **{f'val_{key}': value for key, value in val_stats.items() if "acc5" not in key},
            **{f'adv_val_{key}': value for key, value in adv_stats.items() if "acc5" not in key}
        }

        if args.out_dir and is_main_process():
            with open("log.txt", mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats_train) + "\n")
            if args.wandb.use:
                wandb.log(log_stats_train)






def train_one_epoch(epoch, train_loader, model,
                    optimizer, criterion, scheduler, print_freq=50, dual_bn=False, train_attack=None):

    """
    :param epoch: The current epoch number.
    :param train_loader: The data loader for training data.
    :param model: The model to be trained.
    :param optimizer: The optimizer used to update the model's parameters.
    :param criterion: The loss function used to calculate the loss.
    :param scheduler: The learning rate scheduler.
    :param print_freq: The frequency at which to print the training progress.
    :return: A dictionary containing the averaged metrics from the training epoch.

    """

    # Distributed metric logger
    metric_logger = MetricLogger(delimiter="  ")
    header = f'Train: Epoch {epoch}'

    # Switch to training mode
    model.train()


    for i, batch in enumerate(metric_logger.log_every(train_loader, print_freq, header)):

        # Move the tensors to the GPUs
        images = batch["image"]
        images = images.to("cuda", non_blocking=True)
        targets = batch["label"].to("cuda", non_blocking=True)

        if train_attack:
            adv_images, adv_images_losses = train_attack(images, targets, dual_bn=dual_bn)
        else:
            adv_images = images
        outputs = model(adv_images, 'pgd') if dual_bn else model(adv_images)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Synchronize
        torch.cuda.synchronize()


        metric_logger.update(loss=loss.item())
        metric_logger.update(cost_adv_diff=adv_images_losses[1] - adv_images_losses[0])



        # Add LRs to the metric logger ass well
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    log.info(f"Averaged stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




def validate_clean(val_loader, model, criterion, dual_bn=False):
    # Distributed metric logger
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # Switch to evaluation mode
    model.eval()

    # Evaluate and return the loss and accuracy
    with torch.no_grad():
        for i, batch in enumerate(metric_logger.log_every(val_loader, 10, header)):
            # Move the tensors to the GPUs
            # Move the tensors to the GPUs
            images = batch["image"]
            images = images.to("cuda", non_blocking=True)
            targets = batch["label"].to("cuda", non_blocking=True)

            # Forward pass to the network
            outputs = model(images, 'normal') if dual_bn else model(images)


            # Calculate loss
            loss = criterion(outputs, targets)

            # Measure accuracy
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            acc5 = accuracy(outputs, targets, topk=(5,))[0]

            # Update the losses & top1 accuracy list
            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    log.info(f"* Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate_adv(val_loader, model, criterion, dual_bn=False, eval_attack=None):
    # Distributed metric logger
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Adv Test:'

    # Switch to evaluation mode
    model.eval()

    # Evaluate and return the loss and accuracy
    for i, batch in enumerate(metric_logger.log_every(val_loader, 10, header)):
        # Move the tensors to the GPUs
        # Move the tensors to the GPUs
        images = batch["image"]
        images = images.to("cuda", non_blocking=True)
        targets = batch["label"].to("cuda", non_blocking=True)

        # Forward pass to the network
        adv_images = eval_attack(images, targets, dual_bn=dual_bn)
        outputs = model(adv_images, 'pgd') if dual_bn else model(images)


        # Calculate loss
        loss = criterion(outputs, targets)

        # Measure accuracy
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        acc5 = accuracy(outputs, targets, topk=(5,))[0]

        # Update the losses & top1 accuracy list
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    log.info(f"* Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def save_checkpoints(epoch, model, optimizer, scheduler, train_stats,
                      name='checkpoint.pth'):
    """
    Save model, optimizer and scheduler to a checkpoint file inside out_dir.

    """
    print("Saving checkpoint to: ", os.path.join(os.getcwd(), name))
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'schedule': scheduler.state_dict() if scheduler else None,
        'loss': train_stats['loss'],
    },
        f"{name}")

def restart_from_checkpoint(checkpoint_name, model, optimizer,
                            scheduler):
    """
    New script for this codebase
    Loads model, optimizer and scheduler from a checkpoint. If the checkpoint is not found
    in the out_dir, returns 0 epoch.

    """
    if not os.path.isfile(checkpoint_name):
        print(f"Restarting: No checkpoints found in {os.getcwd()}")
        return 0, 0

    # open checkpoint file
    checkpoint = torch.load(checkpoint_name, pickle_module=dill)
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"=> Restarting from checkpoint {os.path.join(os.getcwd(), checkpoint_name)} (Epoch{start_epoch})")
    if "model" in checkpoint and checkpoint['model'] is not None:
        model_weights = checkpoint["model"]
        # remove the module from the keys
        model_weights = {k.replace("module.model.model", "module.model"): v for k, v in model_weights.items()}
        msg = model.load_state_dict(model_weights, strict=False)
        print("Load model with msg: ", msg)

    if "optimizer" in checkpoint and checkpoint['optimizer'] is not None:
        msg = optimizer.load_state_dict(checkpoint['optimizer'])
        print("Load optimizer with msg: ", msg)

    if "schedule" in checkpoint and checkpoint['schedule'] is not None:
        msg = scheduler.load_state_dict(checkpoint['schedule'])
        print("Load scheduler with msg: ", msg)
    else:
        print("No scheduler in checkpoint")

    return start_epoch, loss

if __name__ == "__main__":
    main()
