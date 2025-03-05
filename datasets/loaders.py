from .improc import get_srh_aug_list, get_srh_base_aug, get_dynacl_aug_v1, get_ft_aug_v1
from .srh_dataset import HiDiscDataset, OpenSRHDataset
import os
from functools import partial
from torchvision.transforms import Compose
import torch
from helpers import get_world_size, get_rank

def get_dataloaders(cf, strength=1.0, dynamic_aug=False):
    """Create dataloader for contrastive experiments."""

    if dynamic_aug:

        dynamic_aug_version = cf["data"]["dynamic_aug_version"]

        if dynamic_aug_version == "v0":
            train_transform = Compose(get_srh_aug_list(cf["data"]["train_augmentation"], dyanamic_aug=True, strength=strength))
            val_transform = Compose(get_srh_aug_list(cf["data"]["valid_augmentation"]))
        elif dynamic_aug_version == "v1":
            train_transform = Compose(get_dynacl_aug_v1(strength=strength))
            val_transform = Compose(get_srh_aug_list(cf["data"]["valid_augmentation"]))
        else:
            raise ValueError(f"Dynamic Augmentation version {dynamic_aug_version} not supported")

    else:
        train_transform = Compose(get_srh_aug_list(cf["data"]["train_augmentation"]))
        val_transform = Compose(get_srh_aug_list(cf["data"]["valid_augmentation"]))

    train_dset = HiDiscDataset(
        data_root=cf["data"]["db_root"],
        meta_json=cf["data"]["meta_json"],
        meta_split_json=cf["data"]["meta_split_json"],
        studies="train",
        transform=train_transform,
        balance_study_per_class=cf["data"]["balance_study_per_class"],
        num_slide_samples=cf["data"]["hidisc"]["num_slide_samples"],
        num_patch_samples=cf["data"]["hidisc"]["num_patch_samples"],
        num_transforms=cf["data"]["hidisc"]["num_transforms"])
    val_dset = HiDiscDataset(
        data_root=cf["data"]["db_root"],
        meta_json=cf["data"]["meta_json"],
        meta_split_json=cf["data"]["meta_split_json"],
        studies="val",
        transform=val_transform,
        balance_study_per_class=False,
        num_slide_samples=cf["data"]["hidisc"]["num_slide_samples"],
        num_patch_samples=cf["data"]["hidisc"]["num_patch_samples"],
        num_transforms=cf["data"]["hidisc"]["num_transforms"])



    if cf['distributed']['distributed']:
        num_tasks = get_world_size()
        global_rank = get_rank()
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        train_loader = torch.utils.data.DataLoader(train_dset, batch_size=cf['training']['batch_size'], sampler=train_sampler, drop_last=False)
        val_loader = torch.utils.data.DataLoader(val_dset, batch_size=cf['training']['batch_size'], sampler=val_sampler, drop_last=False)


    else:

        train_loader = torch.utils.data.DataLoader(train_dset, batch_size=cf['training']['batch_size'], shuffle=True, drop_last=False)
        val_loader = torch.utils.data.DataLoader(val_dset, batch_size=cf['training']['batch_size'], shuffle=True, drop_last=False)

    return train_loader, val_loader



def get_dataloaders_ft(cf):
    """Create dataloader for contrastive experiments."""

    aug_version = cf["data"]["aug_version"]

    if aug_version == "v0":
        train_transform = Compose(get_srh_aug_list(cf["data"]["train_augmentation"]))
        val_transform = Compose(get_srh_base_aug())
    elif aug_version == "v1":
        train_transform = Compose(get_ft_aug_v1())
        val_transform = Compose(get_srh_base_aug())
    else:
        raise ValueError(f"Augmentation version {aug_version} not supported")



    train_dset = OpenSRHDataset(
        data_root=cf["data"]["db_root"],
        studies="train",
        transform=train_transform,
        balance_patch_per_class=cf["data"]["balance_study_per_class"])

    val_dset = OpenSRHDataset(
        data_root=cf["data"]["db_root"],
        studies="val",
        transform=val_transform,
        balance_patch_per_class=False)




    if cf['distributed']['distributed']:
        num_tasks = get_world_size()
        global_rank = get_rank()
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        train_loader = torch.utils.data.DataLoader(train_dset, batch_size=cf['training']['batch_size'], sampler=train_sampler, drop_last=False)
        val_loader = torch.utils.data.DataLoader(val_dset, batch_size=cf['training']['eval_batch_size'], sampler=val_sampler, drop_last=False)


    else:

        train_loader = torch.utils.data.DataLoader(train_dset, batch_size=cf['training']['batch_size'], shuffle=True, drop_last=False)
        val_loader = torch.utils.data.DataLoader(val_dset, batch_size=cf['training']['eval_batch_size'], shuffle=False, drop_last=False)

    return train_loader, val_loader




def get_num_worker():
    """Estimate number of cpu workers."""
    try:
        num_worker = len(os.sched_getaffinity(0))
    except Exception:
        num_worker = os.cpu_count()

    if num_worker > 1:
        return num_worker - 1
    else:
        return torch.cuda.device_count() * 4
