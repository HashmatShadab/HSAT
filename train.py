import logging

import torch
import torchvision

from helpers import MetricLogger

log = logging.getLogger(__name__)
def plot_grid(w, save=False, name="grid.png"):
    import matplotlib.pyplot as plt
    grid_img = torchvision.utils.make_grid(w)
    plt.imshow(grid_img.permute(1,2,0).cpu())
    if save:
        plt.savefig(name)
    plt.show()

import os
import torch
from torchvision import transforms

def save_patient_images(im_reshaped, output_dir="patient_images"):
    """
    Save images in a structured folder format:
    patient_images/
    ├── patient_1/
    │   ├── image_1_original.png
    │   ├── image_1_augmented.png
    │   ├── image_2_original.png
    │   ├── image_2_augmented.png
    ├── patient_2/
    │   ├── image_1_original.png
    │   ├── image_1_augmented.png

    Args:
        im_reshaped (torch.Tensor): A CUDA tensor of shape (16, 3, 300, 300).
        output_dir (str): Directory where images will be saved.
    """

    # Ensure the tensor is on CPU before processing
    im_reshaped = im_reshaped.cpu()

    # Convert tensor to PIL images
    to_pil = transforms.ToPILImage()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for i in range(0, 16, 2):  # Step by 2 since each original has an augmented version
        patient_id = 1 if i < 8 else 2
        image_num = (i // 2) + 1 if patient_id == 1 else ((i - 8) // 2) + 1

        # Create patient folder
        patient_folder = os.path.join(output_dir, f"patient_{patient_id}")
        os.makedirs(patient_folder, exist_ok=True)

        # Save original image
        original_image = to_pil(im_reshaped[i])
        original_image.save(os.path.join(patient_folder, f"image_{image_num}_original.png"))

        # Save augmented image
        augmented_image = to_pil(im_reshaped[i + 1])
        augmented_image.save(os.path.join(patient_folder, f"image_{image_num}_augmented.png"))

    print("Images saved successfully!")





def pgd_attack(model, criterion, images, targets, shape, eps=1 / 255, alpha=1 / 255, iters=2, dual_bn=False, adv_loss_type='p_s_pt'):


    log.info(f"PGD Attack Model is in train mode: {model.training}")

    delta = torch.rand_like(images) * eps * 2 - eps
    delta = torch.nn.Parameter(delta)

    # Attack
    for _ in range(iters):
        # Forward pass to the model
        if dual_bn:
            outputs = model(images+delta, 'pgd')
        else:
            outputs = model(images + delta)

        outputs = outputs.reshape(*shape, outputs.shape[-1])

        model.zero_grad()

        losses = criterion(outputs, targets, loss_type=adv_loss_type)
        loss = losses["sum_loss"]
        # Backpropagate & estimate delta
        loss.backward()

        delta.data = delta.data + alpha * delta.grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(images + delta.data, min=0, max=1) - images

        log.info(f"PGD Attack Loss {_}: {loss.item()}")

    # Create final adversarial images
    adv_images = images + delta

    return adv_images.detach()


def pgd_attack_2(model, criterion, images, targets, shape, eps=1 / 255, alpha=1 / 255, iters=2, dual_bn=False, adv_loss_type='p_s_pt'):


    # Initialize delta and set its bounds
    delta = torch.zeros_like(images).cuda()
    delta.uniform_(-eps, eps)
    delta = torch.clamp(delta, 0 - images, 1 - images)
    delta.requires_grad = True

    log.info(f"PGD Attack Model is in train mode: {model.training}")


    # Attack
    for _ in range(iters):
        adv_images = images + delta
        # Forward pass to the model
        if dual_bn:
            outputs = model(adv_images, 'pgd')
        else:
            outputs = model(adv_images)

        outputs = outputs.reshape(*shape, outputs.shape[-1])

        model.zero_grad()

        losses = criterion(outputs, targets, loss_type=adv_loss_type)
        loss = losses["sum_loss"]
        # Backpropagate & estimate delta
        loss.backward()


        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]

        images_ = images[:, :, :, :]
        d = torch.clamp(d + alpha * torch.sign(g), min=-eps, max=eps)
        d = torch.clamp(d, 0 - images_, 1 - images_)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()
        # log the loss at each iteration
        log.info(f"PGD Attack Loss {_}: {loss.item()}")

    # Create final adversarial images
    adv_images = images + delta

    return adv_images.detach()




def train_one_epoch(epoch, train_loader, model,
                    optimizer, criterion, scheduler, print_freq=50, attack_type='pgd',
                    attack_eps=1/255, attack_alpha=1/255, attack_iters=7, dual_bn=False,
                    dynamic_aug=False, dynamic_weights_lamda=0.5, dynamic_strength=1.0,
                    only_adv=False, adv_loss_type='p_s_pt'):

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

    weight = 0.0

    for i, batch in enumerate(metric_logger.log_every(train_loader, print_freq, header)):

        # Move the tensors to the GPUs
        im_reshaped = batch["image"].reshape(-1, *batch["image"].shape[-3:])
        # orig_im = batch['base_image'].reshape(-1, *batch['base_image'].shape[-3:])
        im_reshaped = im_reshaped.to("cuda", non_blocking=True)
        targets = batch["label"].to("cuda", non_blocking=True)
        targets = targets.reshape(-1, 1)
        #save_patient_images(im_reshaped, output_dir=f"patient_images_{i}")

        if attack_type == 'pgd' or attack_type == 'pgd_2':
            model.eval()
            adv_images = pgd_attack(model=model, criterion=criterion, targets=targets, images=im_reshaped, eps=attack_eps/255.0,
                                    alpha=attack_alpha/255.0, iters=attack_iters, shape=batch["image"].shape[:4], dual_bn=dual_bn,
                                    adv_loss_type=adv_loss_type) if attack_type == 'pgd' else pgd_attack_2(model=model, criterion=criterion, targets=targets, images=im_reshaped, eps=attack_eps/255.0,
                                                                                                alpha=attack_alpha/255.0, iters=attack_iters, shape=batch["image"].shape[:4], dual_bn=dual_bn,
                                                                                                adv_loss_type=adv_loss_type)
            model.train()

        if attack_type.startswith("pgd") and only_adv:
            adv_outputs = model(adv_images, 'pgd') if dual_bn else model(adv_images)
            adv_outputs = adv_outputs.reshape(*batch["image"].shape[:4], adv_outputs.shape[-1])
            adv_losses = criterion(adv_outputs, targets)
            adv_loss = adv_losses["sum_loss"]
            clean_losses = {'sum_loss': torch.tensor(0.0), 'patient_loss': torch.tensor(0.0), 'slide_loss': torch.tensor(0.0), 'patch_loss': torch.tensor(0.0)}
            clean_loss = clean_losses["sum_loss"]

        elif attack_type.startswith("pgd") and not only_adv:
            adv_outputs = model(adv_images, 'pgd') if dual_bn else model(adv_images)
            adv_outputs = adv_outputs.reshape(*batch["image"].shape[:4], adv_outputs.shape[-1])
            adv_losses = criterion(adv_outputs, targets)
            adv_loss = adv_losses["sum_loss"]

            clean_outputs = model(im_reshaped, 'normal') if dual_bn else model(im_reshaped)
            clean_outputs = clean_outputs.reshape(*batch["image"].shape[:4], clean_outputs.shape[-1])
            clean_losses = criterion(clean_outputs, targets)
            clean_loss = clean_losses["sum_loss"]

        else:
            adv_losses = {'sum_loss': torch.tensor(0.0), 'patient_loss': torch.tensor(0.0), 'slide_loss': torch.tensor(0.0), 'patch_loss': torch.tensor(0.0)}
            adv_loss = adv_losses["sum_loss"]
            clean_outputs = model(im_reshaped, 'normal') if dual_bn else model(im_reshaped)
            clean_outputs = clean_outputs.reshape(*batch["image"].shape[:4], clean_outputs.shape[-1])
            clean_losses = criterion(clean_outputs, targets)
            clean_loss = clean_losses["sum_loss"]



        if not attack_type.startswith('pgd'):
            total_loss = clean_loss
        else:
            weight = dynamic_weights_lamda * (1 - dynamic_strength) if dynamic_aug else 0.0
            total_loss = adv_loss  if only_adv else ( (1 - weight)*clean_loss + (1 + weight)*adv_loss ) / 2


        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Synchronize
        torch.cuda.synchronize()

        # Update the metric logger with clean and adversarial losses
        metric_logger.update(total_loss=total_loss.item())
        metric_logger.update(clean_loss=clean_loss.item())
        metric_logger.update(adv_loss=adv_loss.item())
        metric_logger.update(clean_patient_loss=clean_losses["patient_loss"].item())
        metric_logger.update(clean_slide_loss=clean_losses["slide_loss"].item())
        metric_logger.update(clean_patch_loss=clean_losses["patch_loss"].item())
        metric_logger.update(adv_patient_loss=adv_losses["patient_loss"].item())
        metric_logger.update(adv_slide_loss=adv_losses["slide_loss"].item())
        metric_logger.update(adv_patch_loss=adv_losses["patch_loss"].item())
        metric_logger.update(weight_coefficent=weight)
        metric_logger.update(strength=dynamic_strength)

        if attack_type.startswith('pgd'):
            avg_increase = (adv_loss.item() - clean_loss.item())
            metric_logger.update(avg_increase=avg_increase)
        else:
            metric_logger.update(avg_increase=0.0)


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
    # log.info(f"Average stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
