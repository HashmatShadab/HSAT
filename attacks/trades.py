import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.autograd import Variable
from torchvision.transforms.functional import normalize


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


from attacks.attack import Attack
class TRADES(Attack):
    def __init__(self, model, mean=None, std=None, eps=0.007, alpha=0.001, steps=10):
        super().__init__("TRADES", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.set_normalization_used(mean=mean, std=std)
        self.supported_mode = ['default', 'targeted']
        self.criteria = nn.KLDivLoss(reduction="sum")

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        images_adv = images.detach() + 0.001 * torch.randn(images.shape).cuda().detach()

        for _ in range(self.steps):
            images_adv.requires_grad_()

            with torch.enable_grad():
                loss_kl = self.criteria(
                    F.log_softmax(self.get_logits(images_adv), dim=1),
                    F.softmax(self.get_logits(images), dim=1))

            grad = torch.autograd.grad(loss_kl, images_adv, retain_graph=False, create_graph=False)[0]
            images_adv = images_adv.detach() + self.alpha * grad.sign()
            delta = torch.clamp(images_adv - images, min=-self.eps, max=self.eps)
            images_adv = torch.clamp(images + delta, min=0.0, max=1.0).detach()
        print(f"Max grad {torch.max(grad)}               Min grad {torch.min(grad)}")
        return images_adv




def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
                eval_mode=False):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(reduction="sum")
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example

    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():


            loss_kl = criterion_kl(
                    F.log_softmax(model(normalize(x_adv, mean=mean, std=std)),
                                  dim=1),
                    F.softmax(model(normalize(x_natural, mean=mean, std=std)),
                              dim=1))


        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon),
                          x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    print(f"Max grad {torch.max(grad)}               Min grad {torch.min(grad)}")


    if eval_mode:
        print("Model is in eval mode")
    else:
        model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(normalize(x_natural, mean=mean, std=std))
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * criterion_kl(
        F.log_softmax(model(normalize(x_adv, mean=mean, std=std)), dim=1),
        F.softmax(model(normalize(x_natural, mean=mean, std=std)), dim=1))

    loss = loss_natural + beta * loss_robust
    return loss
