import torch
import torch.nn as nn

from .attack import Attack


class FFGSM(Attack):
    r"""
    New FGSM proposed in 'Fast is better than free: Revisiting adversarial training'
    [https://arxiv.org/abs/2001.03994]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 10/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.


    """
    def __init__(self, model, eps=8/255, alpha=10/255):
        super().__init__("FFGSM", model)
        self.eps = eps
        self.alpha = alpha
        self.supported_mode = ['default', 'targeted']

    def forward(self, images, labels, dual_bn):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images + torch.randn_like(images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        adv_images.requires_grad = True

        outputs = self.get_logits(adv_images, dual_bn)

        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, adv_images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = adv_images + self.alpha*grad.sign()
        delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


if __name__=="__main__":
    import torchvision
    model = torchvision.models.resnet18(pretrained=True)
    img = torch.randn(1, 3, 224, 224)
    labels = torch.tensor([1])
    attack = FFGSM(model = model, mean=(0.1, 0.1, 0.1), std=(0.1,0.1,0.1))
    adv_images = attack(img, labels)