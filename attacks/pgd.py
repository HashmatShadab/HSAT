import torch
import torch.nn as nn
from torchvision.transforms.functional import normalize

from .attack import Attack
import torch
import torch.nn.functional as F


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.


    """
    def __init__(self, model, eps=0.3,
                 alpha=2/255, steps=40, random_start=True):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels, dual_bn):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        cost_list=[]
        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images, dual_bn)


            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)
            cost_list.append(cost.item())
            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            # print(f"Max grad: {torch.max(grad)}    Min grad: {torch.min(grad)}    Cost {cost.item()}")
            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        # only keep the first and last cost values
        cost_list = [cost_list[0], cost_list[-1]]
        return adv_images, cost_list





class PGD_KNN(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.


    """
    def __init__(self, model, eps=0.3,
                 alpha=2/255, steps=40, random_start=True, loss_fn="cosine"):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.loss_fn = loss_fn
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels, dual_bn):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)


        if self.loss_fn == "cosine":
            loss = nn.CosineSimilarity()
        elif self.loss_fn == "mse":
            loss = nn.MSELoss()
        elif self.loss_fn == "l1":
            loss = nn.L1Loss()

        adv_images = images.clone().detach()
        orig_images = images.clone().detach()


        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images, dual_bn)
            outputs = outputs.reshape(images.shape[0], -1)
            clean_outputs = self.get_logits(orig_images.clone().detach(), dual_bn)
            clean_outputs = clean_outputs.reshape(images.shape[0], -1)


            cost = loss(clean_outputs, outputs).mean()

            if self.loss_fn == "cosine":
                cost = -cost

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            print(f"  Cost {cost.item()}")
            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class BIM_KNN(Attack):
    r"""
    BIM or iterative-FGSM in the paper 'Adversarial Examples in the Physical World'
    [https://arxiv.org/abs/1607.02533]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)

    .. note:: If steps set to 0, steps will be automatically decided following the paper.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, loss_fn="cosine",
                 random_start=True):
        super().__init__("BIM", model)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps * 255 + 4, 1.25 * eps * 255))
        else:
            self.steps = steps
        self.supported_mode = ["default", "targeted"]
        self.loss_fn = loss_fn
        self.random_start = random_start

    def forward(self, images, labels, dual_bn):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)

        if self.loss_fn == "cosine":
            loss = nn.CosineSimilarity()
        elif self.loss_fn == "mse":
            loss = nn.MSELoss()
        elif self.loss_fn == "l1":
            loss = nn.L1Loss()

        adv_images = images.clone().detach()
        orig_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()


        for _ in range(self.steps):
            adv_images.requires_grad = True

            outputs = self.get_logits(adv_images, dual_bn)
            outputs = outputs.reshape(images.shape[0], -1)
            clean_outputs = self.get_logits(orig_images.clone().detach(), dual_bn)
            clean_outputs = clean_outputs.reshape(images.shape[0], -1)

            cost = loss(clean_outputs, outputs).mean()

            if self.loss_fn == "cosine":
                cost = -cost

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            print(f"  Cost {cost.item()}")

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            a = torch.clamp(orig_images - self.eps, min=0)
            b = (adv_images >= a).float() * adv_images + (
                adv_images < a
            ).float() * a  # nopep8
            c = (b > orig_images + self.eps).float() * (orig_images + self.eps) + (
                b <= orig_images + self.eps
            ).float() * b  # nopep8
            adv_images = torch.clamp(c, max=1).detach()

        return adv_images




class FFGSM_KNN(Attack):
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

    Examples::
        >>> attack = torchattacks.FFGSM(model, eps=8/255, alpha=10/255)
        >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, eps=8 / 255, alpha=10 / 255, loss_fn="cosine"):
        super().__init__("FFGSM", model)
        self.eps = eps
        self.alpha = alpha
        self.supported_mode = ["default", "targeted"]
        self.loss_fn = loss_fn

    def forward(self, images, labels, dual_bn):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)

        if self.loss_fn == "cosine":
            loss = nn.CosineSimilarity()
        elif self.loss_fn == "mse":
            loss = nn.MSELoss()
        elif self.loss_fn == "l1":
            loss = nn.L1Loss()

        orig_images = images.clone().detach()

        adv_images = images + torch.randn_like(images).uniform_(
            -self.eps, self.eps
        )  # nopep8
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        adv_images.requires_grad = True

        outputs = self.get_logits(adv_images, dual_bn)
        outputs = outputs.reshape(images.shape[0], -1)
        clean_outputs = self.get_logits(orig_images.clone().detach(), dual_bn)
        clean_outputs = clean_outputs.reshape(images.shape[0], -1)
        cost = loss(clean_outputs, outputs).mean()

        if self.loss_fn == "cosine":
            cost = -cost

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, adv_images, retain_graph=False, create_graph=False
        )[0]

        print(f"  Cost {cost.item()}")

        adv_images = adv_images + self.alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

class MIFGSM_KNN(Attack):
    r"""
    MI-FGSM in the paper 'Boosting Adversarial Attacks with Momentum'
    [https://arxiv.org/abs/1710.06081]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 10)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.MIFGSM(model, eps=8/255, steps=10, decay=1.0)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=8 / 255, alpha=2 / 255, steps=10, decay=1.0, loss_fn="cosine",
                 random_start=True):
        super().__init__("MIFGSM", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.supported_mode = ["default", "targeted"]
        self.loss_fn = loss_fn
        self.random_start = random_start

    def forward(self, images, labels, dual_bn):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)



        momentum = torch.zeros_like(images).detach().to(self.device)

        if self.loss_fn == "cosine":
            loss = nn.CosineSimilarity()
        elif self.loss_fn == "mse":
            loss = nn.MSELoss()
        elif self.loss_fn == "l1":
            loss = nn.L1Loss()

        orig_images = images.clone().detach()
        adv_images = images.clone().detach()


        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images, dual_bn)
            outputs = outputs.reshape(images.shape[0], -1)

            clean_outputs = self.get_logits(orig_images.clone().detach(), dual_bn)
            clean_outputs = clean_outputs.reshape(images.shape[0], -1)

            cost = loss(clean_outputs, outputs).mean()

            if self.loss_fn == "cosine":
                cost = -cost

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            print(f"  Cost {cost.item()}")
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images


class PGD_CLIP(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.


    """
    def __init__(self, model, eps=0.3,
                 alpha=2/255, steps=40, random_start=True, loss_fn="cosine", feature_loss="projection_features"):
        super().__init__("PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.loss_fn = loss_fn
        self._supported_mode = ['default', 'targeted']
        self.feature_loss = feature_loss

    def forward(self, images, labels, dual_bn):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)


        if self.loss_fn == "cosine":
            loss = nn.CosineSimilarity()
        elif self.loss_fn == "mse":
            loss = nn.MSELoss()
        elif self.loss_fn == "l1":
            loss = nn.L1Loss()

        adv_images = images.clone().detach()
        orig_images = images.clone().detach()


        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images, dual_bn)
            clean_outputs = self.get_logits(orig_images.clone().detach(), dual_bn)


            if self.feature_loss == "projection_features":
                outputs = outputs["projection_features"]  # Nx512
                clean_outputs = clean_outputs["projection_features"]  # Nx512

                outputs = outputs.reshape(images.shape[0], -1)  # Nx512
                clean_outputs = clean_outputs.reshape(images.shape[0], -1)  # Nx512
                # Compute cosine similarity
                cost = F.cosine_similarity(clean_outputs, outputs).mean()

            elif self.feature_loss == "pre_projection_features":
                outputs = outputs["pre_projection_features"]  # Nx197X512
                clean_outputs = clean_outputs["pre_projection_features"]  # Nx197X512

                # Reshape the outputs and clean_outputs into (N*197) x 512
                batch_size, num_tokens, feature_dim = outputs.shape  # N, 197, 512
                reshaped_outputs = outputs.view(-1, feature_dim)  # Shape: (N*197) x 512
                reshaped_clean_outputs = clean_outputs.view(-1, feature_dim)  # Shape: (N*197) x 512

                # Compute cosine similarity between each pair of 512-dimensional vectors
                cosine_similarities = F.cosine_similarity(reshaped_clean_outputs, reshaped_outputs,
                                                          dim=-1)  # Shape: (N*197)

                # Compute the mean cosine similarity across all tokens and batches
                cost = cosine_similarities.mean()  # Mean similarity across N*197 tokens

            elif self.feature_loss == "pre_projection_features_all_layers":
                # Extract all layers of outputs and clean_outputs, which are lists
                layer_outputs_all = outputs[
                    "pre_projection_features_all_layers"]  # List of tensors, each of shape Nx197x512
                layer_clean_outputs_all = clean_outputs[
                    "pre_projection_features_all_layers"]  # List of tensors, each of shape Nx197x512

                # Initialize the cost to 0
                cost = 0

                # Process all layers in parallel
                for layer_outputs, layer_clean_outputs in zip(layer_outputs_all, layer_clean_outputs_all):
                    # Reshape each layer tensor from Nx197x512 to (N*197)x512
                    reshaped_layer_outputs = layer_outputs.view(-1, layer_outputs.shape[-1])  # Shape: (N*197)x512
                    reshaped_layer_clean_outputs = layer_clean_outputs.view(-1, layer_clean_outputs.shape[
                        -1])  # Shape: (N*197)x512

                    # Compute cosine similarity for the entire reshaped layer
                    cosine_similarities = F.cosine_similarity(reshaped_layer_clean_outputs, reshaped_layer_outputs,
                                                              dim=-1)

                    # Accumulate the average cosine similarity of the layer
                    cost += cosine_similarities.mean()

                # Average the cost over all layers
                cost /= len(layer_outputs_all)

            if self.loss_fn == "cosine":
                cost = -cost

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
            print(f"  Cost {cost.item()}")
            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images