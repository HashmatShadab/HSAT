import logging

import torch

log = logging.getLogger(__name__)


IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)

class InputNormalize(torch.nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed
    mean and standard deviation (user-specified).
    '''
    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()
        new_std = new_std[..., None, None]
        new_mean = new_mean[..., None, None]

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean)/self.new_std
        return x_normalized


def normalize(X):
    mu = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).cuda()
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1).cuda()

    return (X - mu) / std


def clip_img_preprocessing(X):
    img_size = 224
    X = torch.nn.functional.interpolate(X, size=(img_size, img_size), mode='bicubic')
    X = normalize(X)

    return X


def convert_models_to_fp32(model):
    for n, p in model.named_parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()



d