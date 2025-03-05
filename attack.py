# PGD attack model

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttackPGD(nn.Module):
    def __init__(self, model, criterion, config):
        super(AttackPGD, self).__init__()
        self.model = model
        self.criteria = criterion
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']


    def forward(self, inputs, targets, train=True):
        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        if train:
            num_step = 10
        else:
            num_step = 20
        for i in range(num_step):
            x.requires_grad_()
            with torch.enable_grad():
                features = self.model(x)
                logits = self.classifier(features)
                loss = F.cross_entropy(logits, targets, size_average=False)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon),
                          inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)
        features = self.model(x)
        return self.classifier(features), x
