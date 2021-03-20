import torch


class HeroVillainNet(torch.nn.Module):
    def __init__(self):
        super(HeroVillainNet, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
