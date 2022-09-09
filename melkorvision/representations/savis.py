import torch
import torch.nn as nn

import torchvision


import moic.mklearn.nn.visual_net as visual_net
import moic.mklearn.nn.functional_net as func_net


class SAVIS(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):return x