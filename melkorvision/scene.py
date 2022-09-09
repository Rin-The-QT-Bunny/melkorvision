import torch
import torch.nn as nn


class SceneParser(nn.Module):
    def __init__(self,backbone):
        super().__init__()
        self.backbone = backbone
    
    def forward(self,x):return x
