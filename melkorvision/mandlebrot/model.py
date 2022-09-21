import torch
import torch.nn as nn

import torch_geometric
import torch_scatter 
import torch_sparse

class AffinityLayer(nn.Module):
    def __init__(self):
        super().__init__()