import torch
import torch.nn as nn

from torch_geometric.nn import max_pool
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_mean
from torch_sparse import coalesce

from abc import ABC, abstractmethod

device = "cuda:0" if torch.cuda.is_available else "cpu"

class AffinityAggregation(nn.Module):
    def __init__(self):
        super().__init__()