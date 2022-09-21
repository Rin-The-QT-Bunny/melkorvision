
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

    @abstractmethod
    def affinities_and_thresholds(self,x,row,col):
        pass

    # Filter edges index based on method's affinity thresholding
    # and coarse graph to produce next level's nodes
    def forward(self,x,edge_index,batch,device=device):
        row, col = edge_index

        # Collect affinities/threholds to filter edges
        affinities,threshold,losses = self.affinities_and_thresholds(x,row,col)
        filtered_edges_index = edge_index[:,affinities>threshold]