
import torch
import torch.nn as nn

from torch_geometric.nn import max_pool
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_mean
from torch_sparse import coalesce

from torch_geometric.nn.models import LabelPropagation

from abc import ABC, abstractmethod

device = "cuda:0" if torch.cuda.is_available else "cpu"

class PropModel(nn.Module):
    def __init__(self):
        self.prop = None
    def forward(self):
        return 0

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
        filtered_edges_index = add_self_loops(filtered_edges_index,
                                        num_nodes = x.size(0))[0].to(device)
        
        # Coarsen graph with filtered adj. list to produce next level's nodes
        node_labels = None

layers = 10
alpha = .9

prop = LabelPropagation(layers,0.9)