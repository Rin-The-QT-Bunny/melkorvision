
import torch
import torch.nn as nn

from torch_geometric.nn import max_pool_x
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_mean
from torch_sparse import coalesce

from torch_geometric.nn.models import LabelPropagation

from abc import ABC, abstractmethod

from moic.mklearn.nn.functional_net import FCBlock

device = "cuda:0" if torch.cuda.is_available else "cpu"

class PropModel(nn.Module):
    prop = LabelPropagation(10,0.8)

    def __init__(self,iters = 10,alpha = 0.9):
        self.prop = LabelPropagation(iters,alpha)

    @staticmethod
    def forward(self,nodes,edge_index):
        return self.prop(nodes,edge_index)

class VAE(nn.Module):
    def __init__(self,in_features,num_hidden_layers = 2,hidden_ch = 128,
    latent_features = 5,beta = 10):
        super().__init__()
        self.beta = beta # the normalization hyper parameter for the kl divergence
        self.encoder = FCBlock(hidden_ch,num_hidden_layers,in_features,2*latent_features)
        self.decoder = FCBlock(hidden_ch,num_hidden_layers,latent_features,in_features)

        # return reconstruction, reconstruction_loss and KL loss
    def forward(self,x):
        encoder_output = self.encoder(x)
        mu,logvars = encoder_output.chunk(2,dim=1)

        std = torch.exp(0.5 * logvars)
        eps = torch.randn_like(mu)
        sample = mu + ( eps * std )

        recon = self.decoder(sample) # create the reconstruction loss
        recon_loss = torch.linalg.norm(recon-x,dim=1)
        kl_loss = -0.5 * (1 + logvars - mu.pow(2) - logvars.exp())
        return {"recon":recon,"recon_loss":recon_loss,"kl_loss":self.beta * kl_loss}

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
        node_labels = PropModel(filtered_edges_index)
        cluster_labels = node_labels.unique(return_inverse=True,sorted=False)[1]

        coarsened_x, coarsened_batch = max_pool_x(cluster_labels, x, batch)
        coarsened_edge_index = coalesce(cluster_labels[filtered_edges_index],
                              None, coarsened_x.size(0), coarsened_x.size(0))[0]

        return (coarsened_x, coarsened_edge_index, coarsened_batch,
                                                         cluster_labels, losses)



print()