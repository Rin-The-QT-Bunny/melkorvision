
import torch
import torch.nn as nn

from torch_geometric.nn import max_pool_x,GraphConv
from torch_geometric.data import Data,Batch
from torch_geometric.utils import add_self_loops,grid,to_dense_batch
from torch_scatter import scatter_mean
from torch_sparse import coalesce

from types import SimpleNamespace

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



class P2AffinityAggregation(AffinityAggregation):
    def __init__(self,node_feature_dim,v2=3.5):
        super().__init__()
        self.v2 = v2
        self.node_pair_vae = VAE(in_features = 2 * node_feature_dim)

    # we assume the difference between nodes are close to each other

    def forward(self,x,row,col):
        # affinities as function of vae reconstruction of node pairs
        vae_outputs = self.node_pair_vae(torch.cat([x[row],x[col]],dim=1))
        recon_loss,kl_loss = vae_outputs["recon_loss"],vae_outputs["kl_loss"]
        edge_affinities = 1 / (1 + self.v2 * recon_loss)

        losses = {"recon_loss":recon_loss.mean(),"kl_loss":kl_loss.mean()}
    
        return edge_affinities,0.5,losses

class P1AffinityAggregation(AffinityAggregation):
    def forward(self,x,row,col):
        # Norm of difference for every node pair on grid
        # similarity calculated with no grad
        edge_affinities = torch.linalg.norm(x[row]-x[col],dim=1).to(x.device)
        inv_mean_affinities = 1/scatter_mean(edge_affinities,row.to(x.device))
        affinity_thresh  = torch.min(inv_mean_affinities[row],inv_mean_affinities[col])

        return edge_affinities,affinity_thresh,{} 

class PSGNetLite(nn.Module):
    def __init__(self,imsize):
        super().__init__()
        
        node_feature_dim = 32
        num_graph_layer = 2

        self.spatial_edges,self.spatial_coords = grid(imsize,imsize,device=device)

        self.rdn = []

        # Affinity modules: for now there is just P1 and P2 net
        self.affinity_aggregations = torch.nn.ModuleList([
            P1AffinityAggregation(),P2AffinityAggregation(node_feature_dim)
        ])

        # Node tranforms: function applied on aggregated node vectors
        self.node_transforms = torch.nn.ModuleList([
            FCBlock(132,2,in_features=node_feature_dim,features=node_feature_dim)
        ])

        # Graph convolutional layers to apply after each graph coarsening
        self.graph_convs = torch.nn.ModuleList([
            GraphConv(node_feature_dim,node_feature_dim)
            for _ in range(len(self.affinity_aggregations))
        ])

        # The render of the final output nodes
        self.node_to_rgb = FCBlock(132,4,node_feature_dim,3)

    def forward(self,img):
        # Collect image features with rdn
        im_features = self.rdn(img.permute([0,3,1,2]))

        coords_added_im_feats = torch.cat([
            self.spatial_coords.unsqueeze(0).repeat(im_features.size(0),1,1),
            im_features.flatten(2,3).permute(0,2,1)
        ],dim=2)

        ### Rum image feature graph through affinity modules
        graph_in = Batch.from_data_list([
            Data(x,self.spatial_edges) for x in coords_added_im_feats
        ])

        x, edge_index, batch = graph_in.x, graph_in.edge_index, graph_in.batch

        clusters,all_losses = [],[] # clusters used only for decoration

        for pool, conv, transf in zip(self.affinity_aggregations,
                                      self.graph_convs, self.node_transforms):
            batch_uncoarsened = batch

            x, edge_index, batch, cluster, losses = pool(x, edge_index, batch)
            x = conv(x, edge_index)
            x = transf(x)

            clusters.append( (cluster, batch_uncoarsened) )
            all_losses.append(losses)

        # Render into image (just constant coloring from each cluster for now)
        # First map each node to pixel and then uncluster pixels

        nodes_rgb = self.node_to_rgb(x)
        img_out   = nodes_rgb
        for cluster,_ in reversed(clusters): img_out = img_out[cluster]

        return to_dense_batch(img_out,graph_in.batch)[0], clusters, all_losses