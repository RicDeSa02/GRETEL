import torch.nn as nn
from torch_geometric.nn.aggr import MeanAggregation,SoftmaxAggregation
from torch_geometric.nn.conv import GCNConv

from src.core.factory_base import build_w_params_string
from src.utils.norms import GNNLayerNormed

class GCN(nn.Module):
   
    def __init__(self, node_features, num_conv_layers=2, conv_booster=1, pooling=MeanAggregation, use_weights=True):
        super(GCN, self).__init__()
        
        self.in_channels = node_features
        self.out_channels = int(self.in_channels * conv_booster)
          
        self.pooling =  build_w_params_string(pooling)
 
        
        if num_conv_layers>1:
            self.num_conv_layers = [(self.in_channels, self.out_channels)] + [(self.out_channels, self.out_channels) * (num_conv_layers - 1)]
        else:
            self.num_conv_layers = [(self.in_channels, self.out_channels)]
        self.graph_convs = self.__init__conv_layers(use_weights)
        
    def forward(self, node_features, edge_index, edge_weight, batch):
        # convolution operations
        edge_index = edge_index.long()
        for conv_layer in self.graph_convs[:-1]:
            node_features = conv_layer(node_features, edge_index, edge_weight, batch)
            node_features = nn.functional.relu(node_features)

        # global pooling
        if isinstance(self.graph_convs[-1],nn.Identity):
            return self.graph_convs[-1](node_features)
        return self.graph_convs[-1](node_features, batch)
    
    def __init__conv_layers(self, use_weights):
        ############################################
        # initialize the convolutional layers interleaved with pooling layers
        graph_convs = []
        for i in range(len(self.num_conv_layers)):#add len
            if use_weights:
                # graph_convs.append(GCNConvWithEdgeTracking(in_channels=self.num_conv_layers[i][0],
                #                       out_channels=self.num_conv_layers[i][1],
                #                       add_self_loops=True).double())
                graph_convs.append(GNNLayerNormed(
                                    GCNConvWithEdgeTracking,
                                    in_dim=self.num_conv_layers[i][0],
                                    out_dim=self.num_conv_layers[i][1],
                                    norm="graphnorm",
                                    add_self_loops=True).double())
            else:
                graph_convs.append(GCNConv(in_channels=self.num_conv_layers[i][0],
                                      out_channels=self.num_conv_layers[i][1],
                                      add_self_loops=True).double())
        graph_convs.append(self.pooling)
        return nn.Sequential(*graph_convs).double()

class GCNConvWithEdgeTracking(GCNConv):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.residual = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        out = super().forward(x, edge_index, edge_weight=edge_weight)
        # if edge_weight is not None:
        #     src = edge_index[0] # source nodes
        #     self.last_messages = x[src] * edge_weight.unsqueeze(1)
        return out + self.residual(x)
