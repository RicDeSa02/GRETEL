import torch.nn as nn
from torch_geometric.nn.aggr import MeanAggregation,SoftmaxAggregation
from torch_geometric.nn.conv import GCNConv

from src.core.factory_base import build_w_params_string

from src.utils.norms import GNNLayerNormed
from src.utils.torch.gcn import GCNConvWithEdgeTracking

class GCN_Multiclass(nn.Module):
   
    def __init__(self, node_features, num_conv_layers=2, conv_booster=1, pooling=MeanAggregation, use_weights=True):
        super(GCN_Multiclass, self).__init__()
        
        self.in_channels = node_features
        self.out_channels = int(self.in_channels * conv_booster)
          
        self.pooling =  build_w_params_string(pooling)
        
        if num_conv_layers>1:
            self.num_conv_layers = [(self.in_channels, self.out_channels)] + [(self.out_channels, self.out_channels) for _ in range (num_conv_layers - 1)]
        else:
            self.num_conv_layers = [(self.in_channels, self.out_channels)]
        self.graph_convs = self.__init__conv_layers(use_weights)

        '''self.classifier = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels, num_classes)
        ).double()'''
        
    def forward(self, node_features, edge_index, edge_weight, batch):
        # convolution operations
        for i,conv_layer in enumerate(self.graph_convs[:-1]):
            node_features = conv_layer(node_features, edge_index, edge_weight, batch)
            # print(i, node_features.shape)
            node_features = nn.functional.relu(node_features)

        # global pooling
        if isinstance(self.graph_convs[-1],nn.Identity):
            pooled_feat = self.graph_convs[-1](node_features)
        pooled_feat = self.graph_convs[-1](node_features, batch)

        # print("pooled_feat.shape: ", pooled_feat.shape)
        return pooled_feat # self.classifier(pooled_feat)

    
    def __init__conv_layers(self, use_weights):
        ############################################
        # initialize the convolutional layers interleaved with pooling layers
        graph_convs = []
        for i in range(len(self.num_conv_layers)):#add len
            if use_weights:
                graph_convs.append(GNNLayerNormed(
                                        GCNConvWithEdgeTracking,
                                        in_dim=self.num_conv_layers[i][0],
                                        out_dim=self.num_conv_layers[i][1],
                                        norm="graphnorm",
                                        add_self_loops=True).double())
            else:
                graph_convs.append(GCNConv(in_channels=self.num_conv_layers[i][0],
                                      out_channels=self.num_conv_layers[i][1]).double())
        graph_convs.append(self.pooling)
        return nn.Sequential(*graph_convs).double()