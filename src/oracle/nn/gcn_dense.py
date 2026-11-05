import torch.nn as nn
from torch_geometric.nn.aggr import MeanAggregation
from torch_geometric.utils import to_dense_adj, to_dense_batch

from src.utils.torch.gcn_dense import GCN


class DownstreamGCN(GCN):
   
    def __init__(self, node_features,
                 n_classes=2,
                 num_conv_layers=2,
                 num_dense_layers=2,
                 conv_booster=2,
                 linear_decay=2,
                 pooling=MeanAggregation(),
                 use_weights=True):

        
        super().__init__(node_features, num_conv_layers, conv_booster, pooling, use_weights)
        
        self.num_dense_layers = num_dense_layers
        self.linear_decay = linear_decay
        self.n_classes = n_classes
        
        self.downstream_layers = self.__init__downstream_layers()
        
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, node_features, edge_index, edge_weight, batch):   # old signature kept
        x, mask = to_dense_batch(node_features, batch)                  # [B,n,f], [B,n]
        A = to_dense_adj(edge_index.long(), batch, edge_attr=edge_weight,
                         max_num_nodes=x.size(1))                       # [B,n,n]
        return self.forward_dense(x, A, mask.float())

    def forward_dense(self, x, A, mask):                                # fast path, new code calls this
        return self.downstream_layers(GCN.forward(self, x, A, mask))
    
    def __init__downstream_layers(self):
        ############################################
        # initialize the linear layers interleaved with activation functions
        downstream_layers = []
        in_linear = self.out_channels
        for _ in range(self.num_dense_layers-1):
            downstream_layers.append(nn.Linear(in_linear, int(in_linear // self.linear_decay)))
            downstream_layers.append(nn.ReLU())
            in_linear = int(in_linear // self.linear_decay)
        # add the output layer
        downstream_layers.append(nn.Linear(in_linear, self.n_classes))
        #downstream_layers.append(nn.Sigmoid())
        #downstream_layers.append(nn.Softmax())
        # put the linear layers in sequential
        return nn.Sequential(*downstream_layers).float()