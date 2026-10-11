import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

# NodeNorm and GraphNorm

class NodeNorm(nn.Module):
    def __init__(self, scale=1.0, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(p=2, dim=-1, keepdim=True)
        return self.scale * x / (norm + self.eps)

class GraphNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias   = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, batch):
        batch_size = int(batch.max().item() + 1)
        out = torch.zeros_like(x).to(self.weight.device)

        for b in range(batch_size):
            idx = (batch == b)
            xb = x[idx]

            mean = xb.mean(dim=0, keepdim=True).to(out.device)
            var  = xb.var(dim=0, keepdim=True, unbiased=False).to(out.device)

            out[idx] = (xb - mean) / (var + self.eps).sqrt()

        return out * self.weight + self.bias


# PreNorm + Residual
    
class GNNLayerNormed(nn.Module):
    """
    Generic prenorm + conv + residual wrapper.
    Pass any `ConvClass` and its kwargs.

    `norm`: _graphnorm_ (default); _layernorm_; _nodenorm_
    """
    def __init__(self, ConvClass, in_dim, out_dim,
                 norm="graphnorm", **conv_kwargs):
        super().__init__()

        # normalization
        if norm == "graphnorm":
            self.norm = GraphNorm(in_dim)
        elif norm == "layernorm":
            self.norm = nn.LayerNorm(in_dim)
        elif norm == "nodenorm":
            self.norm = NodeNorm()
        else:
            raise ValueError("unknown norm type")

        # the conv layer
        self.conv = ConvClass(in_dim, out_dim, **conv_kwargs)

        # residual projection if needed
        self.res = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None, batch=None, **kwargs):
        
        if batch is None:
            return x

        h = self.norm(x, batch)
        out = self.conv(h, edge_index, edge_weight, **kwargs)
        return out + self.res(x)

