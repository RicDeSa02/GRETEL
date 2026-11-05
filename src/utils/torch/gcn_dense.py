import torch
import torch.nn as nn


# dense helpers (replacing PyG gather/scatter)
def normalize_adj(A, mask):
    """A: [B,n,n], mask: [B,n]. Same as gcn_norm with add_self_loops=True. Computed ONCE per forward."""
    A = A * mask.unsqueeze(1) * mask.unsqueeze(2)
    A = A + torch.diag_embed(mask)
    dinv = A.sum(-1).clamp(min=1e-12).pow(-0.5) * mask
    return dinv.unsqueeze(-1) * A * dinv.unsqueeze(-2)

def masked_mean_pool(x, mask):  # replaces MeanAggregation(x, batch)
    return (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)

class DenseGraphNorm(nn.Module):  # masked GraphNorm
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.mean_scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x, mask):
        m = mask.unsqueeze(-1); cnt = m.sum(1, keepdim=True)
        out = (x - self.mean_scale * (x * m).sum(1, keepdim=True) / cnt) * m
        var = (out ** 2).sum(1, keepdim=True) / cnt
        return self.weight * out / (var + self.eps).sqrt() + self.bias

class DenseGCNConvWithResidual(nn.Module):  # replaces GCNConvWithEdgeTracking
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.residual = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels)
    def forward(self, x, A_norm):
        return A_norm @ self.lin(x) + self.bias + self.residual(x)

class DenseGCNLayer(nn.Module): # conv + norm (stands in for GNNLayerNormed)
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = DenseGCNConvWithResidual(in_dim, out_dim)
        self.norm = DenseGraphNorm(out_dim)
    def forward(self, x, A_norm, mask):
        return self.norm(self.conv(x, A_norm), mask)

class GCN(nn.Module):
    def __init__(self, node_features, num_conv_layers=2, conv_booster=1, pooling=None, use_weights=True):
        super().__init__()
        self.in_channels = node_features
        self.out_channels = int(self.in_channels * conv_booster)
        # CHANGED: fixed tuple bug ( (o,o)*k -> [(o,o)]*k ); pooling is masked mean only
        self.num_conv_layers = [(self.in_channels, self.out_channels)] + \
                               [(self.out_channels, self.out_channels)] * (num_conv_layers - 1)
        self.graph_convs = nn.ModuleList(DenseGCNLayer(i, o) for i, o in self.num_conv_layers)  # CHANGED

    def forward(self, x, A, mask):           # CHANGED signature: x [B,n,f], A [B,n,n], mask [B,n]
        A_norm = normalize_adj(A, mask)      # NEW: once, reused by all layers
        for layer in self.graph_convs:
            x = torch.relu(layer(x, A_norm, mask))
        return masked_mean_pool(x, mask)     # CHANGED