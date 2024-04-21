import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn.utils.parametrize as parametrize
from torch_geometric.utils import to_edge_index

from .layers import External, Pairwise, Source, GRAFFConv

class GRAFFNet(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, self_loops=True, step_size=0.1, model_type="graffgcn", 
                 linear=True, num_layers=2, activation="tanh", normalize="none"):

        super().__init__()
        self.step_size = step_size
        self.linear = linear
        self.num_layers = num_layers
        self.activation = activation

        # Encoder
        self.enc = Linear(nfeat, nhid, bias=False)

        # Initialize the linear layers
        self.ext_lin = External(nhid)
        self.pair_lin = Pairwise(nhid)
        self.source_lin = Source()

        # Initialize the GRAFF layer
        self.conv = GRAFFConv(self.ext_lin, self.pair_lin, self.source_lin, self_loops=self_loops, model_type=model_type)

        # Decoder
        self.dec = Linear(nhid, nclass, bias=False)
        self.normalize=None
        if normalize != "none":
            if normalize=="layer":
                self.normalize = torch.nn.LayerNorm(nhid)
            elif normalize=="batch":
                self.normalize = torch.nn.BatchNorm1d(nhid)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.enc.reset_parameters()
        self.ext_lin.reset_parameters()
        self.pair_lin.reset_parameters()
        self.source_lin.reset_parameters()
        self.dec.reset_parameters()

    def forward(self, x, adj):      # adj; sparse adjacency
        # Apply the encoder
        x = self.enc(x)
        # Copy the initial features
        x0 = x.clone()
        edge_index, _ = to_edge_index(adj)

        # This context manager caches the parametrization to reduce redundant calculations
        # ODE style update
        with parametrize.cached():
            for _ in range(self.num_layers):
                if self.linear:            
                    x = x + self.step_size * (self.conv(x, edge_index, x0))
                else:
                    if self.activation=="tanh":  
                        x = x + self.step_size * F.tanh(self.conv(x, edge_index, x0))
                    elif self.activation=="relu":
                        x = x + self.step_size * F.relu(self.conv(x, edge_index, x0))
                    else:
                        raise NotImplementedError
                if self.normalize is not None:
                    x = self.normalize(x)

        # Apply the decoder
        x = self.dec(x)
        return F.log_softmax(x, dim=1)