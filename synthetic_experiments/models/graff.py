import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn.utils.parametrize as parametrize
from torch_geometric.utils import to_edge_index

from .layers import External, Pairwise, Source, GRAFFConv

class GRAFFNet(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, self_loops=True, step_size=1.):
        super().__init__()
        self.step_size = step_size

        # Encoder
        self.enc = Linear(nfeat, nhid, bias=False)

        # Initialize the linear layers
        self.ext_lin = External(nhid)
        self.pair_lin = Pairwise(nhid)
        self.source_lin = Source()

        # Initialize the GRAFF layer
        self.conv = GRAFFConv(self.ext_lin, self.pair_lin, self.source_lin, self_loops=self_loops)

        # Decoder
        self.dec = Linear(nhid, nclass, bias=False)

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
            x = x + self.step_size * F.relu(self.conv(x, edge_index, x0))
            x = x + self.step_size * F.relu(self.conv(x, edge_index, x0))

        # Apply the decoder
        x = self.dec(x)
        return F.log_softmax(x, dim=1)