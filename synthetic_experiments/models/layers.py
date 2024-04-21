import torch

from torch.nn.parameter import Parameter
import torch.nn.utils.parametrize as parametrize
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GRAFFConv(MessagePassing):
    def __init__(self, ext_lin, pair_lin, source_lin, self_loops=True, model_type="graffgcn"):
        super().__init__(aggr='add')
        self.ext_lin = ext_lin
        self.pair_lin = pair_lin
        self.source_lin = source_lin
        self.self_loops = self_loops
        self.model_type = model_type

    def forward(self, x, edge_index, x0):
        # (Optionally) Add self-loops to the adjacency matrix.
        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Linearly transform node feature matrix.
        out = self.pair_lin(x)

        # Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Start propagating messages.
        out = self.propagate(edge_index, x=out, norm=norm)

        if self.model_type == "graffgcn":
            return out
        elif self.model_type == "graff":
            # Add the external and source contributions
            out -= self.ext_lin(x) + self.source_lin(x0)
        else:
            raise NotImplementedError

    def message(self, x_j, norm):
        # Normalize node features.
        return norm.view(-1, 1) * x_j

class External(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((1, num_features)))
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight)

    def forward(self, x):
        return x * self.weight


class PairwiseParametrization(torch.nn.Module):
    def forward(self, W):
        # Construct a symmetric matrix with zero diagonal
        W0 = W[:, :-2].triu(1)
        W0 = W0 + W0.T

        # Retrieve the `q` and `r` vectors from the last two columns
        q = W[:, -2]
        r = W[:, -1]
        # Construct the main diagonal
        w_diag = torch.diag(q * torch.sum(torch.abs(W0), 1) + r) 

        return W0 + w_diag

class Pairwise(torch.nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        # Pay attention to the dimensions
        self.lin = torch.nn.Linear(num_hidden + 2, num_hidden, bias=False)
        # Add parametrization
        parametrize.register_parametrization(self.lin, "weight", PairwiseParametrization(), unsafe=True)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()
    
    def forward(self, x):
        return self.lin(x)
    
class Source(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(1))
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight)
    
    def forward(self, x):
        return x * self.weight