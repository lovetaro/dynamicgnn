import math
import torch

from torch.nn.parameter import Parameter
import torch.nn.utils.parametrize as parametrize
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GRAFFConv(MessagePassing):
    def __init__(self, ext_lin, pair_lin, source_lin, self_loops=True):
        super().__init__(aggr='add')
        self.ext_lin = ext_lin
        self.pair_lin = pair_lin
        self.source_lin = source_lin
        self.self_loops = self_loops

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

        # Add the external and source contributions
        out -= self.ext_lin(x) + self.source_lin(x0)

        return out

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

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, model_type, output_layer=0):
        super(GraphConvolution, self).__init__()
        self.in_features, self.out_features, self.output_layer, self.model_type = (
            in_features,
            out_features,
            output_layer,
            model_type,
        )
        self.low_act, self.high_act, self.mlp_act = (
            nn.ELU(alpha=3),
            nn.ELU(alpha=3),
            nn.ELU(alpha=3),
        )
        self.att_low, self.att_high, self.att_mlp = 0, 0, 0
        if torch.cuda.is_available():
            self.weight_low, self.weight_high, self.weight_mlp = (
                Parameter(torch.FloatTensor(in_features, out_features).cuda()),
                Parameter(torch.FloatTensor(in_features, out_features).cuda()),
                Parameter(torch.FloatTensor(in_features, out_features).cuda()),
            )
            self.att_vec_low, self.att_vec_high, self.att_vec_mlp = (
                Parameter(torch.FloatTensor(out_features, 1).cuda()),
                Parameter(torch.FloatTensor(out_features, 1).cuda()),
                Parameter(torch.FloatTensor(out_features, 1).cuda()),
            )
            self.low_param, self.high_param, self.mlp_param = (
                Parameter(torch.FloatTensor(1, 1).cuda()),
                Parameter(torch.FloatTensor(1, 1).cuda()),
                Parameter(torch.FloatTensor(1, 1).cuda()),
            )
            self.attention_param = Parameter(
                torch.FloatTensor(3 * out_features, 3).cuda()
            )

            self.att_vec = Parameter(torch.FloatTensor(3, 3).cuda())

        else:
            self.weight_low, self.weight_high, self.weight_mlp = (
                Parameter(torch.FloatTensor(in_features, out_features)),
                Parameter(torch.FloatTensor(in_features, out_features)),
                Parameter(torch.FloatTensor(in_features, out_features)),
            )
            self.att_vec_low, self.att_vec_high, self.att_vec_mlp = (
                Parameter(torch.FloatTensor(out_features, 1)),
                Parameter(torch.FloatTensor(out_features, 1)),
                Parameter(torch.FloatTensor(out_features, 1)),
            )
            self.low_param, self.high_param, self.mlp_param = (
                Parameter(torch.FloatTensor(1, 1)),
                Parameter(torch.FloatTensor(1, 1)),
                Parameter(torch.FloatTensor(1, 1)),
            )
            self.attention_param = Parameter(torch.FloatTensor(3 * out_features, 3))

            self.att_vec = Parameter(torch.FloatTensor(3, 3))
        self.reset_parameters()

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight_mlp.size(1))
        std_att = 1.0 / math.sqrt(self.att_vec_mlp.size(1))
        std_att_vec = 1.0 / math.sqrt(self.att_vec.size(1))

        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)
        
        self.att_vec_high.data.uniform_(-std_att, std_att)
        self.att_vec_low.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)

        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)
        self.attention_param.data.uniform_(-std_att_vec, std_att_vec)

    def attention(self, output_low, output_high, output_mlp):
        T = 3
        att = torch.softmax(
            torch.mm(
                torch.sigmoid(
                    torch.cat(
                        [
                            torch.mm((output_low), self.att_vec_low),
                            torch.mm((output_high), self.att_vec_high),
                            torch.mm((output_mlp), self.att_vec_mlp),
                        ],
                        1,
                    )
                ),
                self.att_vec,
            )
            / T,
            1,
        )
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

    def forward(self, input, adj_low, adj_high):
        if self.model_type == "mlp":
            output_mlp = torch.mm(input, self.weight_mlp)
            return output_mlp
        elif self.model_type == "sgc" or self.model_type == "gcn":
            output_low = torch.mm(adj_low, torch.mm(input, self.weight_low))
            return output_low
        elif self.model_type == "acmgcn":
            output_low = F.relu(torch.spmm(adj_low, torch.mm(input, self.weight_low)))
            output_high = F.relu(
                torch.spmm(adj_high, torch.mm(input, self.weight_high))
            )
            output_mlp = F.relu(torch.mm(input, self.weight_mlp))

            self.att_low, self.att_high, self.att_mlp = self.attention(
                (output_low), (output_high), (output_mlp)
            )
            return 3 * (
                self.att_low * output_low
                + self.att_high * output_high
                + self.att_mlp * output_mlp
            )
        elif self.model_type == "acmsgc":
            output_low = torch.spmm(adj_low, torch.mm(input, self.weight_low))
            output_high = torch.spmm(
                adj_high, torch.mm(input, self.weight_high)
            )
            output_mlp = torch.mm(input, self.weight_mlp)

            self.att_low, self.att_high, self.att_mlp = self.attention(
                (output_low), (output_high), (output_mlp)
            )
            return 3 * (
                self.att_low * output_low
                + self.att_mlp * output_mlp
                + self.att_high * output_high
            )


# class CPLayer_v0(torch.nn.Module):
#     """
#     HO-pooling CP layer
#     """
#     def __init__(self, in_features, rank, out_features):
#         super(CPLayer_v0, self).__init__()
#         self.in_features, self.rank, self.out_features = (
#             in_features,
#             rank,
#             out_features,
#         )
#         self.agg_act, self.update_act, self.lin_act = (
#             torch.nn.Tanh(),
#             torch.nn.ReLU(),
#             torch.nn.ReLU(),
#         )
        
#         if torch.cuda.is_available():
#             self.weight_agg, self.weight_update, self.weight_lin = (
#                 Parameter(torch.FloatTensor(in_features, rank).cuda()),
#                 Parameter(torch.FloatTensor(rank, out_features).cuda()),
#                 Parameter(torch.FloatTensor(in_features, out_features).cuda()),
#             )
#         else:
#             self.weight_agg, self.weight_update, self.weight_lin = (
#                 Parameter(torch.FloatTensor(in_features, rank)),
#                 Parameter(torch.FloatTensor(rank, out_features)),
#                 Parameter(torch.FloatTensor(in_features, out_features)),
#             )
#         self.reset_parameters()

#     def __repr__(self):
#         return (
#             self.__class__.__name__
#             + " ("
#             + str(self.in_features)
#             + " -> "
#             + str(self.out_features)
#             + ")"
#         )

#     def reset_parameters(self):
#         stdv = 1.0 / math.sqrt(self.weight_agg.size(1))
#         self.weight_agg.data.uniform_(-stdv, stdv)
#         self.weight_update.data.uniform_(-stdv, stdv)
#         self.weight_lin.data.uniform_(-stdv, stdv)

#     def forward(self, input, adj):        
#         Wx = torch.mm(adj, torch.mm(input, self.weight_agg))
#         element_wise = torch.prod(Wx, dim=0, keepdim=True)
#         element_wise = self.agg_act(element_wise)
#         MWx = torch.mm(element_wise, self.weight_update)
#         MWx = self.update_act(MWx)
        
#         return MWx